import math
import numpy as np

from zfel import sase1d


def _default_input():
    return dict(
        npart=512,
        s_steps=200,
        z_steps=200,
        energy=4313.34e6,
        eSpread=0,
        emitN=1.2e-6,
        currentMax=3900,
        beta=26,
        unduPeriod=0.03,
        unduK=np.full(200, 3.5),
        unduL=70,
        radWavelength=None,
        random_seed=31,
        particle_position=np.genfromtxt("docs/examples/data/SASE_particle_position.csv", delimiter=","),
        hist_rule="square-root",
        iopt="sase",
    )


def _j0_series(x, n_terms=20):
    import jax.numpy as jnp

    acc = jnp.zeros_like(x)
    for m in range(n_terms):
        acc = acc + ((-1.0) ** m) / (math.factorial(m) ** 2) * (x**2 / 4.0) ** m
    return acc


def _j1_series(x, n_terms=20):
    import jax.numpy as jnp

    acc = jnp.zeros_like(x)
    for m in range(n_terms):
        acc = acc + ((-1.0) ** m) / (math.factorial(m) * math.factorial(m + 1)) * (x / 2.0) ** (2 * m + 1)
    return acc


def _k_taper(k0=3.5, a=0.06, n=200, split_ix=80):
    u = np.linspace(0.0, 1.0, n - split_ix)
    return np.hstack([np.ones(split_ix), (1.0 - a * u**2)]) * k0


def _build_interp_matrix(n_steps, n_ctrl):
    import jax.numpy as jnp

    z_ctrl = np.linspace(0, n_steps - 1, n_ctrl)
    W = np.zeros((n_steps, n_ctrl))
    for z_i in range(n_steps):
        if z_i <= z_ctrl[0]:
            W[z_i, 0] = 1.0
        elif z_i >= z_ctrl[-1]:
            W[z_i, -1] = 1.0
        else:
            j = np.searchsorted(z_ctrl, z_i) - 1
            t = (z_i - z_ctrl[j]) / (z_ctrl[j + 1] - z_ctrl[j])
            W[z_i, j] = 1.0 - t
            W[z_i, j + 1] = t
    return jnp.asarray(W)


def _params_from_k_profile(k_profile, base_params, base_input):
    import jax.numpy as jnp

    mc2 = sase1d.mc2
    c = sase1d.c
    e = sase1d.e
    epsilon_0 = sase1d.epsilon_0

    energy = jnp.asarray(base_input["energy"])
    emitN = jnp.asarray(base_input["emitN"])
    currentMax = jnp.asarray(base_input["currentMax"])
    beta = jnp.asarray(base_input["beta"])
    unduPeriod = jnp.asarray(base_input["unduPeriod"])

    x = k_profile**2 / (4 + 2 * k_profile**2)
    unduJJ = _j0_series(x) - _j1_series(x)

    gamma0 = energy / mc2
    sigmaX2 = emitN * beta / gamma0
    kappa_1 = e * k_profile * unduJJ / 4 / epsilon_0 / gamma0
    Kai = e * k_profile * unduJJ / (2 * gamma0**2 * mc2 * e)
    density = currentMax / (e * c * 2 * jnp.pi * sigmaX2)

    resWavelength = unduPeriod * (1 + k_profile[0] ** 2 / 2.0) / (2 * gamma0**2)
    Pbeam = energy * currentMax
    coopLength = resWavelength / unduPeriod
    z0 = jnp.asarray(base_input["unduL"])
    z_steps = k_profile.shape[0]
    delt = z0 / z_steps

    params = dict(base_params)
    params.update(
        {
            "kappa_1": kappa_1,
            "Kai": Kai,
            "density": density,
            "resWavelength": resWavelength,
            "coopLength": coopLength,
            "delt": delt,
            "dels": delt,
            "E02": jnp.asarray(0.0),
            "gbar": jnp.asarray(0.0),
            "Ns": currentMax * z0 / unduPeriod / z_steps * resWavelength / c / e,
            "deta": jnp.sqrt((1 + 0.5 * k_profile[0] ** 2) / (1 + 0.5 * k_profile**2)) - 1,
        }
    )
    return params


def test_gradient_taper_optimizer_reaches_minimum_ratio():
    try:
        import jax
        import jax.numpy as jnp
    except ImportError:
        return

    jax.config.update("jax_enable_x64", True)

    base_input = _default_input()
    base_params_np = sase1d.params_calc(**base_input)
    base_params = {k: jnp.asarray(v) for k, v in base_params_np.items()}

    bucket_np = sase1d.fixed_or_external_bucket_data(
        params=base_params_np,
        npart=base_input["npart"],
        s_steps=base_input["s_steps"],
        particle_position=base_input["particle_position"],
        hist_rule=base_input["hist_rule"],
        iopt=base_input["iopt"],
        random_seed=base_input["random_seed"],
    )
    bucket = {
        "thet_init": jnp.asarray(bucket_np["thet_init"]),
        "eta_init": jnp.asarray(bucket_np["eta_init"]),
        "N_real": jnp.asarray(bucket_np["N_real"]),
        "s_steps": int(bucket_np["s_steps"]),
    }

    n_steps = base_input["z_steps"]
    W = _build_interp_matrix(n_steps, n_ctrl=8)

    K_min = 3.15
    K_max = 3.5
    K_ref = jnp.asarray(_k_taper(a=0.06, n=n_steps, split_ix=80), dtype=jnp.float64)
    K0 = jnp.asarray(np.full(n_steps, 3.5), dtype=jnp.float64)

    def inv_sigmoid(y):
        y = jnp.clip(y, 1e-6, 1 - 1e-6)
        return jnp.log(y / (1.0 - y))

    K_ref_ctrl = np.asarray(W.T @ K_ref / np.sum(np.asarray(W.T), axis=1))
    theta = inv_sigmoid((jnp.asarray(K_ref_ctrl) - K_min) / (K_max - K_min))

    def theta_to_k_profile(theta_local):
        K_ctrl = K_min + (K_max - K_min) * jax.nn.sigmoid(theta_local)
        return W @ K_ctrl

    def pulse_energy_like(output):
        return output["power_s"][-1].sum()

    def objective(theta_local, lambda_smooth=5e2, lambda_mono=5e3):
        k_profile = theta_to_k_profile(theta_local)
        params = _params_from_k_profile(k_profile, base_params, base_input)
        out = sase1d.sase_from_initial_conditions_jax(params, bucket)
        pulse = pulse_energy_like(out)
        smooth_pen = jnp.mean((k_profile[2:] - 2 * k_profile[1:-1] + k_profile[:-2]) ** 2)
        mono_pen = jnp.mean(jnp.maximum(k_profile[1:] - k_profile[:-1], 0.0) ** 2)
        return jnp.log(pulse) - lambda_smooth * smooth_pen - lambda_mono * mono_pen, pulse

    base_out = sase1d.sase_from_initial_conditions_jax(
        _params_from_k_profile(K0, base_params, base_input), bucket
    )
    base_pulse = float(pulse_energy_like(base_out))

    best_ratio = 0.0
    m = jnp.zeros_like(theta)
    v = jnp.zeros_like(theta)
    lr = 0.08
    b1, b2 = 0.9, 0.999
    eps = 1e-8

    for t in range(1, 6):
        (obj, pulse), g = jax.value_and_grad(objective, has_aux=True)(theta)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * (g * g)
        m_hat = m / (1 - b1**t)
        v_hat = v / (1 - b2**t)
        theta = theta + lr * m_hat / (jnp.sqrt(v_hat) + eps)
        best_ratio = max(best_ratio, float(pulse) / base_pulse)

    assert best_ratio >= 10.0
