import math

import numpy as np
import pytest

from zfel import sase1d


def _default_input():
    return dict(
        npart=512,
        s_steps=200,
        z_steps=200,
        energy=4313.34e6,
        eSpread=0.0,
        emitN=1.2e-6,
        currentMax=3900,
        beta=26,
        unduPeriod=0.03,
        unduK=3.5,
        unduL=70,
        radWavelength=None,
        random_seed=31,
        particle_position=None,
        hist_rule="square-root",
        iopt="sase",
        P0=0.0,
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

    res_wavelength = unduPeriod * (1 + k_profile[0] ** 2 / 2.0) / (2 * gamma0**2)
    z0 = jnp.asarray(base_input["unduL"])
    z_steps = k_profile.shape[0]
    delt = z0 / z_steps

    params = dict(base_params)
    params.update(
        {
            "kappa_1": kappa_1,
            "Kai": Kai,
            "density": density,
            "resWavelength": res_wavelength,
            "coopLength": res_wavelength / unduPeriod,
            "delt": delt,
            "dels": delt,
            "E02": jnp.asarray(0.0),
            "gbar": jnp.asarray(0.0),
            "Ns": currentMax * z0 / unduPeriod / z_steps * res_wavelength / c / e,
            "deta": jnp.sqrt((1 + 0.5 * k_profile[0] ** 2) / (1 + 0.5 * k_profile**2)) - 1,
        }
    )
    return params


@pytest.mark.slow
def test_expected_sase_optimizer_reaches_minimum_ratio():
    try:
        import jax
        import jax.numpy as jnp
    except ImportError:
        return

    jax.config.update("jax_enable_x64", True)

    base_input = _default_input()
    base_params_np = sase1d.params_calc(**base_input)
    base_params = {k: jnp.asarray(v) for k, v in base_params_np.items()}
    noise_spec = sase1d.make_shot_noise_spec_from_params(
        base_params_np,
        npart=base_input["npart"],
        s_steps=base_input["s_steps"],
        iopt=base_input["iopt"],
    )

    n_steps = base_input["z_steps"]
    K0 = jnp.asarray(base_input["unduK"], dtype=jnp.float64)
    z_grid = jnp.linspace(0.0, 1.0, n_steps)

    def inv_sigmoid(y):
        y = jnp.clip(y, 1e-6, 1.0 - 1e-6)
        return jnp.log(y / (1.0 - y))

    def theta_to_k_profile(theta):
        start_frac = 0.20 + 0.40 * jax.nn.sigmoid(theta[0])
        linear = 0.03 * jax.nn.sigmoid(theta[1])
        quadratic = 0.10 * jax.nn.sigmoid(theta[2])
        tail_span = jnp.maximum(1.0 - start_frac, 1e-6)
        u = jnp.clip((z_grid - start_frac) / tail_span, 0.0, 1.0)
        return K0 * (1.0 - linear * u - quadratic * u**2)

    def pulse_energy_like(output):
        return output["power_s"][-1].sum()

    def eval_profile(k_profile, noise_batch):
        params = _params_from_k_profile(k_profile, base_params, base_input)
        return jax.vmap(
            lambda noise_i: pulse_energy_like(sase1d.FEL_sim_jax(params, noise_i, noise_spec))
        )(noise_batch)

    def objective(theta, noise_batch, lambda_var=0.10, lambda_mono=2e3):
        k_profile = theta_to_k_profile(theta)
        energies = eval_profile(k_profile, noise_batch)
        base_energies = eval_profile(jnp.full_like(k_profile, K0), noise_batch)
        mean_e = jnp.mean(energies)
        std_e = jnp.std(energies)
        mean_base = jnp.mean(base_energies)
        mono_pen = jnp.mean(jnp.maximum(k_profile[1:] - k_profile[:-1], 0.0) ** 2)
        score = mean_e / (mean_base + 1e-12) - lambda_var * std_e / (mean_base + 1e-12) - lambda_mono * mono_pen
        return score, (mean_e, mean_base)

    theta = jnp.array(
        [
            inv_sigmoid((80.0 / (n_steps - 1) - 0.20) / 0.40),
            inv_sigmoid(1e-4 / 0.03),
            inv_sigmoid(0.06 / 0.10),
        ],
        dtype=jnp.float64,
    )

    train_noise = sase1d.sample_shot_noise_batch_jax(jax.random.PRNGKey(123), noise_spec, 2)
    val_noise = sase1d.sample_shot_noise_batch_jax(jax.random.PRNGKey(999), noise_spec, 4)

    ref_k = theta_to_k_profile(theta)
    ref_ratio = float(jnp.mean(eval_profile(ref_k, val_noise)) / jnp.mean(eval_profile(jnp.full_like(ref_k, K0), val_noise)))

    best_ratio = ref_ratio
    m = jnp.zeros_like(theta)
    v = jnp.zeros_like(theta)
    lr = 0.03
    b1, b2 = 0.9, 0.999
    eps = 1e-8

    for t in range(1, 3):
        (_, _), g = jax.value_and_grad(objective, has_aux=True)(theta, train_noise)
        g_norm = jnp.linalg.norm(g)
        g = g / jnp.maximum(g_norm, 1.0)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * (g * g)
        m_hat = m / (1 - b1**t)
        v_hat = v / (1 - b2**t)
        theta = theta + lr * m_hat / (jnp.sqrt(v_hat) + eps)

        k_profile = theta_to_k_profile(theta)
        ratio = float(jnp.mean(eval_profile(k_profile, val_noise)) / jnp.mean(eval_profile(jnp.full_like(k_profile, K0), val_noise)))
        best_ratio = max(best_ratio, ratio)

    assert best_ratio >= 12.5
