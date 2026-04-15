import math

import jax
import jax.numpy as jnp
import numpy as np

from zfel import sase1d


jax.config.update("jax_enable_x64", True)


BASE_INPUT = dict(
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


def j0_series(x, n_terms=20):
    acc = jnp.zeros_like(x)
    for m in range(n_terms):
        acc = acc + ((-1.0) ** m) / (math.factorial(m) ** 2) * (x**2 / 4.0) ** m
    return acc


def j1_series(x, n_terms=20):
    acc = jnp.zeros_like(x)
    for m in range(n_terms):
        acc = acc + ((-1.0) ** m) / (math.factorial(m) * math.factorial(m + 1)) * (x / 2.0) ** (2 * m + 1)
    return acc


def inv_sigmoid(y):
    y = jnp.clip(y, 1e-6, 1.0 - 1e-6)
    return jnp.log(y / (1.0 - y))


def params_from_k_profile(k_profile, base_params, base_input):
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
    unduJJ = j0_series(x) - j1_series(x)

    gamma0 = energy / mc2
    sigmaX2 = emitN * beta / gamma0
    kappa_1 = e * k_profile * unduJJ / 4 / epsilon_0 / gamma0
    Kai = e * k_profile * unduJJ / (2 * gamma0**2 * mc2 * e)
    density = currentMax / (e * c * 2 * jnp.pi * sigmaX2)

    resWavelength = unduPeriod * (1 + k_profile[0] ** 2 / 2.0) / (2 * gamma0**2)
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
            "coopLength": resWavelength / unduPeriod,
            "delt": delt,
            "dels": delt,
            "E02": jnp.asarray(0.0),
            "gbar": jnp.asarray(0.0),
            "Ns": currentMax * z0 / unduPeriod / z_steps * resWavelength / c / e,
            "deta": jnp.sqrt((1 + 0.5 * k_profile[0] ** 2) / (1 + 0.5 * k_profile**2)) - 1,
        }
    )
    return params


def pulse_energy_like(output):
    return output["power_s"][-1].sum()


def main():
    base_params_np = sase1d.params_calc(**BASE_INPUT)
    base_params = {k: jnp.asarray(v) for k, v in base_params_np.items()}
    noise_spec = sase1d.make_shot_noise_spec_from_params(
        base_params_np,
        npart=BASE_INPUT["npart"],
        s_steps=BASE_INPUT["s_steps"],
        iopt=BASE_INPUT["iopt"],
    )

    n_steps = BASE_INPUT["z_steps"]
    K0 = jnp.asarray(BASE_INPUT["unduK"], dtype=jnp.float64)
    z_grid = jnp.linspace(0.0, 1.0, n_steps)

    def theta_to_k_profile(theta):
        # theta = [start_logit, linear_logit, quadratic_logit]
        start_frac = 0.20 + 0.40 * jax.nn.sigmoid(theta[0])
        linear = 0.03 * jax.nn.sigmoid(theta[1])
        quadratic = 0.10 * jax.nn.sigmoid(theta[2])

        tail_span = jnp.maximum(1.0 - start_frac, 1e-6)
        u = jnp.clip((z_grid - start_frac) / tail_span, 0.0, 1.0)
        k_profile = K0 * (1.0 - linear * u - quadratic * u**2)
        return k_profile

    def eval_profile(k_profile, noise_batch):
        params = params_from_k_profile(k_profile, base_params, BASE_INPUT)
        energies = jax.vmap(
            lambda noise_i: pulse_energy_like(sase1d.FEL_sim_jax(params, noise_i, noise_spec))
        )(noise_batch)
        return energies

    def batch_objective(theta, noise_batch, lambda_var=0.10, lambda_mono=2e3):
        k_profile = theta_to_k_profile(theta)
        energies = eval_profile(k_profile, noise_batch)
        base_energies = eval_profile(jnp.full_like(k_profile, K0), noise_batch)

        mean_e = jnp.mean(energies)
        std_e = jnp.std(energies)
        mean_base = jnp.mean(base_energies)

        dK = k_profile[1:] - k_profile[:-1]
        mono_pen = jnp.mean(jnp.maximum(dK, 0.0) ** 2)

        normalized_mean = mean_e / (mean_base + 1e-12)
        normalized_std = std_e / (mean_base + 1e-12)
        obj = normalized_mean - lambda_var * normalized_std - lambda_mono * mono_pen
        return obj, (mean_e, std_e, mean_base, k_profile)

    objective_and_grad = jax.value_and_grad(batch_objective, has_aux=True)

    train_batch_size = 2
    val_batch_size = 4
    refresh_every = 2
    n_iter = 4

    val_noise = sase1d.sample_shot_noise_batch_jax(jax.random.PRNGKey(999), noise_spec, val_batch_size)

    ref_theta = jnp.array(
        [
            inv_sigmoid((80.0 / (n_steps - 1) - 0.20) / 0.40),
            inv_sigmoid(1e-4 / 0.03),
            inv_sigmoid(0.06 / 0.10),
        ],
        dtype=jnp.float64,
    )
    theta = ref_theta

    m = jnp.zeros_like(theta)
    v = jnp.zeros_like(theta)
    lr = 0.03
    b1, b2 = 0.9, 0.999
    eps = 1e-8

    def summarize(theta_eval, noise_batch):
        k_profile = theta_to_k_profile(theta_eval)
        energies = eval_profile(k_profile, noise_batch)
        base_energies = eval_profile(jnp.full_like(k_profile, K0), noise_batch)
        return {
            "ratio": float(jnp.mean(energies) / jnp.mean(base_energies)),
            "mean": float(jnp.mean(energies)),
            "std": float(jnp.std(energies)),
            "base_mean": float(jnp.mean(base_energies)),
            "k_profile": np.asarray(k_profile),
        }

    ref_summary = summarize(ref_theta, val_noise)
    print("Stochastic SASE reference setup")
    print(f"  validation no-taper mean pulse: {ref_summary['base_mean']:.6e}")
    print(f"  reference taper mean pulse:     {ref_summary['mean']:.6e}")
    print(f"  reference taper std:            {ref_summary['std']:.6e}")
    print(f"  reference taper ratio:          {ref_summary['ratio']:.3f}x")
    print()

    best = {
        "iter": 0,
        **ref_summary,
    }

    train_key = jax.random.PRNGKey(123)
    for t in range(1, n_iter + 1):
        if (t - 1) % refresh_every == 0:
            train_key, subkey = jax.random.split(train_key)
            train_noise = sase1d.sample_shot_noise_batch_jax(subkey, noise_spec, train_batch_size)

        ((obj, (mean_e, std_e, mean_base, _)), g) = objective_and_grad(theta, train_noise)
        g_norm = jnp.linalg.norm(g)
        g = g / jnp.maximum(g_norm, 1.0)

        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * (g * g)
        m_hat = m / (1 - b1**t)
        v_hat = v / (1 - b2**t)
        theta = theta + lr * m_hat / (jnp.sqrt(v_hat) + eps)

        if t in [1, 2, 3, 4]:
            val_summary = summarize(theta, val_noise)
            print(
                f"iter={t:02d} obj={float(obj):.4f} train_ratio={float(mean_e / mean_base):.3f}x "
                f"val_ratio={val_summary['ratio']:.3f}x val_mean={val_summary['mean']:.6e}"
            )
            if val_summary["ratio"] > best["ratio"]:
                best = {
                    "iter": t,
                    **val_summary,
                }

    print()
    print("Best stochastic expected-value taper")
    print(f"  best iteration:                {best['iter']}")
    print(f"  validation mean pulse:         {best['mean']:.6e}")
    print(f"  validation pulse std:          {best['std']:.6e}")
    print(f"  validation no-taper mean:      {best['base_mean']:.6e}")
    print(f"  validation ratio:              {best['ratio']:.3f}x")
    print(f"  min(K):                        {float(np.min(best['k_profile'])):.3f}")
    print(f"  max(K):                        {float(np.max(best['k_profile'])):.3f}")


if __name__ == "__main__":
    main()
