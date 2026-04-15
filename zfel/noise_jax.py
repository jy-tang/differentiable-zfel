"""JAX-native shot-noise helpers for Monte Carlo expectation objectives."""

import jax
import jax.numpy as jnp


def sample_shot_noise_jax(key, spec):
    """Sample explicit shot noise for one realization."""
    s_steps = int(spec["s_steps"])
    nb = int(spec["nb"])
    M = int(spec["M"])
    iopt = spec["iopt"]

    k_eta, k_theta = jax.random.split(key, 2)
    eta_randn = jax.random.normal(k_eta, (s_steps, nb))
    if iopt == "sase":
        theta_rand = jax.random.uniform(k_theta, (s_steps, nb, M))
    else:
        theta_rand = jnp.zeros((s_steps, nb, M))
    return {"eta_randn": eta_randn, "theta_rand": theta_rand}


def sample_shot_noise_batch_jax(key, spec, n_samples):
    """Sample a batch of explicit shot-noise realizations."""
    keys = jax.random.split(key, int(n_samples))
    return jax.vmap(lambda k: sample_shot_noise_jax(k, spec))(keys)


def bucket_from_shot_noise_jax(spec, noise, *, gbar, delg, Ns):
    """Deterministically map explicit noise -> bucket_data (JAX arrays)."""
    s_steps = int(spec["s_steps"])
    npart = int(spec["npart"])
    nb = int(spec["nb"])
    M = int(spec["M"])
    iopt = spec["iopt"]

    eta_randn = jnp.asarray(noise["eta_randn"])
    theta_rand = jnp.asarray(noise["theta_rand"])

    base = 2 * jnp.pi * (jnp.arange(M) + 1) / M
    theta_base = jnp.tile(base, nb)[jnp.newaxis, :]
    eta_beamlet = delg * eta_randn + gbar
    eta_init = jnp.repeat(eta_beamlet, M, axis=1)

    if iopt == "sase":
        effnoise = jnp.sqrt(3 * M / (Ns / nb))
        thet_init = theta_base + 2 * effnoise * theta_rand.reshape(s_steps, npart)
    elif iopt == "seeded":
        thet_init = jnp.broadcast_to(theta_base, (s_steps, npart))
    else:
        raise ValueError(f"Unknown iopt: {iopt}")

    return {
        "thet_init": thet_init,
        "eta_init": eta_init,
        "N_real": jnp.ones((s_steps,)),
        "s_steps": s_steps,
    }
