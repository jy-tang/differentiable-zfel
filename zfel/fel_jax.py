"""JAX-backed deterministic FEL kernels."""

import jax
import jax.numpy as jnp


@jax.custom_jvp
def _seed_amplitude(E02):
    """
    Safe seed-field amplitude for autodiff.

    The physical seed amplitude is sqrt(E02). At E02 = 0, reverse-mode AD
    through sqrt produces an undefined derivative, which shows up as NaNs when
    users set P0 = 0. We define the derivative to be zero at non-positive E02,
    which matches the intended "no seed" behavior.
    """
    return jnp.where(E02 > 0.0, jnp.sqrt(E02), 0.0)


@_seed_amplitude.defjvp
def _seed_amplitude_jvp(primals, tangents):
    (E02,) = primals
    (dE02,) = tangents
    y = _seed_amplitude(E02)
    safe_E02 = jnp.where(E02 > 0.0, E02, 1.0)
    dy = jnp.where(E02 > 0.0, 0.5 * dE02 / jnp.sqrt(safe_E02), 0.0)
    return y, dy


def FEL_process_complex_jax(
    npart,
    z_steps,
    kappa_1,
    density,
    Kai,
    ku,
    delt,
    dels,
    deta,
    thet_init,
    eta_init,
    N_real,
    s_steps,
    E02=0,
):
    """
    JAX version of FEL_process_complex, written with functional updates.
    """
    shape = N_real / jnp.max(N_real)

    E = jnp.zeros((s_steps + 1, z_steps + 1), dtype=jnp.complex128)
    thet_final = jnp.zeros((npart, s_steps))
    eta_final = jnp.zeros((npart, s_steps))
    last_thet_output = jnp.zeros((npart, z_steps + 1))
    last_eta = jnp.zeros((npart, z_steps + 1))

    def outer_body(k, state):
        E, thet_final, eta_final, _, _ = state

        thet0 = thet_init[k, :]
        eta0 = eta_init[k, :]
        E = E.at[k, 0].set(_seed_amplitude(E02))

        eta = jnp.zeros((npart, z_steps + 1)).at[:, 0].set(eta0)
        thethalf = jnp.zeros((npart, z_steps + 1)).at[:, 0].set(
            thet0 - 2 * ku * eta[:, 0] * delt / 2
        )
        thet_output = jnp.zeros((npart, z_steps + 1)).at[:, 0].set(thet0)
        thet_last = thet0

        def inner_body(j, inner_state):
            E, eta, thethalf, thet_output, thet_last = inner_state

            thet = thethalf[:, j] + 2 * ku * (eta[:, j] + deta[j]) * delt / 2
            Ehalf = (
                E[k, j]
                + kappa_1[j] * shape[k] * density * jnp.mean(jnp.exp(-1j * thet)) * dels / 2
            )

            thethalf_next = thethalf[:, j] + 2 * ku * (eta[:, j] + deta[j]) * delt
            thethalf = thethalf.at[:, j + 1].set(thethalf_next)
            eta = eta.at[:, j + 1].set(
                eta[:, j] - Kai[j] * delt * 2 * jnp.real(Ehalf * jnp.exp(1j * thethalf_next))
            )

            E = E.at[k + 1, j + 1].set(
                E[k, j]
                + kappa_1[j] * shape[k] * density * jnp.mean(jnp.exp(-1j * thethalf_next)) * dels
            )
            thet_output = thet_output.at[:, j + 1].set(thet)
            return E, eta, thethalf, thet_output, thet

        E, eta, thethalf, thet_output, thet_last = jax.lax.fori_loop(
            0, z_steps, inner_body, (E, eta, thethalf, thet_output, thet_last)
        )
        thet_final = thet_final.at[:, k].set(thet_last)
        eta_final = eta_final.at[:, k].set(eta[:, -1])
        return E, thet_final, eta_final, thet_output, eta

    E, thet_final, eta_final, last_thet_output, last_eta = jax.lax.fori_loop(
        0, s_steps, outer_body, (E, thet_final, eta_final, last_thet_output, last_eta)
    )

    output = {}
    output["Er"] = jnp.real(E)
    output["Ei"] = jnp.imag(E)
    output["thet_final"] = thet_final
    output["eta_final"] = eta_final
    output["theta_final_slice_history"] = last_thet_output
    output["eta_final_slice_history"] = last_eta
    return output


def final_calc_jax(
    Er,
    Ei,
    s_steps,
    z_steps,
    kappa_1,
    density,
    Kai,
    Pbeam,
    delt,
    dels,
):
    """JAX version of final_calc."""
    scale = Kai / (density * kappa_1) * Pbeam

    power_s = (Er[1 : s_steps + 1, :z_steps] ** 2 + Ei[1 : s_steps + 1, :z_steps] ** 2).T
    power_s = power_s * scale[:, jnp.newaxis]

    power_z = jnp.sum(Er[:, :z_steps] ** 2 + Ei[:, :z_steps] ** 2, axis=0) * scale / s_steps

    field_s = (Er + Ei * 1j) * jnp.sqrt(
        jnp.concatenate((jnp.array([Kai[0]]), Kai))[jnp.newaxis, :]
        / (
            density
            * jnp.concatenate((jnp.array([kappa_1[0]]), kappa_1))[jnp.newaxis, :]
            * Pbeam
        )
    )

    spectrum = jnp.abs(jnp.fft.fftshift(jnp.fft.fft(field_s, axis=0), axes=0)) ** 2

    d = {}
    d["power_s"] = power_s
    d["power_z"] = power_z
    d["spectrum"] = spectrum
    return d
