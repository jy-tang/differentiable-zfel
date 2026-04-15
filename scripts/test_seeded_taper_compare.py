import math

import jax
import jax.numpy as jnp
import numpy as np

from zfel import sase1d


jax.config.update("jax_enable_x64", True)


HC_EV_M = 1.2398419843320026e-6
MC2_EV = 510998.95


def resonance_k(energy_eV, photon_energy_eV, undu_period):
    wavelength = HC_EV_M / photon_energy_eV
    gamma = energy_eV / MC2_EV
    value = 2.0 * gamma**2 * wavelength / undu_period - 1.0
    if value <= 0.0:
        raise ValueError("Requested photon energy is above the fundamental resonance.")
    return math.sqrt(2.0 * value), wavelength


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


K0, RAD_WAVELENGTH = resonance_k(8.0e9, 9.8e3, 0.03)


BASE_INPUT = dict(
    npart=512,
    s_steps=200,
    z_steps=200,
    energy=8.0e9,
    eSpread=0.0,
    emitN=0.3e-6,
    currentMax=2000,
    beta=26,
    unduPeriod=0.03,
    unduK=np.full(200, K0),
    unduL=130,
    radWavelength=RAD_WAVELENGTH,
    random_seed=31,
    particle_position=None,
    hist_rule="square-root",
    iopt="seeded",
    P0=2.0e8,
)


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
    P0 = jnp.asarray(base_input["P0"])
    radWavelength = jnp.asarray(base_input["radWavelength"])

    x = k_profile**2 / (4 + 2 * k_profile**2)
    unduJJ = j0_series(x) - j1_series(x)

    gamma0 = energy / mc2
    sigmaX2 = emitN * beta / gamma0
    kappa_1 = e * k_profile * unduJJ / 4 / epsilon_0 / gamma0
    Kai = e * k_profile * unduJJ / (2 * gamma0**2 * mc2 * e)
    density = currentMax / (e * c * 2 * jnp.pi * sigmaX2)
    res_wavelength = unduPeriod * (1 + k_profile[0] ** 2 / 2.0) / (2 * gamma0**2)
    Pbeam = energy * currentMax
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
            "E02": density * kappa_1[0] * P0 / Pbeam / Kai[0],
            "gbar": res_wavelength / radWavelength - 1.0,
            "Ns": currentMax * z0 / unduPeriod / z_steps * res_wavelength / c / e,
            "deta": jnp.sqrt((1 + 0.5 * k_profile[0] ** 2) / (1 + 0.5 * k_profile**2)) - 1,
        }
    )
    return params


def pulse_energy_like(output):
    return output["power_s"][-1].sum()


def final_power(output):
    return output["power_z"][-1]


def main():
    base_params_np = sase1d.params_calc(**BASE_INPUT)
    base_params = {k: jnp.asarray(v) for k, v in base_params_np.items()}

    bucket_np = sase1d.fixed_or_external_bucket_data(
        params=base_params_np,
        npart=BASE_INPUT["npart"],
        s_steps=BASE_INPUT["s_steps"],
        particle_position=BASE_INPUT["particle_position"],
        hist_rule=BASE_INPUT["hist_rule"],
        iopt=BASE_INPUT["iopt"],
        random_seed=BASE_INPUT["random_seed"],
    )
    bucket = {
        "thet_init": jnp.asarray(bucket_np["thet_init"]),
        "eta_init": jnp.asarray(bucket_np["eta_init"]),
        "N_real": jnp.asarray(bucket_np["N_real"]),
        "s_steps": int(bucket_np["s_steps"]),
    }

    n_steps = BASE_INPUT["z_steps"]
    z_grid = jnp.linspace(0.0, 1.0, n_steps)
    k0_profile = jnp.asarray(np.full(n_steps, K0), dtype=jnp.float64)
    def theta_to_k_profile(theta):
        start_frac = jnp.clip(theta[0], 0.18, 0.50)
        linear = jnp.clip(theta[1], 0.0, 0.03)
        quadratic = jnp.clip(theta[2], 0.0, 0.06)
        tail_span = jnp.maximum(1.0 - start_frac, 1e-6)
        u = jnp.clip((z_grid - start_frac) / tail_span, 0.0, 1.0)
        gate = jax.nn.sigmoid((z_grid - start_frac) / 0.03)
        taper = gate * (linear * u + quadratic * u**2)
        return K0 * (1.0 - taper)

    base_out = sase1d.sase_from_initial_conditions_jax(
        params_from_k_profile(k0_profile, base_params, BASE_INPUT),
        bucket,
    )
    base_power = float(final_power(base_out))
    base_energy = float(pulse_energy_like(base_out))

    print("Seeded fixed-bucket quadratic taper gradient optimization")
    print(f"  photon energy target [keV]:    {9.8:.3f}")
    print(f"  resonant wavelength [m]:       {RAD_WAVELENGTH:.6e}")
    print(f"  resonant K0:                   {K0:.6f}")
    print(f"  seed power P0 [W]:             {BASE_INPUT['P0']:.3e}")
    print(f"  no-taper final power [GW]:     {base_power / 1e9:.3f}")
    print(f"  no-taper pulse energy metric:  {base_energy:.6e}")
    print()

    def objective(theta, lambda_mono=2e3, lambda_amp=1e-2):
        k_profile = theta_to_k_profile(theta)
        params = params_from_k_profile(k_profile, base_params, BASE_INPUT)
        out = sase1d.sase_from_initial_conditions_jax(params, bucket)
        power_ratio = final_power(out) / base_power
        energy_ratio = pulse_energy_like(out) / base_energy
        mono_pen = jnp.mean(jnp.maximum(k_profile[1:] - k_profile[:-1], 0.0) ** 2)
        amp_pen = jnp.mean(theta**2)
        score = jnp.log(power_ratio) + jnp.log(energy_ratio) - lambda_mono * mono_pen - lambda_amp * amp_pen
        return score, (out, k_profile, power_ratio, energy_ratio)

    def run_gradient(theta0, label, n_iter=12, lr=0.02):
        theta = jnp.asarray(theta0, dtype=jnp.float64)
        m = jnp.zeros_like(theta)
        v = jnp.zeros_like(theta)
        b1, b2 = 0.9, 0.999
        eps = 1e-8
        best_local = None

        for t in range(1, n_iter + 1):
            (obj, (out, k_profile, power_ratio, energy_ratio)), g = jax.value_and_grad(
                objective, has_aux=True
            )(theta)
            g = g / jnp.maximum(jnp.linalg.norm(g), 1.0)
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * (g * g)
            m_hat = m / (1 - b1**t)
            v_hat = v / (1 - b2**t)
            theta = theta + lr * m_hat / (jnp.sqrt(v_hat) + eps)

            candidate = {
                "iter": t,
                "score": float(obj),
                "power_ratio": float(power_ratio),
                "energy_ratio": float(energy_ratio),
                "out": out,
                "k_profile": np.asarray(k_profile),
                "theta": np.asarray(theta),
                "label": label,
            }
            if best_local is None or candidate["score"] > best_local["score"]:
                best_local = candidate

            if t in [1, 2, 4, 8, 12]:
                print(
                    f"{label} iter={t:02d} power_ratio={float(power_ratio):.3f}x "
                    f"energy_ratio={float(energy_ratio):.3f}x"
                )

        return best_local

    start_points = [
        np.array([0.31, 0.02, 0.05], dtype=float),
        np.array([0.28, 0.02, 0.05], dtype=float),
        np.array([0.34, 0.02, 0.05], dtype=float),
        np.array([0.31, 0.01, 0.04], dtype=float),
        np.array([0.31, 0.03, 0.06], dtype=float),
    ]

    best = None
    for idx, theta0 in enumerate(start_points, start=1):
        print(f"Start {idx}/{len(start_points)}: theta0={theta0.tolist()}")
        candidate = run_gradient(theta0, label=f"grad{idx}", n_iter=12, lr=0.02)
        if best is None or candidate["score"] > best["score"]:
            best = candidate
        print()

    print()
    print("Best gradient quadratic taper")
    print(f"  iteration:                     {best['iter']}")
    print(f"  final power [GW]:              {float(final_power(best['out'])) / 1e9:.3f}")
    print(f"  pulse energy metric:           {float(pulse_energy_like(best['out'])):.6e}")
    print(f"  power ratio vs no taper:       {best['power_ratio']:.3f}x")
    print(f"  energy ratio vs no taper:      {best['energy_ratio']:.3f}x")
    print(f"  min(K):                        {float(np.min(best['k_profile'])):.3f}")
    print(f"  max(K):                        {float(np.max(best['k_profile'])):.3f}")


if __name__ == "__main__":
    main()
