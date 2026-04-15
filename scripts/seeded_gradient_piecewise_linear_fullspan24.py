import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from zfel import sase1d


jax.config.update("jax_enable_x64", True)


HC_EV_M = 1.2398419843320026e-6
MC2_EV = 510998.95
LOG2 = math.log(2.0)


def resonance_k(energy_eV, photon_energy_eV, undu_period):
    wavelength = HC_EV_M / photon_energy_eV
    gamma = energy_eV / MC2_EV
    value = 2.0 * gamma**2 * wavelength / undu_period - 1.0
    if value <= 0.0:
        raise ValueError("Requested photon energy is above the fundamental resonance.")
    return math.sqrt(2.0 * value), wavelength


TARGET_PHOTON_ENERGY_EV = 9.8e3
K0, RAD_WAVELENGTH = resonance_k(8.0e9, TARGET_PHOTON_ENERGY_EV, 0.03)


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


OUTDIR = Path("docs/examples/generated_plots/seeded_gradient_piecewise_linear_fullspan24_p2e8_2026-04-15")
N_KNOTS = 24
KNOT_X = jnp.linspace(0.0, 1.0, N_KNOTS + 1)
DELTA_SCALE = 0.03


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


def params_from_k_profile(k_profile, base_params, base_input):
    energy = jnp.asarray(base_input["energy"])
    emitN = jnp.asarray(base_input["emitN"])
    currentMax = jnp.asarray(base_input["currentMax"])
    beta = jnp.asarray(base_input["beta"])
    unduPeriod = jnp.asarray(base_input["unduPeriod"])
    p0 = jnp.asarray(base_input["P0"])
    rad_wavelength = jnp.asarray(base_input["radWavelength"])

    x = k_profile**2 / (4 + 2 * k_profile**2)
    unduJJ = j0_series(x) - j1_series(x)

    gamma0 = energy / sase1d.mc2
    sigmaX2 = emitN * beta / gamma0
    kappa_1 = sase1d.e * k_profile * unduJJ / 4 / sase1d.epsilon_0 / gamma0
    Kai = sase1d.e * k_profile * unduJJ / (2 * gamma0**2 * sase1d.mc2 * sase1d.e)
    density = currentMax / (sase1d.e * sase1d.c * 2 * jnp.pi * sigmaX2)
    res_wavelength = unduPeriod * (1 + k_profile[0] ** 2 / 2.0) / (2 * gamma0**2)
    pbeam = energy * currentMax
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
            "E02": density * kappa_1[0] * p0 / pbeam / Kai[0],
            "gbar": res_wavelength / rad_wavelength - 1.0,
            "Ns": currentMax * z0 / unduPeriod / z_steps * res_wavelength / sase1d.c / sase1d.e,
            "deta": jnp.sqrt((1 + 0.5 * k_profile[0] ** 2) / (1 + 0.5 * k_profile**2)) - 1,
        }
    )
    return params


def final_power(output):
    return output["power_z"][-1]


def pulse_energy_like(output):
    return output["power_s"][-1].sum()


def make_context():
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

    return base_params, bucket


def inv_centered_softplus(value, scale):
    shifted = value / scale + LOG2
    return math.log(math.exp(shifted) - 1.0)


def inv_sigmoid(y):
    y = np.clip(y, 1e-6, 1.0 - 1e-6)
    return math.log(y / (1.0 - y))


def render_reference_quadratic(theta, z_grid, k0):
    start_frac = 0.00 + 0.30 * jax.nn.sigmoid(theta[0])
    linear = 0.18 * (jax.nn.softplus(theta[1]) - LOG2)
    quadratic = 0.36 * (jax.nn.softplus(theta[2]) - LOG2)
    tail_span = jnp.maximum(1.0 - start_frac, 1e-6)
    u = jnp.clip((z_grid - start_frac) / tail_span, 0.0, 1.0)
    gate = jax.nn.sigmoid((z_grid - start_frac) / 0.03)
    taper_fraction = gate * (linear * u + quadratic * u**2)
    return k0 * (1.0 - taper_fraction), start_frac


def render_k_profile(theta, z_grid, k0):
    start_frac = 0.0
    knot_increments = DELTA_SCALE * (jax.nn.softplus(theta) - LOG2)
    taper_knots = jnp.concatenate([jnp.zeros(1), jnp.cumsum(knot_increments)])

    tail_span = jnp.maximum(1.0 - start_frac, 1e-6)
    u = jnp.clip((z_grid - start_frac) / tail_span, 0.0, 1.0)
    gate = jax.nn.sigmoid((z_grid - start_frac) / 0.03)
    taper_fraction = gate * jnp.interp(u, KNOT_X, taper_knots)
    return k0 * (1.0 - taper_fraction)


def warm_start_from_reference():
    # Warm-start from the best 12-knot full-span piecewise-linear run.
    z_grid = np.linspace(0.0, 1.0, BASE_INPUT["z_steps"])
    piecewise_summary = json.loads(
        Path(
            "docs/examples/generated_plots/seeded_gradient_piecewise_linear_fullspan_p2e8_2026-04-15/"
            "seeded_gradient_piecewise_linear_fullspan_summary.json"
        ).read_text(encoding="utf-8")
    )
    theta_ref = jnp.asarray(piecewise_summary["best_theta"], dtype=jnp.float64)
    old_start_frac = 0.00 + 0.30 * jax.nn.sigmoid(theta_ref[0])
    old_knot_increments = DELTA_SCALE * (jax.nn.softplus(theta_ref[1:]) - LOG2)
    old_taper_knots = jnp.concatenate([jnp.zeros(1), jnp.cumsum(old_knot_increments)])
    old_knot_x = jnp.linspace(0.0, 1.0, old_taper_knots.shape[0])
    old_tail_span = jnp.maximum(1.0 - old_start_frac, 1e-6)
    old_u = jnp.clip((jnp.asarray(z_grid) - old_start_frac) / old_tail_span, 0.0, 1.0)
    old_gate = jax.nn.sigmoid((jnp.asarray(z_grid) - old_start_frac) / 0.03)
    old_taper_fraction = old_gate * jnp.interp(old_u, old_knot_x, old_taper_knots)
    k_ref = np.asarray(K0 * (1.0 - old_taper_fraction))
    taper_ref = np.maximum(1.0 - k_ref / K0, 0.0)
    u_knots = np.asarray(KNOT_X)
    taper_knots_ref = np.interp(
        u_knots,
        z_grid,
        taper_ref,
    )
    taper_knots_ref[0] = 0.0
    taper_knots_ref = np.maximum.accumulate(taper_knots_ref)
    knot_increments_ref = np.diff(taper_knots_ref)
    theta0 = [inv_centered_softplus(max(v, 1e-6), DELTA_SCALE) for v in knot_increments_ref]
    return jnp.asarray(theta0, dtype=jnp.float64), np.asarray(k_ref)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    base_params, bucket = make_context()
    n_steps = BASE_INPUT["z_steps"]
    z_grid = np.linspace(BASE_INPUT["unduL"] / n_steps, BASE_INPUT["unduL"], n_steps)
    z_grid_unit = jnp.linspace(0.0, 1.0, n_steps)

    theta0, k_ref = warm_start_from_reference()
    k0_profile = jnp.full(n_steps, K0, dtype=jnp.float64)

    base_out = sase1d.sase_from_initial_conditions_jax(
        params_from_k_profile(k0_profile, base_params, BASE_INPUT),
        bucket,
    )
    ref_out = sase1d.sase_from_initial_conditions_jax(
        params_from_k_profile(jnp.asarray(k_ref), base_params, BASE_INPUT),
        bucket,
    )
    base_power = float(final_power(base_out))
    base_energy = float(pulse_energy_like(base_out))

    print("Seeded FEL full-span piecewise-linear gradient taper")
    print(f"  photon energy target [keV]:    {TARGET_PHOTON_ENERGY_EV / 1e3:.3f}")
    print(f"  resonant wavelength [m]:       {RAD_WAVELENGTH:.6e}")
    print(f"  resonant K0:                   {K0:.6f}")
    print(f"  seed power P0 [W]:             {BASE_INPUT['P0']:.3e}")
    print(f"  no-taper final power [GW]:     {base_power / 1e9:.3f}")
    print(f"  reference warm-start [GW]:     {float(final_power(ref_out)) / 1e9:.3f}")
    print(f"  taper start fixed at:          0.0000")
    print(f"  piecewise-linear knots:        {N_KNOTS}")
    print()

    def oscillation_penalty(power_z_gw, threshold=0.75, sharpness=20.0):
        log_power = jnp.log(jnp.maximum(power_z_gw, 1e-12))
        norm_power = power_z_gw / (jnp.max(power_z_gw) + 1e-12)
        sat_mask = jax.nn.sigmoid((norm_power[:-1] - threshold) * sharpness)
        dlog = log_power[1:] - log_power[:-1]
        return jnp.mean(sat_mask * jnp.maximum(-dlog, 0.0))

    def objective(
        theta,
        lambda_mono=5e3,
        lambda_upper=1e3,
        lambda_lower=1e2,
        lambda_smooth=1e1,
        lambda_osc=4.0,
    ):
        k_profile = render_k_profile(theta, z_grid_unit, K0)
        params = params_from_k_profile(k_profile, base_params, BASE_INPUT)
        out = sase1d.sase_from_initial_conditions_jax(params, bucket)

        power_ratio = final_power(out) / base_power
        energy_ratio = pulse_energy_like(out) / base_energy
        power_z_gw = out["power_z"] / 1e9
        dK = k_profile[1:] - k_profile[:-1]

        mono_pen = jnp.mean(jnp.maximum(dK, 0.0) ** 2)
        upper_pen = jnp.mean(jnp.maximum(k_profile - K0, 0.0) ** 2)
        lower_pen = jnp.mean(jnp.maximum(0.80 - k_profile, 0.0) ** 2)
        smooth_pen = jnp.mean((k_profile[2:] - 2 * k_profile[1:-1] + k_profile[:-2]) ** 2)
        osc_pen = oscillation_penalty(power_z_gw)

        score = (
            jnp.log(power_ratio + 1e-12)
            - lambda_mono * mono_pen
            - lambda_upper * upper_pen
            - lambda_lower * lower_pen
            - lambda_smooth * smooth_pen
            - lambda_osc * osc_pen
        )
        return score, {
            "out": out,
            "k_profile": k_profile,
            "power_ratio": power_ratio,
            "energy_ratio": energy_ratio,
            "mono_pen": mono_pen,
            "osc_pen": osc_pen,
            "theta": theta,
        }

    theta = theta0
    m = jnp.zeros_like(theta)
    v = jnp.zeros_like(theta)
    lr = 0.003
    b1, b2 = 0.9, 0.999
    eps = 1e-8
    n_iter = 50

    history = []
    best = None

    for step in range(1, n_iter + 1):
        (score, aux), grad = jax.value_and_grad(objective, has_aux=True)(theta)

        m = b1 * m + (1 - b1) * grad
        v = b2 * v + (1 - b2) * (grad * grad)
        m_hat = m / (1 - b1**step)
        v_hat = v / (1 - b2**step)
        theta = theta + lr * m_hat / (jnp.sqrt(v_hat) + eps)

        record = {
            "step": step,
            "score": float(score),
            "power_ratio": float(aux["power_ratio"]),
            "energy_ratio": float(aux["energy_ratio"]),
            "final_power_gw": float(final_power(aux["out"])) / 1e9,
            "pulse_energy": float(pulse_energy_like(aux["out"])),
            "mono_pen": float(aux["mono_pen"]),
            "osc_pen": float(aux["osc_pen"]),
            "k_profile": np.asarray(aux["k_profile"]),
            "theta": np.asarray(aux["theta"]),
            "power_z_gw": np.asarray(aux["out"]["power_z"]) / 1e9,
        }
        history.append(record)

        if best is None or record["final_power_gw"] > best["final_power_gw"]:
            best = record

        if step in [1, 2, 5, 10, 20, 30, 40, 50]:
            print(
                f"iter={step:02d} power_ratio={record['power_ratio']:.3f}x "
                f"energy_ratio={record['energy_ratio']:.3f}x "
                f"final_power={record['final_power_gw']:.3f} GW "
                f"osc_pen={record['osc_pen']:.3e}"
            )

    summary = {
        "run_date": "2026-04-15",
        "seed_power_w": BASE_INPUT["P0"],
        "photon_energy_kev": TARGET_PHOTON_ENERGY_EV / 1e3,
        "resonant_k0": K0,
        "no_taper_final_power_gw": base_power / 1e9,
        "warm_start_final_power_gw": float(final_power(ref_out)) / 1e9,
        "best_iter": best["step"],
        "best_final_power_gw": best["final_power_gw"],
        "best_power_ratio": best["power_ratio"],
        "best_energy_ratio": best["energy_ratio"],
        "min_k": float(np.min(best["k_profile"])),
        "max_k": float(np.max(best["k_profile"])),
        "best_theta": np.asarray(best["theta"]).tolist(),
        "best_start_frac": 0.0,
        "n_piecewise_knots": N_KNOTS,
    }

    (OUTDIR / "seeded_gradient_piecewise_linear_fullspan_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.plot(z_grid, np.full_like(z_grid, K0), label="No taper", linewidth=2.2, color="#5B6C8F")
    ax.plot(z_grid, k_ref, label="Quadratic warm start", linewidth=2.0, color="#C2410C")
    ax.plot(z_grid, best["k_profile"], label="Full-span piecewise-linear taper", linewidth=2.6, color="#0E7490")
    ax.set_xlabel("Undulator position z (m)")
    ax.set_ylabel("Undulator parameter K")
    ax.set_title("Seeded FEL full-span piecewise-linear taper profile")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(OUTDIR / "seeded_gradient_piecewise_linear_fullspan_taper_profile.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.plot(z_grid, np.asarray(base_out["power_z"]) / 1e9, label="No taper", linewidth=2.2, color="#5B6C8F")
    ax.plot(z_grid, np.asarray(ref_out["power_z"]) / 1e9, label="Quadratic warm start", linewidth=2.0, color="#C2410C")
    ax.plot(z_grid, best["power_z_gw"], label="Full-span piecewise-linear taper", linewidth=2.6, color="#0E7490")
    ax.set_yscale("log")
    ax.set_xlabel("Undulator position z (m)")
    ax.set_ylabel("Power (GW)")
    ax.set_title("Seeded FEL power growth with full-span piecewise-linear taper")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(OUTDIR / "seeded_gradient_piecewise_linear_fullspan_power_growth.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    steps = [item["step"] for item in history]
    axes[0].plot(steps, [item["power_ratio"] for item in history], linewidth=2.2, color="#0E7490")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Final power ratio vs no taper")
    axes[0].set_title("Power gain during full-span piecewise-linear optimization")
    axes[1].plot(steps, [item["energy_ratio"] for item in history], linewidth=2.2, color="#0F766E")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Pulse-energy ratio vs no taper")
    axes[1].set_title("Pulse-energy gain during full-span piecewise-linear optimization")
    fig.tight_layout()
    fig.savefig(OUTDIR / "seeded_gradient_piecewise_linear_fullspan_history.png", dpi=180)
    plt.close(fig)

    print()
    print("Best seeded full-span piecewise-linear taper")
    print(f"  best iteration:                {best['step']}")
    print(f"  final power [GW]:              {best['final_power_gw']:.3f}")
    print(f"  power ratio vs no taper:       {best['power_ratio']:.3f}x")
    print(f"  pulse energy ratio vs no taper:{best['energy_ratio']:.3f}x")
    print("  start fraction:                0.0000")
    print(f"  min(K):                        {float(np.min(best['k_profile'])):.3f}")
    print(f"  max(K):                        {float(np.max(best['k_profile'])):.3f}")
    print()
    print("Saved artifacts:")
    print(f"  {OUTDIR / 'seeded_gradient_piecewise_linear_fullspan_taper_profile.png'}")
    print(f"  {OUTDIR / 'seeded_gradient_piecewise_linear_fullspan_power_growth.png'}")
    print(f"  {OUTDIR / 'seeded_gradient_piecewise_linear_fullspan_history.png'}")
    print(f"  {OUTDIR / 'seeded_gradient_piecewise_linear_fullspan_summary.json'}")


if __name__ == "__main__":
    main()
