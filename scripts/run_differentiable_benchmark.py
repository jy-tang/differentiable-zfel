import json
import math
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from zfel import sase1d


jax.config.update("jax_enable_x64", True)


DEFAULT_INPUT = dict(
    npart=64,
    s_steps=16,
    z_steps=16,
    energy=4313.34e6,
    eSpread=0.0,
    emitN=1.2e-6,
    currentMax=3900,
    beta=26,
    unduPeriod=0.03,
    unduK=np.full(16, 3.5),
    unduL=70,
    radWavelength=None,
    random_seed=31,
    particle_position=np.genfromtxt("docs/examples/data/SASE_particle_position.csv", delimiter=","),
    hist_rule="square-root",
    iopt="sase",
    P0=0.0,
)


OUTDIR = Path("docs/examples/generated_plots/benchmark_run_2026-04-14")
SUMMARY_JSON = OUTDIR / "benchmark_summary.json"
SUMMARY_CSV = OUTDIR / "benchmark_summary.csv"
RAW_JSON = OUTDIR / "benchmark_raw_data.json"


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


def k_reference_taper(k0=3.5, a=0.06, n=16, split_ix=6):
    u = np.linspace(0.0, 1.0, n - split_ix)
    return np.hstack([np.ones(split_ix), (1.0 - a * u**2)]) * k0


def build_interp_matrix(n_steps, n_ctrl):
    z_ctrl = np.linspace(0, n_steps - 1, n_ctrl)
    weights = np.zeros((n_steps, n_ctrl))
    for z_i in range(n_steps):
        if z_i <= z_ctrl[0]:
            weights[z_i, 0] = 1.0
        elif z_i >= z_ctrl[-1]:
            weights[z_i, -1] = 1.0
        else:
            j = np.searchsorted(z_ctrl, z_i) - 1
            t = (z_i - z_ctrl[j]) / (z_ctrl[j + 1] - z_ctrl[j])
            weights[z_i, j] = 1.0 - t
            weights[z_i, j + 1] = t
    return jnp.asarray(weights)


def inv_sigmoid(y):
    y = jnp.clip(y, 1e-6, 1.0 - 1e-6)
    return jnp.log(y / (1.0 - y))


def params_from_k_profile(k_profile, base_params, base_input):
    energy = jnp.asarray(base_input["energy"])
    emitN = jnp.asarray(base_input["emitN"])
    currentMax = jnp.asarray(base_input["currentMax"])
    beta = jnp.asarray(base_input["beta"])
    unduPeriod = jnp.asarray(base_input["unduPeriod"])

    x = k_profile**2 / (4 + 2 * k_profile**2)
    unduJJ = j0_series(x) - j1_series(x)

    gamma0 = energy / sase1d.mc2
    sigmaX2 = emitN * beta / gamma0
    kappa_1 = sase1d.e * k_profile * unduJJ / 4 / sase1d.epsilon_0 / gamma0
    Kai = sase1d.e * k_profile * unduJJ / (2 * gamma0**2 * sase1d.mc2 * sase1d.e)
    density = currentMax / (sase1d.e * sase1d.c * 2 * jnp.pi * sigmaX2)

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
            "Ns": currentMax * z0 / unduPeriod / z_steps * resWavelength / sase1d.c / sase1d.e,
            "deta": jnp.sqrt((1 + 0.5 * k_profile[0] ** 2) / (1 + 0.5 * k_profile**2)) - 1,
        }
    )
    return params


def pulse_energy_like(output):
    return output["power_s"][-1].sum()


def make_context():
    base_input = dict(DEFAULT_INPUT)
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

    noise_spec = sase1d.make_shot_noise_spec_from_params(
        base_params_np,
        npart=base_input["npart"],
        s_steps=base_input["s_steps"],
        iopt=base_input["iopt"],
    )

    n_steps = base_input["z_steps"]
    n_ctrl = 6
    weight_matrix = build_interp_matrix(n_steps, n_ctrl)
    k_min = 3.15
    k_max = 3.5
    k_no = jnp.asarray(np.full(n_steps, 3.5), dtype=jnp.float64)
    k_ref = jnp.asarray(k_reference_taper(n=n_steps), dtype=jnp.float64)

    k_ref_ctrl = np.asarray(weight_matrix.T @ k_ref / np.sum(np.asarray(weight_matrix.T), axis=1))
    theta_ref = jnp.clip(inv_sigmoid((jnp.asarray(k_ref_ctrl) - k_min) / (k_max - k_min)), -6.0, 6.0)

    return {
        "base_input": base_input,
        "base_params": base_params,
        "base_params_np": base_params_np,
        "bucket": bucket,
        "noise_spec": noise_spec,
        "weight_matrix": weight_matrix,
        "k_min": k_min,
        "k_max": k_max,
        "k_no": k_no,
        "k_ref": k_ref,
        "theta_ref": theta_ref,
    }


def theta_to_k_profile(theta, weight_matrix, k_min, k_max):
    k_ctrl = k_min + (k_max - k_min) * jax.nn.sigmoid(theta)
    return weight_matrix @ k_ctrl


def objective_from_theta(theta, context, lambda_smooth=5e2, lambda_mono=5e3):
    k_profile = theta_to_k_profile(theta, context["weight_matrix"], context["k_min"], context["k_max"])
    params = params_from_k_profile(k_profile, context["base_params"], context["base_input"])
    out = sase1d.sase_from_initial_conditions_jax(params, context["bucket"])
    pulse = pulse_energy_like(out)
    smooth_pen = jnp.mean((k_profile[2:] - 2 * k_profile[1:-1] + k_profile[:-2]) ** 2)
    mono_pen = jnp.mean(jnp.maximum(k_profile[1:] - k_profile[:-1], 0.0) ** 2)
    objective = jnp.log(pulse) - lambda_smooth * smooth_pen - lambda_mono * mono_pen
    return objective, {
        "pulse": pulse,
        "k_profile": k_profile,
        "smooth_pen": smooth_pen,
        "mono_pen": mono_pen,
    }


def deterministic_summary(k_profile, context):
    params = params_from_k_profile(jnp.asarray(k_profile), context["base_params"], context["base_input"])
    out = sase1d.sase_from_initial_conditions_jax(params, context["bucket"])
    return {
        "pulse": float(pulse_energy_like(out)),
        "final_power_gw": float(out["power_z"][-1]) / 1e9,
        "out": out,
    }


def stochastic_validation(k_profile, context, noise_batch):
    params = sase1d.params_calc(**{**context["base_input"], "unduK": np.asarray(k_profile), "z_steps": len(k_profile)})
    energies = []
    for eta_randn, theta_rand in zip(noise_batch["eta_randn"], noise_batch["theta_rand"]):
        out = sase1d.FEL_sim(
            params,
            {"eta_randn": eta_randn, "theta_rand": theta_rand},
            context["noise_spec"],
        )
        energies.append(float(pulse_energy_like(out)))
    energies = np.asarray(energies)
    return {
        "mean": float(np.mean(energies)),
        "std": float(np.std(energies)),
        "samples": energies,
    }


def run_gradient(context, base_pulse):
    theta = jnp.asarray(context["theta_ref"], dtype=jnp.float64)
    m = jnp.zeros_like(theta)
    v = jnp.zeros_like(theta)
    lr = 0.08
    b1, b2 = 0.9, 0.999
    eps = 1e-8

    history = []
    best = None
    t0 = time.perf_counter()

    for step in range(1, 6):
        (obj, aux), grad = jax.value_and_grad(objective_from_theta, has_aux=True)(theta, context)
        pulse = float(aux["pulse"])
        k_profile = np.asarray(aux["k_profile"])

        m = b1 * m + (1 - b1) * grad
        v = b2 * v + (1 - b2) * (grad * grad)
        m_hat = m / (1 - b1**step)
        v_hat = v / (1 - b2**step)
        theta = theta + lr * m_hat / (jnp.sqrt(v_hat) + eps)

        record = {
            "step": step,
            "calls": step,
            "elapsed_s": time.perf_counter() - t0,
            "objective": float(obj),
            "pulse": pulse,
            "ratio_vs_no_taper": pulse / base_pulse,
            "k_profile": k_profile,
        }
        history.append(record)
        if best is None or record["pulse"] > best["pulse"]:
            best = record

    return {
        "label": "Gradient (autodiff)",
        "history": history,
        "best": best,
    }


def run_random_search(context, theta_center, base_pulse, n_calls=8, seed=11):
    rng = np.random.default_rng(seed)
    history = []
    best = None
    t0 = time.perf_counter()

    for step in range(1, n_calls + 1):
        theta = np.asarray(theta_center) + rng.normal(scale=0.9, size=np.asarray(theta_center).shape)
        theta = np.clip(theta, -6.0, 6.0)
        obj, aux = objective_from_theta(jnp.asarray(theta), context)
        pulse = float(aux["pulse"])
        record = {
            "step": step,
            "calls": step,
            "elapsed_s": time.perf_counter() - t0,
            "objective": float(obj),
            "pulse": pulse,
            "ratio_vs_no_taper": pulse / base_pulse,
            "k_profile": np.asarray(aux["k_profile"]),
        }
        history.append(record)
        if best is None or record["pulse"] > best["pulse"]:
            best = record

    return {
        "label": "Random search",
        "history": history,
        "best": best,
    }


def run_powell(context, theta_center, base_pulse, max_calls=8):
    history = []
    best = {"pulse": -np.inf}
    t0 = time.perf_counter()

    def objective_numpy(theta):
        obj, aux = objective_from_theta(jnp.asarray(theta), context)
        pulse = float(aux["pulse"])
        record = {
            "step": len(history) + 1,
            "calls": len(history) + 1,
            "elapsed_s": time.perf_counter() - t0,
            "objective": float(obj),
            "pulse": pulse,
            "ratio_vs_no_taper": pulse / base_pulse,
            "k_profile": np.asarray(aux["k_profile"]),
        }
        history.append(record)
        if pulse > best["pulse"]:
            best.update(record)
        return -float(obj)

    minimize(
        objective_numpy,
        np.asarray(theta_center, dtype=float),
        method="Powell",
        bounds=[(-6.0, 6.0)] * len(np.asarray(theta_center)),
        options={"maxfev": max_calls, "disp": False},
    )

    return {
        "label": "Powell (derivative-free)",
        "history": history,
        "best": best,
    }


def write_summary_csv(rows, outpath):
    header = (
        "method,best_ratio_vs_no_taper,best_final_power_gw,heldout_mean_ratio,heldout_std_ratio,"
        "objective_calls,elapsed_s\n"
    )
    lines = [header]
    for row in rows:
        lines.append(
            f"{row['label']},{row['best_ratio_vs_no_taper']:.6f},{row['best_final_power_gw']:.6f},"
            f"{row['heldout_mean_ratio']:.6f},{row['heldout_std_ratio']:.6f},{row['objective_calls']},"
            f"{row['elapsed_s']:.6f}\n"
        )
    outpath.write_text("".join(lines), encoding="utf-8")


def serialize_history(history):
    serial = []
    for item in history:
        serial.append(
            {
                "step": int(item["step"]),
                "calls": int(item["calls"]),
                "elapsed_s": float(item["elapsed_s"]),
                "objective": float(item["objective"]),
                "pulse": float(item["pulse"]),
                "ratio_vs_no_taper": float(item["ratio_vs_no_taper"]),
                "k_profile": np.asarray(item["k_profile"]).tolist(),
            }
        )
    return serial


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print("Preparing benchmark context...")
    context = make_context()
    base_summary = deterministic_summary(context["k_no"], context)
    ref_summary = deterministic_summary(context["k_ref"], context)
    base_pulse = base_summary["pulse"]

    print("Running gradient benchmark...")
    gradient = run_gradient(context, base_pulse)
    print("Running Powell benchmark...")
    powell = run_powell(context, context["theta_ref"], base_pulse, max_calls=8)
    print("Running random-search benchmark...")
    random_search = run_random_search(context, context["theta_ref"], base_pulse, n_calls=8, seed=11)

    best_profiles = {
        "No taper": np.asarray(context["k_no"]),
        "Reference taper": np.asarray(context["k_ref"]),
        gradient["label"]: gradient["best"]["k_profile"],
        powell["label"]: powell["best"]["k_profile"],
        random_search["label"]: random_search["best"]["k_profile"],
    }

    deterministic_rows = []
    power_curves = {}
    for label, k_profile in best_profiles.items():
        summary = deterministic_summary(k_profile, context)
        out = summary["out"]
        deterministic_rows.append(
            {
                "label": label,
                "pulse": summary["pulse"],
                "ratio_vs_no_taper": summary["pulse"] / base_pulse,
                "final_power_gw": summary["final_power_gw"],
            }
        )
        power_curves[label] = {
            "z": np.asarray(out["z"]),
            "k_profile": np.asarray(k_profile),
            "power_z_gw": np.asarray(out["power_z"]) / 1e9,
        }

    print("Running held-out stochastic validation...")
    heldout_noise = sase1d.sample_shot_noise_batch_numpy(
        context["noise_spec"], n_samples=4, seed=20260414
    )

    no_taper_val = stochastic_validation(context["k_no"], context, heldout_noise)
    reference_val = stochastic_validation(context["k_ref"], context, heldout_noise)
    gradient_val = stochastic_validation(gradient["best"]["k_profile"], context, heldout_noise)
    powell_val = stochastic_validation(powell["best"]["k_profile"], context, heldout_noise)
    random_val = stochastic_validation(random_search["best"]["k_profile"], context, heldout_noise)

    heldout_base_mean = no_taper_val["mean"]
    validation_rows = [
        {
            "label": "No taper",
            "mean_ratio": no_taper_val["mean"] / heldout_base_mean,
            "std_ratio": no_taper_val["std"] / heldout_base_mean,
        },
        {
            "label": "Reference taper",
            "mean_ratio": reference_val["mean"] / heldout_base_mean,
            "std_ratio": reference_val["std"] / heldout_base_mean,
        },
        {
            "label": gradient["label"],
            "mean_ratio": gradient_val["mean"] / heldout_base_mean,
            "std_ratio": gradient_val["std"] / heldout_base_mean,
        },
        {
            "label": powell["label"],
            "mean_ratio": powell_val["mean"] / heldout_base_mean,
            "std_ratio": powell_val["std"] / heldout_base_mean,
        },
        {
            "label": random_search["label"],
            "mean_ratio": random_val["mean"] / heldout_base_mean,
            "std_ratio": random_val["std"] / heldout_base_mean,
        },
    ]

    summary_rows = []
    method_lookup = {
        gradient["label"]: gradient,
        powell["label"]: powell,
        random_search["label"]: random_search,
    }
    heldout_lookup = {
        gradient["label"]: gradient_val,
        powell["label"]: powell_val,
        random_search["label"]: random_val,
    }
    deterministic_lookup = {row["label"]: row for row in deterministic_rows}

    for label in [gradient["label"], powell["label"], random_search["label"]]:
        result = method_lookup[label]
        heldout = heldout_lookup[label]
        det = deterministic_lookup[label]
        summary_rows.append(
            {
                "label": label,
                "best_ratio_vs_no_taper": det["ratio_vs_no_taper"],
                "best_final_power_gw": det["final_power_gw"],
                "heldout_mean_ratio": heldout["mean"] / heldout_base_mean,
                "heldout_std_ratio": heldout["std"] / heldout_base_mean,
                "objective_calls": len(result["history"]),
                "elapsed_s": result["history"][-1]["elapsed_s"],
            }
        )

    payload = {
        "run_date": "2026-04-14",
        "deterministic_baselines": {
            "no_taper": {
                "pulse": base_summary["pulse"],
                "final_power_gw": base_summary["final_power_gw"],
            },
            "reference_taper": {
                "pulse": ref_summary["pulse"],
                "final_power_gw": ref_summary["final_power_gw"],
                "ratio_vs_no_taper": ref_summary["pulse"] / base_pulse,
            },
        },
        "methods": summary_rows,
        "raw_results": {
            "gradient_history": serialize_history(gradient["history"]),
            "powell_history": serialize_history(powell["history"]),
            "random_history": serialize_history(random_search["history"]),
            "power_curves": {
                label: {
                    "z": curve["z"].tolist(),
                    "k_profile": curve["k_profile"].tolist(),
                    "power_z_gw": curve["power_z_gw"].tolist(),
                }
                for label, curve in power_curves.items()
            },
            "validation_rows": validation_rows,
        },
        "plots": {
            "progress": str(OUTDIR / "benchmark_progress.png"),
            "profiles_power": str(OUTDIR / "benchmark_profiles_power.png"),
            "stochastic_validation": str(OUTDIR / "benchmark_stochastic_validation.png"),
        },
    }

    RAW_JSON.write_text(json.dumps(payload["raw_results"], indent=2), encoding="utf-8")
    SUMMARY_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_summary_csv(summary_rows, SUMMARY_CSV)

    print("Saved benchmark artifacts:")
    print(f"  {RAW_JSON}")
    print(f"  {SUMMARY_JSON}")
    print(f"  {SUMMARY_CSV}")
    print()
    print("Key results:")
    print(f"  Reference taper ratio vs no taper: {ref_summary['pulse'] / base_pulse:.3f}x")
    for row in summary_rows:
        print(
            f"  {row['label']}: deterministic {row['best_ratio_vs_no_taper']:.3f}x, "
            f"held-out stochastic mean {row['heldout_mean_ratio']:.3f}x, calls={row['objective_calls']}"
        )


if __name__ == "__main__":
    main()
