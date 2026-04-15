import math
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from zfel import sase1d


jax.config.update("jax_enable_x64", True)


DEFAULT_INPUT = dict(
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


def k_taper(k0=3.5, a=0.06, n=200, split_ix=80):
    u = np.linspace(0.0, 1.0, n - split_ix)
    return np.hstack([np.ones(split_ix), (1.0 - a * u**2)]) * k0


def build_interp_matrix(n_steps, n_ctrl):
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


def pulse_energy_like(output):
    return output["power_s"][-1].sum()


def run_output(k_profile, base_params, base_input, bucket):
    params = params_from_k_profile(k_profile, base_params, base_input)
    return sase1d.sase_from_initial_conditions_jax(params, bucket)


def main():
    outdir = Path("docs/examples/generated_plots")
    outdir.mkdir(parents=True, exist_ok=True)

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

    n_steps = base_input["z_steps"]
    W = build_interp_matrix(n_steps, n_ctrl=8)
    K_min = 3.15
    K_max = 3.5
    K_no = jnp.asarray(np.full(n_steps, 3.5), dtype=jnp.float64)
    K_ref = jnp.asarray(k_taper(a=0.06, n=n_steps, split_ix=80), dtype=jnp.float64)

    def inv_sigmoid(y):
        y = jnp.clip(y, 1e-6, 1 - 1e-6)
        return jnp.log(y / (1.0 - y))

    K_ref_ctrl = np.asarray(W.T @ K_ref / np.sum(np.asarray(W.T), axis=1))
    theta = inv_sigmoid((jnp.asarray(K_ref_ctrl) - K_min) / (K_max - K_min))

    def theta_to_k_profile(theta_local):
        K_ctrl = K_min + (K_max - K_min) * jax.nn.sigmoid(theta_local)
        return W @ K_ctrl

    def objective(theta_local, lambda_smooth=5e2, lambda_mono=5e3):
        k_profile = theta_to_k_profile(theta_local)
        out = run_output(k_profile, base_params, base_input, bucket)
        pulse = pulse_energy_like(out)
        smooth_pen = jnp.mean((k_profile[2:] - 2 * k_profile[1:-1] + k_profile[:-2]) ** 2)
        mono_pen = jnp.mean(jnp.maximum(k_profile[1:] - k_profile[:-1], 0.0) ** 2)
        return jnp.log(pulse) - lambda_smooth * smooth_pen - lambda_mono * mono_pen, (pulse, k_profile)

    m = jnp.zeros_like(theta)
    v = jnp.zeros_like(theta)
    lr = 0.08
    b1, b2 = 0.9, 0.999
    eps = 1e-8
    best_pulse = -np.inf
    best_k = None
    pulse_hist = []

    for t in range(1, 6):
        (obj, (pulse, k_profile)), g = jax.value_and_grad(objective, has_aux=True)(theta)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * (g * g)
        m_hat = m / (1 - b1**t)
        v_hat = v / (1 - b2**t)
        theta = theta + lr * m_hat / (jnp.sqrt(v_hat) + eps)
        pulse_val = float(pulse)
        pulse_hist.append(pulse_val)
        if pulse_val > best_pulse:
            best_pulse = pulse_val
            best_k = np.asarray(k_profile)

    out_no = run_output(K_no, base_params, base_input, bucket)
    out_ref = run_output(K_ref, base_params, base_input, bucket)
    out_opt = run_output(jnp.asarray(best_k), base_params, base_input, bucket)

    z = np.asarray(out_no["z"])
    pulse_vals = np.array(
        [
            float(pulse_energy_like(out_no)),
            float(pulse_energy_like(out_ref)),
            float(pulse_energy_like(out_opt)),
        ]
    )
    final_power_vals = np.array(
        [
            float(out_no["power_z"][-1]) / 1e9,
            float(out_ref["power_z"][-1]) / 1e9,
            float(out_opt["power_z"][-1]) / 1e9,
        ]
    )

    labels = ["No taper", "Reference taper", "Gradient-optimized"]
    colors = ["#5B6C8F", "#D07A28", "#1E9C89"]

    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(z, np.asarray(K_no), label=labels[0], color=colors[0], linewidth=2.5)
    ax.plot(z, np.asarray(K_ref), label=labels[1], color=colors[1], linewidth=2.5)
    ax.plot(z, best_k, label=labels[2], color=colors[2], linewidth=2.8)
    ax.set_xlabel("Undulator position z (m)")
    ax.set_ylabel("Undulator parameter K")
    ax.set_title("Taper Profiles")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(outdir / "presentation_taper_profiles.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(z, np.asarray(out_no["power_z"]) / 1e9, label=labels[0], color=colors[0], linewidth=2.5)
    ax.plot(z, np.asarray(out_ref["power_z"]) / 1e9, label=labels[1], color=colors[1], linewidth=2.5)
    ax.plot(z, np.asarray(out_opt["power_z"]) / 1e9, label=labels[2], color=colors[2], linewidth=2.8)
    ax.set_yscale("log")
    ax.set_xlabel("Undulator position z (m)")
    ax.set_ylabel("Power (GW)")
    ax.set_title("Power Growth Along the Undulator")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(outdir / "presentation_power_growth.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    x = np.arange(len(labels))
    axes[0].bar(x, pulse_vals / 1e12, color=colors, width=0.65)
    axes[0].set_xticks(x, labels, rotation=12)
    axes[0].set_ylabel("Pulse-energy-like objective")
    axes[0].set_title("Pulse Energy Comparison")

    axes[1].bar(x, final_power_vals, color=colors, width=0.65)
    axes[1].set_xticks(x, labels, rotation=12)
    axes[1].set_ylabel("Final power (GW)")
    axes[1].set_title("Final Power Comparison")

    fig.tight_layout()
    fig.savefig(outdir / "presentation_summary_bars.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(np.arange(1, len(pulse_hist) + 1), np.array(pulse_hist) / pulse_vals[0], marker="o", color=colors[2], linewidth=2.5)
    ax.axhline(pulse_vals[1] / pulse_vals[0], color=colors[1], linestyle="--", linewidth=2.0, label="Reference taper")
    ax.set_xlabel("Gradient iteration")
    ax.set_ylabel("Pulse-energy ratio vs no taper")
    ax.set_title("Gradient Optimization Progress")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(outdir / "presentation_gradient_progress.png", dpi=180)
    plt.close(fig)

    print("Saved plots:")
    for name in [
        "presentation_taper_profiles.png",
        "presentation_power_growth.png",
        "presentation_summary_bars.png",
        "presentation_gradient_progress.png",
    ]:
        print(f"  {outdir / name}")


if __name__ == "__main__":
    main()
