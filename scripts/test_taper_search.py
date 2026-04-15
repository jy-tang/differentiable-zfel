import itertools
import numpy as np

from zfel import sase1d


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


def run_profile(unduK):
    sase_input = DEFAULT_INPUT.copy()
    sase_input["unduK"] = unduK
    sase_input["z_steps"] = len(unduK)
    out = sase1d.sase(sase_input)
    return {
        "pulse_energy_like": float(out["power_s"][-1].sum()),
        "final_power": float(out["power_z"][-1]),
        "z": out["z"],
        "power_z": out["power_z"],
    }


def quadratic_taper_profile(*, k0=3.5, n=200, split_ix=80, linear=0.0, quadratic=0.06):
    u = np.linspace(0.0, 1.0, n - split_ix)
    tail = 1.0 - linear * u - quadratic * u**2
    return np.hstack([np.ones(split_ix), tail]) * k0


def main():
    baseline = run_profile(quadratic_taper_profile(split_ix=80, linear=0.0, quadratic=0.0))
    baseline_pulse = baseline["pulse_energy_like"]

    print("Baseline no-taper case")
    print(f"  pulse-energy-like objective: {baseline_pulse:.6e}")
    print(f"  final power [GW]:            {baseline['final_power'] / 1e9:.3f}")
    print()

    reference = run_profile(quadratic_taper_profile(split_ix=80, linear=0.0, quadratic=0.06))
    print("Reference taper_jax-like case")
    print(f"  split_ix=80 linear=0.00 quadratic=0.06")
    print(f"  pulse-energy-like objective: {reference['pulse_energy_like']:.6e}")
    print(f"  final power [GW]:            {reference['final_power'] / 1e9:.3f}")
    print(f"  pulse-energy ratio:          {reference['pulse_energy_like'] / baseline_pulse:.3f}x")
    print()

    best = None
    print("Local 3-parameter taper scan")
    for split_ix, linear, quadratic in itertools.product(
        [60, 70, 80],
        [0.0, 0.01, 0.02, 0.03],
        [0.04, 0.06, 0.08],
    ):
        unduK = quadratic_taper_profile(
            split_ix=split_ix, linear=linear, quadratic=quadratic
        )
        if float(unduK.min()) < 2.5:
            continue

        out = run_profile(unduK)
        ratio = out["pulse_energy_like"] / baseline_pulse
        print(
            f"  split_ix={split_ix:3d} linear={linear:0.02f} quadratic={quadratic:0.02f} "
            f"ratio={ratio:0.3f}x final_power={out['final_power'] / 1e9:0.3f} GW"
        )
        candidate = {
            "split_ix": split_ix,
            "linear": linear,
            "quadratic": quadratic,
            "ratio": ratio,
            "pulse_energy_like": out["pulse_energy_like"],
            "final_power": out["final_power"],
            "min_k": float(unduK.min()),
        }
        if best is None or candidate["ratio"] > best["ratio"]:
            best = candidate

    print()
    print("Best 3-parameter taper found")
    print(
        f"  split_ix={best['split_ix']} linear={best['linear']:.2f} "
        f"quadratic={best['quadratic']:.2f}"
    )
    print(f"  min(K):                      {best['min_k']:.3f}")
    print(f"  pulse-energy-like objective: {best['pulse_energy_like']:.6e}")
    print(f"  final power [GW]:            {best['final_power'] / 1e9:.3f}")
    print(f"  pulse-energy ratio:          {best['ratio']:.3f}x")


if __name__ == "__main__":
    main()
