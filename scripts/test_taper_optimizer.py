import numpy as np
from scipy.optimize import minimize

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
    }


def profile_from_x(x, *, n=200, k0=3.5):
    split_ix = int(np.clip(np.round(x[0]), 40, 120))
    linear = float(np.clip(x[1], 0.0, 0.03))
    quadratic = float(np.clip(x[2], 0.0, 0.10))
    u = np.linspace(0.0, 1.0, n - split_ix)
    tail = 1.0 - linear * u - quadratic * u**2
    unduK = np.hstack([np.ones(split_ix), tail]) * k0
    return unduK, split_ix, linear, quadratic


def main():
    baseline = run_profile(profile_from_x([80, 0.0, 0.0])[0])
    baseline_pulse = baseline["pulse_energy_like"]

    print("Baseline no-taper case")
    print(f"  pulse-energy-like objective: {baseline_pulse:.6e}")
    print(f"  final power [GW]:            {baseline['final_power'] / 1e9:.3f}")
    print()

    cache = {}

    def evaluate(x, *, verbose=True):
        unduK, split_ix, linear, quadratic = profile_from_x(x)
        key = (split_ix, round(linear, 6), round(quadratic, 6))
        if key not in cache:
            if float(unduK.min()) < 2.5:
                cache[key] = {
                    "pulse_energy_like": -np.inf,
                    "final_power": np.nan,
                    "ratio": -np.inf,
                    "min_k": float(unduK.min()),
                }
            else:
                out = run_profile(unduK)
                cache[key] = {
                    **out,
                    "ratio": out["pulse_energy_like"] / baseline_pulse,
                    "min_k": float(unduK.min()),
                }
        result = cache[key]
        if verbose:
            print(
                f"  split_ix={split_ix:3d} linear={linear:0.4f} quadratic={quadratic:0.4f} "
                f"ratio={result['ratio']:0.3f}x final_power={result['final_power'] / 1e9:0.3f} GW"
            )
        return result, split_ix, linear, quadratic

    print("Stage 1: coarse search")
    coarse_candidates = []
    for split_ix in [60, 70, 80]:
        for linear in [0.0, 0.01]:
            for quadratic in [0.04, 0.06, 0.08]:
                result, split_ix, linear, quadratic = evaluate([split_ix, linear, quadratic])
                coarse_candidates.append((result["ratio"], split_ix, linear, quadratic))

    coarse_best = max(coarse_candidates, key=lambda item: item[0])
    x0 = np.array(coarse_best[1:], dtype=float)
    coarse_result, coarse_split, coarse_linear, coarse_quadratic = evaluate(x0, verbose=False)
    print()
    print("Best coarse seed")
    print(f"  split_ix={int(x0[0])} linear={x0[1]:.4f} quadratic={x0[2]:.4f}")
    print(f"  pulse-energy ratio: {coarse_best[0]:.3f}x")
    print()

    print("Stage 2: bounded Powell polish")

    def objective(x):
        result, _, _, _ = evaluate(x, verbose=True)
        if not np.isfinite(result["ratio"]):
            return 1e30
        return -result["pulse_energy_like"]

    res = minimize(
        objective,
        x0,
        method="Powell",
        bounds=[(40, 120), (0.0, 0.03), (0.0, 0.10)],
        options={"maxiter": 6, "xtol": 1.0, "ftol": 1e-3, "disp": True},
    )

    polished_result, split_ix, linear, quadratic = evaluate(res.x, verbose=False)
    if polished_result["ratio"] >= coarse_result["ratio"]:
        final_result = polished_result
        final_split = split_ix
        final_linear = linear
        final_quadratic = quadratic
        final_label = "Powell-polished taper"
    else:
        final_result = coarse_result
        final_split = coarse_split
        final_linear = coarse_linear
        final_quadratic = coarse_quadratic
        final_label = "Best coarse-search taper"

    print()
    print(final_label)
    print(f"  split_ix={final_split}")
    print(f"  linear={final_linear:.6f}")
    print(f"  quadratic={final_quadratic:.6f}")
    print(f"  min(K):                      {final_result['min_k']:.3f}")
    print(f"  pulse-energy-like objective: {final_result['pulse_energy_like']:.6e}")
    print(f"  final power [GW]:            {final_result['final_power'] / 1e9:.3f}")
    print(f"  pulse-energy ratio:          {final_result['ratio']:.3f}x")


if __name__ == "__main__":
    main()
