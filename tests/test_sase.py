import numpy as np

from zfel import sase1d


def _base_sase_input():
    return dict(
        npart=128,  # n-macro-particles per bucket
        s_steps=32,  # n-sample points along bunch length
        z_steps=32,  # n-sample points along undulator
        energy=4313.34e6,  # electron energy [eV]
        eSpread=0,  # relative rms energy spread [1]
        emitN=1.2e-6,  # normalized transverse emittance [m-rad]
        currentMax=3400,  # peak current [Ampere]
        beta=26,  # mean beta [meter]
        unduPeriod=0.03,  # undulator period [meter]
        unduK=3.5,  # undulator parameter, K [1], array could taper.
        unduL=70,  # length of undulator [meter]
        radWavelength=None,  # use resonant wavelength from unduK[0]
        random_seed=31,  # reproducible stochastic bucket loading
        particle_position=None,
        hist_rule="square-root",
        iopt="sase",
        P0=0,  # small seed input power [W]
    )


def test_sase():
    output = sase1d.sase(_base_sase_input())
    assert "power_z" in output
    assert output["power_z"].shape == (32,)


def test_sase_from_initial_conditions_matches_wrapper():
    sase_input = _base_sase_input()
    wrapper_output = sase1d.sase(sase_input)

    params = sase1d.params_calc(**sase_input)
    bucket_data = sase1d.fixed_or_external_bucket_data(
        params=params,
        npart=sase_input["npart"],
        s_steps=sase_input["s_steps"],
        particle_position=sase_input["particle_position"],
        hist_rule=sase_input["hist_rule"],
        iopt=sase_input["iopt"],
        random_seed=sase_input["random_seed"],
    )
    split_output = sase1d.sase_from_initial_conditions(params, bucket_data)

    assert np.allclose(wrapper_output["power_z"], split_output["power_z"])
    assert np.allclose(wrapper_output["power_s"], split_output["power_s"])


def test_sase_from_initial_conditions_jax_matches_numpy():
    try:
        import jax
        import jax.numpy as jnp
    except ImportError:
        return

    sase_input = _base_sase_input()
    params = sase1d.params_calc(**sase_input)
    bucket_data = sase1d.fixed_or_external_bucket_data(
        params=params,
        npart=sase_input["npart"],
        s_steps=sase_input["s_steps"],
        particle_position=sase_input["particle_position"],
        hist_rule=sase_input["hist_rule"],
        iopt=sase_input["iopt"],
        random_seed=sase_input["random_seed"],
    )

    out_np = sase1d.sase_from_initial_conditions(params, bucket_data)

    params_jax = {k: jnp.asarray(v) for k, v in params.items()}
    bucket_jax = {k: (int(v) if k == "s_steps" else jnp.asarray(v)) for k, v in bucket_data.items()}
    out_jax = sase1d.sase_from_initial_conditions_jax(params_jax, bucket_jax)

    # Different backends (NumPy vs JAX/XLA) can diverge slightly in long,
    # nonlinear recurrences. We assert close agreement, not bitwise equality.
    assert np.allclose(np.asarray(out_jax["power_z"]), out_np["power_z"], rtol=2e-2, atol=1e-6)
    power_s_jax = np.asarray(out_jax["power_s"])
    power_s_np = out_np["power_s"]
    assert power_s_jax.shape == power_s_np.shape
    rel_l2 = np.linalg.norm(power_s_jax - power_s_np) / np.linalg.norm(power_s_np)
    assert rel_l2 < 0.2

    def objective(kappa_1):
        p = dict(params_jax)
        p["kappa_1"] = kappa_1
        out = sase1d.sase_from_initial_conditions_jax(p, bucket_jax)
        return jnp.sum(out["power_z"])

    grad_kappa = jax.grad(objective)(params_jax["kappa_1"])
    assert grad_kappa.shape == params_jax["kappa_1"].shape
