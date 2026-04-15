import numpy as np

from zfel import sase1d
from zfel.particles import bucket_from_shot_noise


def _base_input():
    return dict(
        npart=128,
        s_steps=24,
        z_steps=24,
        energy=4313.34e6,
        eSpread=0.01,
        emitN=1.2e-6,
        currentMax=3400,
        beta=26,
        unduPeriod=0.03,
        unduK=3.5,
        unduL=70,
        radWavelength=None,
        random_seed=31,
        particle_position=None,
        hist_rule="square-root",
        iopt="sase",
        P0=1e3,
    )


def test_single_realization_with_explicit_noise_matches_bucket_solver():
    inp = _base_input()
    params = sase1d.params_calc(**inp)
    spec = sase1d.make_shot_noise_spec_from_params(
        params, npart=inp["npart"], s_steps=inp["s_steps"], iopt=inp["iopt"]
    )
    noise = sase1d.sample_shot_noise_batch_numpy(spec, n_samples=1, seed=7)
    noise1 = {"eta_randn": noise["eta_randn"][0], "theta_rand": noise["theta_rand"][0]}

    out_api = sase1d.FEL_sim(params, noise1, spec)
    bucket = bucket_from_shot_noise(
        {**spec, **noise1}, gbar=params["gbar"], delg=params["delg"], Ns=params["Ns"]
    )
    out_direct = sase1d.sase_from_initial_conditions(params, bucket)

    assert np.allclose(out_api["power_z"], out_direct["power_z"])
    assert np.allclose(out_api["power_s"], out_direct["power_s"])


def test_mc_value_and_grad_crn_reproducible():
    try:
        import jax
        import jax.numpy as jnp
    except ImportError:
        return

    inp = _base_input()
    params_np = sase1d.params_calc(**inp)
    params = {k: jnp.asarray(v) for k, v in params_np.items()}

    spec = sase1d.make_shot_noise_spec_from_params(
        params_np, npart=inp["npart"], s_steps=inp["s_steps"], iopt=inp["iopt"]
    )
    noise_batch = sase1d.sample_shot_noise_batch_jax(jax.random.PRNGKey(42), spec, 6)

    val1, grad1 = sase1d.mc_value_and_grad_jax(
        params, noise_batch, spec, loss_fn=sase1d.default_loss_final_power_jax
    )
    val2, grad2 = sase1d.mc_value_and_grad_jax(
        params, noise_batch, spec, loss_fn=sase1d.default_loss_final_power_jax
    )

    assert np.allclose(np.asarray(val1), np.asarray(val2))
    assert np.allclose(np.asarray(grad1["kappa_1"]), np.asarray(grad2["kappa_1"]))
    assert grad1["kappa_1"].shape == params["kappa_1"].shape
    assert np.all(np.isfinite(np.asarray(grad1["kappa_1"])))


def test_fel_sim_jax_matches_numpy_single_realization():
    try:
        import jax.numpy as jnp
    except ImportError:
        return

    inp = _base_input()
    inp.update(npart=64, s_steps=8, z_steps=8, P0=1e3)

    params_np = sase1d.params_calc(**inp)
    params_jax = {k: jnp.asarray(v) for k, v in params_np.items()}

    spec = sase1d.make_shot_noise_spec_from_params(
        params_np, npart=inp["npart"], s_steps=inp["s_steps"], iopt=inp["iopt"]
    )
    noise_batch = sase1d.sample_shot_noise_batch_numpy(spec, n_samples=1, seed=5)
    noise_np = {"eta_randn": noise_batch["eta_randn"][0], "theta_rand": noise_batch["theta_rand"][0]}
    noise_jax = {k: jnp.asarray(v) for k, v in noise_np.items()}

    out_np = sase1d.FEL_sim(params_np, noise_np, spec)
    out_jax = sase1d.FEL_sim_jax(params_jax, noise_jax, spec)

    assert np.allclose(np.asarray(out_jax["power_z"]), np.asarray(out_np["power_z"]), rtol=1e-6, atol=1e-3)
    assert np.allclose(np.asarray(out_jax["power_s"]), np.asarray(out_np["power_s"]), rtol=1e-6, atol=1e-3)


def test_mc_value_and_grad_jax_p0_zero_has_finite_gradient():
    try:
        import jax
        import jax.numpy as jnp
    except ImportError:
        return

    inp = _base_input()
    inp.update(P0=0.0)

    params_np = sase1d.params_calc(**inp)
    params = {k: jnp.asarray(v) for k, v in params_np.items()}

    spec = sase1d.make_shot_noise_spec_from_params(
        params_np, npart=inp["npart"], s_steps=inp["s_steps"], iopt=inp["iopt"]
    )
    noise_batch = sase1d.sample_shot_noise_batch_jax(jax.random.PRNGKey(123), spec, 4)

    value, grad = sase1d.mc_value_and_grad_jax(
        params, noise_batch, spec, loss_fn=sase1d.default_loss_pulse_energy_jax
    )

    assert np.isfinite(np.asarray(value))
    assert np.all(np.isfinite(np.asarray(grad["kappa_1"])))


def test_mc_value_and_grad_jax_has_nonzero_seed_gradient_when_seeded():
    try:
        import jax
        import jax.numpy as jnp
    except ImportError:
        return

    inp = _base_input()
    inp.update(P0=1e3)

    params_np = sase1d.params_calc(**inp)
    spec = sase1d.make_shot_noise_spec_from_params(
        params_np, npart=inp["npart"], s_steps=inp["s_steps"], iopt=inp["iopt"]
    )
    noise_batch = sase1d.sample_shot_noise_batch_jax(jax.random.PRNGKey(123), spec, 4)

    def objective(log_e02):
        e02 = jnp.exp(log_e02)
        params = {k: jnp.asarray(v) for k, v in params_np.items()}
        params["E02"] = e02
        value, _ = sase1d.mc_value_and_grad_jax(
            params, noise_batch, spec, loss_fn=sase1d.default_loss_pulse_energy_jax
        )
        return value

    grad_log_e02 = jax.grad(objective)(jnp.log(jnp.asarray(params_np["E02"])))
    assert np.isfinite(np.asarray(grad_log_e02))
    assert abs(float(grad_log_e02)) > 0.0
