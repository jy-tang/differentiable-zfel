import numpy as np
import scipy
from scipy import special

from zfel.particles import (
    general_load_bucket,
    shot_noise_spec,
    sample_shot_noise_numpy,
    bucket_from_shot_noise,
)
from zfel.fel import FEL_process_complex, final_calc

# Some constant values
alfvenCurrent = 17045.0  # Alfven current ~ 17 kA
mc2 = 0.51099906e6  # 510.99906E-3      # Electron rest mass in eV
c = 2.99792458e8  # light speed in meter
e = 1.60217733e-19  # electron charge in Coulomb
epsilon_0 = 8.85418782e-12  # electric constant
hbar = 6.582e-16  # in eV


def sase(inp_struct):
    """
    SASE 1D FEL run function
    TODO: needs updating
    Input:
    npart                       # n-macro-particles per bucket
    s_steps                     # n-sample points along bunch length
    z_steps                     # n-sample points along undulator
    energy                      # electron energy [eV]
    eSpread                     # relative rms energy spread [1]
    emitN                       # normalized transverse emittance [m-rad]
    currentMax                  # peak current [Ampere]
    beta                        # mean beta [meter]
    unduPeriod                  # undulator period [meter]
    unduK                       # undulator parameter, K [1]
    unduL                       # length of undulator [meter]
    radWavelength               # seed wavelength? [meter], used only in single-freuqency runs
    dEdz                        # rate of relative energy gain or taper [keV/m], optimal~130
    iopt                        # 'sase' or 'seeded'
    P0                          # small seed input power [W]
    random_seed                 # A random number seed. Default: None
    particle_position           # particle information with positions in meter and eta. Default: None
    hist_rule                   # different rules to select number of intervals to generate the histogram of eta value in a bucket

    Output:
    (TODO: needs updating)
    z                           # longitudinal steps along undulator
    power_z                     # power profile along undulator
    s                           # longitudinal steps along beam
    power_s                     # power profile along beam
    rho                         # FEL Pierce parameter
    detune                      # deviation from the central energy
    field                       # final output field along beam
    field_s                     # output field along beam for different z position
    gainLength                  # 1D FEL gain Length
    resWavelength               # resonant wavelength
    thet_out                    # output phase
    eta_out                     # output energy in unit of mc2
    bunching                    # bunching factor
    spectrum                    # spectrum power
    freq                        # frequency in eV
    Ns                          # real number of examples
    """

    params = params_calc(**inp_struct)
    bucket_data = fixed_or_external_bucket_data(
        params=params,
        npart=inp_struct["npart"],
        s_steps=inp_struct["s_steps"],
        particle_position=inp_struct.get("particle_position"),
        hist_rule=inp_struct.get("hist_rule", "square-root"),
        iopt=inp_struct.get("iopt", "sase"),
        random_seed=inp_struct.get("random_seed"),
    )
    return sase_from_initial_conditions(params, bucket_data)


def fixed_or_external_bucket_data(
    *,
    params,
    npart,
    s_steps,
    particle_position=None,
    hist_rule="square-root",
    iopt="sase",
    random_seed=None,
):
    """
    Build bucket data outside the deterministic FEL solver.

    This function can be replaced by externally generated fixed bucket data
    when using autodiff workflows.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    bucket_params = {
        "npart": npart,
        "Ns": params["Ns"],
        "coopLength": params["coopLength"],
        "particle_position": particle_position,
        "s_steps": s_steps,
        "dels": params["dels"],
        "hist_rule": hist_rule,
        "gbar": params["gbar"],
        "delg": params["delg"],
        "iopt": iopt,
    }
    return general_load_bucket(**bucket_params)


def sase_from_initial_conditions(params, bucket_data):
    """
    Deterministic FEL solve from fixed parameters and bucket initial conditions.
    """
    p = params
    b = bucket_data
    npart = b["thet_init"].shape[1]
    s_steps = int(b["s_steps"])
    z_steps = len(p["kappa_1"])

    FEL_data = FEL_process_complex(
        npart,
        z_steps,
        p["kappa_1"],
        p["density"],
        p["Kai"],
        p["ku"],
        p["delt"],
        p["dels"],
        p["deta"],
        b["thet_init"],
        b["eta_init"],
        b["N_real"],
        s_steps,
        E02=p["E02"],
    )

    final_data = final_calc(
        FEL_data["Er"],
        FEL_data["Ei"],
        s_steps,
        z_steps,
        p["kappa_1"],
        p["density"],
        p["Kai"],
        p["Pbeam"],
        p["delt"],
        p["dels"],
    )

    output = FEL_data
    output.update(final_data)
    output["params"] = params

    s = (
        np.arange(1, s_steps + 1) * p["dels"] * p["coopLength"]
    )  # longitundinal steps along beam in m
    z = np.arange(1, z_steps + 1) * p["delt"]  # longitundinal steps along undulator in m
    bunchLength = s[-1]  # beam length in meter
    bunch_steps = np.round(
        bunchLength / p["delt"] / p["coopLength"]
    )  # rms (Gaussian) or half width (flattop) bunch length in s_step
    output["s"] = s
    output["z"] = z
    output["bunchLength"] = bunchLength
    output["bunch_steps"] = bunch_steps

    omega = hbar * 2.0 * np.pi / (p["resWavelength"] / c)
    df = hbar * 2.0 * np.pi * 1 / (bunchLength / c)
    output["freq"] = np.linspace(
        omega - s_steps / 2 * df, omega + s_steps / 2 * df, s_steps
    )

    return output


def sase_from_initial_conditions_jax(params, bucket_data):
    """
    JAX-backed deterministic FEL solve from fixed parameters and bucket data.

    The interface matches sase_from_initial_conditions(params, bucket_data).
    """
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:
        raise ImportError(
            "JAX is required for sase_from_initial_conditions_jax. "
            "Install jax and jaxlib to enable autodiff."
        ) from exc

    # Match NumPy's default float64 behavior for better numerical agreement.
    jax.config.update("jax_enable_x64", True)

    from zfel.fel_jax import FEL_process_complex_jax, final_calc_jax

    def _to_jax_float64(x):
        arr = jnp.asarray(x)
        if jnp.issubdtype(arr.dtype, jnp.integer):
            return arr
        return arr.astype(jnp.float64)

    p = {k: _to_jax_float64(v) for k, v in params.items()}
    b = {
        k: (_to_jax_float64(v) if k != "s_steps" else int(v))
        for k, v in bucket_data.items()
    }

    npart = b["thet_init"].shape[1]
    s_steps = int(b["s_steps"])
    z_steps = len(p["kappa_1"])

    FEL_data = FEL_process_complex_jax(
        npart,
        z_steps,
        p["kappa_1"],
        p["density"],
        p["Kai"],
        p["ku"],
        p["delt"],
        p["dels"],
        p["deta"],
        b["thet_init"],
        b["eta_init"],
        b["N_real"],
        s_steps,
        E02=p["E02"],
    )

    final_data = final_calc_jax(
        FEL_data["Er"],
        FEL_data["Ei"],
        s_steps,
        z_steps,
        p["kappa_1"],
        p["density"],
        p["Kai"],
        p["Pbeam"],
        p["delt"],
        p["dels"],
    )

    output = FEL_data
    output.update(final_data)
    output["params"] = p

    s = jnp.arange(1, s_steps + 1) * p["dels"] * p["coopLength"]
    z = jnp.arange(1, z_steps + 1) * p["delt"]
    bunchLength = s[-1]
    bunch_steps = jnp.round(bunchLength / p["delt"] / p["coopLength"])
    output["s"] = s
    output["z"] = z
    output["bunchLength"] = bunchLength
    output["bunch_steps"] = bunch_steps

    omega = hbar * 2.0 * jnp.pi / (p["resWavelength"] / c)
    df = hbar * 2.0 * jnp.pi * 1 / (bunchLength / c)
    output["freq"] = jnp.linspace(omega - s_steps / 2 * df, omega + s_steps / 2 * df, s_steps)

    return output


def FEL_sim(params, noise, noise_spec):
    """
    Deterministic single-realization FEL simulation with explicit shot noise.
    """
    bucket_data = bucket_from_shot_noise(
        {
            **noise_spec,
            "eta_randn": noise["eta_randn"],
            "theta_rand": noise["theta_rand"],
        },
        gbar=params["gbar"],
        delg=params["delg"],
        Ns=params["Ns"],
    )
    return sase_from_initial_conditions(params, bucket_data)


def FEL_sim_jax(params, noise, noise_spec):
    """
    JAX deterministic single-realization FEL simulation with explicit shot noise.
    """
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:
        raise ImportError(
            "JAX is required for FEL_sim_jax. Install jax and jaxlib."
        ) from exc

    jax.config.update("jax_enable_x64", True)
    from zfel.noise_jax import bucket_from_shot_noise_jax

    def _to_jax_float64(x):
        arr = jnp.asarray(x)
        if jnp.issubdtype(arr.dtype, jnp.integer):
            return arr
        return arr.astype(jnp.float64)

    p = {k: _to_jax_float64(v) for k, v in params.items()}
    n = {k: _to_jax_float64(v) for k, v in noise.items()}

    bucket_data = bucket_from_shot_noise_jax(
        noise_spec,
        n,
        gbar=p["gbar"],
        delg=p["delg"],
        Ns=p["Ns"],
    )
    return sase_from_initial_conditions_jax(p, bucket_data)


def make_shot_noise_spec_from_params(params, *, npart, s_steps, iopt="sase"):
    """
    Build a reusable shot-noise specification for explicit-noise workflows.
    """
    # params is currently unused but kept to align with theta/physics workflows.
    _ = params
    return shot_noise_spec(npart=npart, s_steps=s_steps, iopt=iopt)


def sample_shot_noise_batch_numpy(noise_spec, n_samples, seed=0):
    """
    Generate a batch of explicit noise realizations (NumPy), useful for CRN.
    """
    rng = np.random.default_rng(seed)
    noises = [sample_shot_noise_numpy(noise_spec, rng=rng) for _ in range(int(n_samples))]
    return {
        "eta_randn": np.stack([n["eta_randn"] for n in noises], axis=0),
        "theta_rand": np.stack([n["theta_rand"] for n in noises], axis=0),
    }


def sample_shot_noise_batch_jax(key, noise_spec, n_samples):
    """Generate a batch of explicit noise realizations (JAX)."""
    try:
        from zfel.noise_jax import sample_shot_noise_batch_jax as _sample
    except ImportError as exc:
        raise ImportError(
            "JAX is required for sample_shot_noise_batch_jax. Install jax and jaxlib."
        ) from exc
    return _sample(key, noise_spec, n_samples)


def default_loss_final_power(output):
    """Default scalar objective: final-z output power."""
    return output["power_z"][-1]


def default_loss_pulse_energy(output):
    """Pulse-energy-like scalar objective from final-z longitudinal profile."""
    return np.sum(output["power_s"][-1])


def default_loss_final_power_jax(output):
    """JAX default scalar objective: final-z output power."""
    return output["power_z"][-1]


def default_loss_pulse_energy_jax(output):
    """JAX pulse-energy-like scalar objective from final-z longitudinal profile."""
    return output["power_s"][-1].sum()


def mc_objective_jax(params, noise_batch, noise_spec, loss_fn=None):
    """
    Monte Carlo objective:
        J_hat(theta) = (1/N) sum_i loss_fn(FEL_sim_jax(theta, noise_i))
    """
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:
        raise ImportError(
            "JAX is required for mc_objective_jax. Install jax and jaxlib."
        ) from exc

    if loss_fn is None:
        loss_fn = default_loss_final_power_jax

    jax.config.update("jax_enable_x64", True)

    def _cast_leaf(x):
        arr = jnp.asarray(x)
        if jnp.issubdtype(arr.dtype, jnp.integer):
            return arr.astype(jnp.float64)
        return arr

    params = jax.tree_util.tree_map(_cast_leaf, params)

    def _one(noise_i):
        out = FEL_sim_jax(params, noise_i, noise_spec)
        return loss_fn(out)

    losses = jax.vmap(_one)(noise_batch)
    return jnp.mean(losses)


def mc_value_and_grad_jax(params, noise_batch, noise_spec, loss_fn=None):
    """
    Return Monte Carlo objective value and gradient w.r.t. params.
    """
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:
        raise ImportError(
            "JAX is required for mc_value_and_grad_jax. Install jax and jaxlib."
        ) from exc

    jax.config.update("jax_enable_x64", True)

    def _cast_leaf(x):
        arr = jnp.asarray(x)
        if jnp.issubdtype(arr.dtype, jnp.integer):
            return arr.astype(jnp.float64)
        return arr

    params_cast = jax.tree_util.tree_map(_cast_leaf, params)

    def _obj(p):
        return mc_objective_jax(p, noise_batch, noise_spec, loss_fn=loss_fn)

    return jax.value_and_grad(_obj)(params_cast)


def params_calc(
    *,  # Require kwargs explicitly to avoid typos
    npart=512,
    s_steps=200,
    z_steps=200,
    energy=4313.34e6,
    eSpread=0,
    emitN=1.2e-6,
    currentMax=3400,
    beta=26,
    unduPeriod=0.03,
    unduK=3.5,
    unduL=70,
    iopt="sase",
    P0=0,
    random_seed=None,
    particle_position=None,
    radWavelength=None,
    hist_rule="square-root"
):
    """
    calculating intermediate parameters
    """
    # random_seed is accepted for backward compatibility but is intentionally
    # ignored here. Keep stochastic behavior in bucket loading functions.

    # Check if unduK is array. Otherwise, fill it out.
    if not isinstance(unduK, np.ndarray):
        unduK = np.full(z_steps, unduK)

    unduJJ = scipy.special.jv(0, unduK**2 / (4 + 2 * unduK**2)) - scipy.special.jv(
        1, unduK**2 / (4 + 2 * unduK**2)
    )  # undulator JJ
    gamma0 = energy / mc2  # central energy of the beam in unit of mc2
    sigmaX2 = (
        emitN * beta / gamma0
    )  # rms transverse size, divergence of the electron beam

    # Needed for FEL_process
    kappa_1 = (
        e * unduK * unduJJ / 4 / epsilon_0 / gamma0
    )  # Eq. 4.10 in Kim, Huang, Lindberg (2017)
    Kai = e * unduK * unduJJ / (2 * gamma0**2 * mc2 * e)  # Ibid.
    density = currentMax / (e * c * 2 * np.pi * sigmaX2)
    ku = 2 * np.pi / unduPeriod

    rho = (0.5 / gamma0) * (
        (currentMax / alfvenCurrent)
        * (unduPeriod * unduK * unduJJ / (2 * np.pi)) ** 2
        / (2 * sigmaX2)
    ) ** (
        1 / 3
    )  # FEL Pierce parameter

    resWavelength = (
        unduPeriod * (1 + unduK[0] ** 2 / 2.0) / (2 * gamma0**2)
    )  # resonant wavelength

    if radWavelength is None:
        radWavelength = resWavelength

    Pbeam = energy * currentMax  # beam power [W]
    coopLength = resWavelength / unduPeriod  # cooperation length
    # cs0  = bunchLength/coopLength                          # bunch length in units of cooperation length
    z0 = unduL  # wiggler length
    delt = z0 / z_steps  # integration step in z0 ~ 0.1 gain length
    dels = delt  # integration step in s0 must be same as in z0
    E02 = density * kappa_1[0] * P0 / Pbeam / Kai[0]  # scaled input power
    gbar = resWavelength / radWavelength - 1.0  # scaled detune parameter

    delg = eSpread  # Gaussian energy spread in units of rho
    Ns = (
        currentMax * unduL / unduPeriod / z_steps * resWavelength / c / e
    )  # N electrons per s-slice [ ]

    deta = np.sqrt((1 + 0.5 * unduK[0] ** 2) / (1 + 0.5 * unduK**2)) - 1

    params = {
        "unduJJ": unduJJ,
        "gamma0": gamma0,
        "sigmaX2": sigmaX2,
        "kappa_1": kappa_1,
        "density": density,
        "Kai": Kai,
        "ku": ku,
        "resWavelength": resWavelength,
        "Pbeam": Pbeam,
        "coopLength": coopLength,
        "z0": z0,
        "delt": delt,
        "dels": dels,
        "E02": E02,
        "gbar": gbar,
        "delg": delg,
        "Ns": Ns,
        "deta": deta,
        "rho": rho,
    }

    return params
