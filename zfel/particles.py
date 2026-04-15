import numpy as np


def shot_noise_spec(npart, s_steps, iopt="sase"):
    """
    Describe the random degrees of freedom used to generate shot noise.

    This allows noise sampling to be externalized and passed explicitly to the
    solver as an input xi.
    """
    if iopt == "seeded":
        M = 128
    elif iopt == "sase":
        M = 32
    else:
        raise ValueError(f"Unknown iopt: {iopt}")

    nb = int(np.round(npart / M))
    if M * nb != npart:
        raise ValueError("npart must be a multiple of beamlet size M")

    return {
        "npart": int(npart),
        "s_steps": int(s_steps),
        "iopt": iopt,
        "M": int(M),
        "nb": int(nb),
    }


def sample_shot_noise_numpy(spec, seed=None, rng=None):
    """
    Sample explicit shot-noise variables (xi) for one simulation realization.
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    s_steps = int(spec["s_steps"])
    nb = int(spec["nb"])
    M = int(spec["M"])
    iopt = spec["iopt"]

    noise = {"eta_randn": rng.standard_normal((s_steps, nb))}
    if iopt == "sase":
        noise["theta_rand"] = rng.random((s_steps, nb, M))
    else:
        noise["theta_rand"] = np.zeros((s_steps, nb, M))
    return noise


def bucket_from_shot_noise(spec, *, gbar, delg, Ns):
    """
    Deterministically build bucket data from explicit shot-noise variables.

    Parameters
    ----------
    spec:
        dict from shot_noise_spec(...), augmented with:
        - eta_randn: shape (s_steps, nb)
        - theta_rand: shape (s_steps, nb, M), used for iopt='sase'
    """
    s_steps = int(spec["s_steps"])
    npart = int(spec["npart"])
    nb = int(spec["nb"])
    M = int(spec["M"])
    iopt = spec["iopt"]

    eta_randn = np.asarray(spec["eta_randn"])
    theta_rand = np.asarray(spec["theta_rand"])

    if eta_randn.shape != (s_steps, nb):
        raise ValueError("eta_randn has incompatible shape")
    if theta_rand.shape != (s_steps, nb, M):
        raise ValueError("theta_rand has incompatible shape")

    base = 2 * np.pi * (np.arange(M) + 1) / M
    theta_base = np.tile(base, nb)[np.newaxis, :]
    eta_beamlet = delg * eta_randn + gbar
    eta_init = np.repeat(eta_beamlet, M, axis=1)

    if iopt == "sase":
        effnoise = np.sqrt(3 * M / (Ns / nb))
        thet_init = theta_base + 2 * effnoise * theta_rand.reshape(s_steps, npart)
    elif iopt == "seeded":
        thet_init = np.broadcast_to(theta_base, (s_steps, npart))
    else:
        raise ValueError(f"Unknown iopt: {iopt}")

    return {
        "thet_init": np.asarray(thet_init),
        "eta_init": np.asarray(eta_init),
        "N_real": np.ones(s_steps),
        "s_steps": s_steps,
    }


def general_load_bucket(
    npart,
    Ns,
    coopLength,
    s_steps,
    dels,
    hist_rule="square-root",
    particle_position=None,
    gbar=0,
    delg=None,
    iopt="sase",
):
    """
    random initialization of the beam load_bucket
    inputs:
    npart               # n-macro-particles per bucket
    gbar                # scaled detune parameter
    delg                # Gaussian energy spread in units of rho
    iopt                # 'sase' or 'seeded'
    Ns                  # N electrons per s-slice at maximum current
    coopLength          # cooperation length
    particle_position   # particle information with positions in meter and eta
    s_steps             # n-sample points along bunch length
    dels                # integration step in s0
    hist_rule           # different rules to select number of intervals to generate the histogram of eta value in a bucket

    outputs:
    thet_init           # all buckets macro particles position
    eta_init            # all buckets macro particles relative energy
    N_real              # real number of particles along the beam
    """
    if particle_position is None:
        thet_init = np.zeros((s_steps, npart))
        eta_init = np.zeros((s_steps, npart))
        for j in range(s_steps):
            [thet0, eta0] = load_bucket(
                npart, gbar, delg, Ns, iopt=iopt
            )  # load each bucket
            thet_init[j, :] = thet0
            eta_init[j, :] = eta0
        N_real = np.ones(s_steps)
    else:
        # load particle information and classify them to different intervals
        s_all = particle_position[:, 0]
        eta_all = particle_position[:, 1]
        s_steps = (
            int(np.max(s_all) / (dels * coopLength)) + 1
            if np.max(s_all) % (dels * coopLength) != 0
            else np.max(s_all) / (dels * coopLength)
        )
        N_input = np.zeros(s_steps)
        eta_step = [[] for x in range(s_steps)]
        for k in range(s_all.shape[0]):
            location = int(s_all[k] / (dels * coopLength))
            N_input[location] += 1
            eta_step[location].append(eta_all[k])
        N_real = N_input / np.max(N_input) * Ns
        # generate theta and eta
        thet_init = np.zeros((s_steps, npart))
        eta_init = np.zeros((s_steps, npart))
        for k in range(s_steps):
            if N_real[k] == 0:
                thet_init[k, :] = np.random.rand() * 2 * np.pi
                eta_init[k, :] = np.zeros(npart)
            else:
                thet_init[k, :] = make_theta(npart, N_real[k])
                eta_init[k, :] = make_eta(eta_step[k], npart, hist_rule)

    return {
        "thet_init": thet_init,
        "eta_init": eta_init,
        "N_real": N_real,
        "s_steps": s_steps,
    }


def load_bucket(n, gbar, delg, Ns, iopt="sase"):
    """
    random initialization of the beam load_bucket
    inputs:
    n               # n-macro-particles per bucket
    gbar            # scaled detune parameter
    delg            # Gaussian energy spread in units of rho
    iopt            # 'sase' or 'seeded'
    Ns              # N electrons per s-slice
    outputs:
    thet            # bucket macro particles position
    eta             # bucket macro particles relative energy
    """
    nmax = 10000
    if n > nmax:
        raise ValueError("increase nmax, subr load")

    eta = np.zeros(n)
    thet = np.zeros(n)
    if iopt == "seeded":
        M = 128  # number of particles in each beamlet
        nb = int(
            np.round(n / M)
        )  # number of beamlet via Fawley between 64 to 256 (x16=1024 to 4096)
        if M * nb != n:
            raise ValueError("n must be a multiple of 4")
        for i in range(nb):
            etaa = delg * np.random.randn() + gbar
            # etaa=delg*(np.random.rand()-0.5)+gbar
            for j in range(M):
                eta[i * M + j] = etaa
                thet[i * M + j] = 2 * np.pi * (j + 1) / M
    elif iopt == "sase":
        M = 32  # number of particles in each beamlet
        nb = int(
            np.round(n / M)
        )  # number of beamlet via Fawley between 64 to 256 (x16=1024 to 4096)
        if M * nb != n:
            raise ValueError("n must be a multiple of 4")
        effnoise = np.sqrt(3 * M / (Ns / nb))  # Penman algorithm for Ns/nb >> M
        for i in range(nb):
            etaa = delg * np.random.randn() + gbar
            # etaa=delg*(np.random.rand()-0.5)+gbar
            for j in range(M):
                eta[i * M + j] = etaa
                thet[i * M + j] = (
                    2 * np.pi * (j + 1) / M + 2 * np.random.rand() * effnoise
                )
    else:
        raise ValueError(f"Unknown iopt: {iopt}")

    return thet, eta


def make_theta(n, N_real_bucket):
    """
    random initialization of a bucket's particle positions
    inputs:
    n               # n-macro-particles per bucket
    N_real_bucket   # real number of particles in a bucket
    outputs:
    thet            # macro particles position in a bucket
    """

    thet = np.zeros(n)
    M = 32  # number of particles in each beamlet
    nb = int(
        np.round(n / M)
    )  # number of beamlet via Fawley between 64 to 256 (x16=1024 to 4096)
    if M * nb != n:
        raise ValueError("n must be a multiple of 4")

    effnoise = np.sqrt(3 * M / (N_real_bucket / nb))  # Penman algorithm for Ns/nb >> M
    for i in range(nb):
        for j in range(M):
            thet[i * M + j] = 2 * np.pi * (j + 1) / M + 2 * np.random.rand() * effnoise
    return thet


def make_eta(eta_step_bucket, npart, hist_rule="square-root"):
    """
    eta_step_bucket     # input particles' eta values in a bucket
    npart               # n-macro-particles per bucket
    hist_rule          # different rules to select number of intervals to generate the histogram of eta value in a bucket
    outputs:
    eta_sampled         # sampled macro particles relative energy in a bucket
    """

    lowbound = np.min(eta_step_bucket)
    upbound = np.max(eta_step_bucket) + 1e-10
    pts = len(eta_step_bucket)
    if hist_rule == "square-root":
        hist_num = int(np.sqrt(pts))
    elif hist_rule == "sturges":
        hist_num = int(np.log2(pts)) + 1
    elif hist_rule == "rice-rule":
        hist_num = int(2 * pts ** (1 / 3))
    eta_hist = np.zeros(hist_num)
    eta_hist, bins = np.histogram(
        np.array(eta_step_bucket), bins=np.linspace(lowbound, upbound, num=hist_num + 1)
    )
    # plt.figure()
    # _=plt.hist(np.array(eta_step_bucket),bins=np.linspace(lowbound, upbound, num=hist_num+1))
    # plt.title('Input eta histogram')
    eta_hist = eta_hist / np.sum(eta_hist)

    # make cdf
    eta_cdf = np.zeros(eta_hist.shape)
    eta_cdf[0] = eta_hist[0]
    for j in range(1, hist_num):
        eta_cdf[j] = eta_hist[j] + eta_cdf[j - 1]
    eta_cdf = np.concatenate((np.zeros(1), eta_cdf))

    # make eta
    x = np.random.rand(npart)
    eta_sampled = np.interp(x, eta_cdf, bins)
    # plt.figure()
    # _=plt.hist(eta_sampled,bins=np.linspace(lowbound, upbound, num=hist_num+1))
    # plt.title('Sampled eta histogram')
    return eta_sampled
