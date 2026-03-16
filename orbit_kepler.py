import argparse
import datetime

import numpy as np
import matplotlib.pyplot as plt
import scopesim as sim
import scopesim_templates as sim_tp
from astropy import units as u
from astropy.table import Table
from astropy.io import fits as astropy_fits
import sep


PIXEL_SCALE = 0.0015        # arcsec per pixel
SEP_THRESHOLD_MAX = 250
t_obs = 2030

names_val = spectral_val = kmag_val = None
a_val = a_unc = e_val = e_unc = None
i_val = i_unc = Omega_val = Omega_unc = None
w_val = w_unc = Tp_val = Tp_unc = Per_val = Per_unc = None
orbit_table = None


def load_data(filename="J_ApJ_837_30_table3.dat.fits"):

    global names_val, spectral_val, kmag_val
    global a_val, a_unc, e_val, e_unc, i_val, i_unc
    global Omega_val, Omega_unc, w_val, w_unc, Tp_val, Tp_unc, Per_val, Per_unc
    global orbit_table

    with astropy_fits.open(filename) as hdul:
        d = hdul[1].data

    def _str(col):
        return np.array([s.decode().strip() if isinstance(s, bytes) else s.strip() for s in col])

    names_val    = _str(d['Star'])
    spectral_val = _str(d['SpT'])
    kmag_val     = d['Kmag'].astype(float)

    a_val,     a_unc     = d['a'].astype(float),     d['e_a'].astype(float)
    e_val,     e_unc     = d['e'].astype(float),     d['e_e'].astype(float)
    i_val,     i_unc     = d['i'].astype(float),     d['e_i'].astype(float)
    Omega_val, Omega_unc = d['Omega'].astype(float), d['e_Omega'].astype(float)
    w_val,     w_unc     = d['w'].astype(float),     d['e_w'].astype(float)
    Tp_val,    Tp_unc    = d['Tp'].astype(float),    d['e_Tp'].astype(float)
    Per_val,   Per_unc   = d['Per'].astype(float),   d['e_Per'].astype(float)

    orbit_table = orbitTable(t_obs, names_val, a_val, e_val, i_val,
                             Omega_val, w_val, Tp_val, Per_val)

def resolveStars(star_name=None):
    '''
    Assigns the input stars to the corresponding out of the data set.

    :param star_name: None, "all, "S2", "S2, S4, S8"
    :return: (id, label), (None, None)
    '''

    if star_name is None:
        return np.arange(len(names_val)), "all stars"

    # normalise to list
    names_req = [star_name] if isinstance(star_name, str) else list(star_name)

    if len(names_req) == 1 and names_req[0].lower() == "all":
        return np.arange(len(names_val)), "all stars"

    idxs = []
    not_found = []
    for name in names_req:
        matches = np.where(names_val == name)[0]
        if len(matches) == 0:
            not_found.append(name)
        else:
            idxs.append(matches[0])

    if not_found:
        print(f"Star(s) not found: {not_found}. Available: {list(names_val)}")
    if not idxs:
        return None, None

    label = ", ".join(names_val[i] for i in idxs)
    return np.array(idxs), label

# Newton-Raphson Method für kepler glg: M = E - e*sin(E)
def kepler(M, e, epsilon=1e-9, max_it=1000):
    M = np.mod(M, 2 * np.pi) # 0 <= M <= 2pi
    E = np.copy(M)

    for i in range(max_it):
        delta = (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
        E -= delta
        if np.all(np.abs(delta) < epsilon):
            break

    return E

def orbitalPosition(t, a, e, i, Omega, w_, Tp, Per):
    w =  w_ - Omega
    i, Omega, w = np.radians([i, Omega, w])
    M = (2 * np.pi / Per) * (t - Tp) # mean anomaly
    E = kepler(M, e) # eccentric anomaly
    f = 2 * np.arctan(np.sqrt((1 + e) / (1-e)) * np.tan(E / 2)) # true anomaly
    r = a * (1 - e * np.cos(E))

    dec = r * (np.cos(Omega) * np.cos(w_ + f) - np.sin(Omega) * np.sin(w_ + f) * np.cos(i))
    ra = -r * (np.sin(Omega) * np.cos(w_ + f) + np.cos(Omega) * np.sin(w_ + f) * np.cos(i))
    z_ = r * (np.sin(w_ + f) * np.sin(i))

    return ra, dec, z_

def orbitalVelocity(t, a, e, i, Omega, w, Tp, Per):

    n = 2 * np.pi / Per  # mean motion [rad/yr]
    M = n * (t - Tp)     # mean anomaly
    E = kepler(M, e)     # eccentric anomaly
    r = a * (1 - e * np.cos(E))

    v = n * a * np.sqrt(1 - e**2 * np.cos(E)**2) / (1 - e * np.cos(E)) # vis-viva speed in the orbital plane
    return v

def orbitTable(t_obs, names_arr, a_arr, e_arr, i_arr, Omega_arr, w_arr, Tp_arr, Per_arr,
               kmag_arr=None, spectral_arr=None):
    if kmag_arr is None:
        kmag_arr = kmag_val
    if spectral_arr is None:
        spectral_arr = spectral_val
    x_pos = []
    y_pos = []

    for id in range(len(names_arr)):
        x, y, z = orbitalPosition(t_obs, a_arr[id], e_arr[id], i_arr[id], Omega_arr[id], w_arr[id], Tp_arr[id], Per_arr[id])

        x_pos.append(x)
        y_pos.append(y)

    table = Table(
        names=["x", "y", "ref", "mag", "type"],
        data=[x_pos, y_pos, names_arr, kmag_arr, spectral_arr],
        units=[u.arcsec, u.arcsec, None, None, None]
    )
    table['x'].unit = 'arcsec'
    table['y'].unit = 'arcsec'

    return table

def simulate(star_name=None):

    #sim.download_packages(["Armazones", "ELT", "MICADO"])

    idxs, label = resolveStars(star_name)
    if idxs is None:
        return None
    tbl = orbit_table[idxs]

    cmds = sim.UserCommands(use_instrument="MICADO", set_modes=["SCAO", "IMG_1.5mas"])

    # EXPTIME = 3600 = ndit * dit
    cmds["!DET.dit"] = 30
    cmds["!DET.ndit"] = 120
    micado = sim.OpticalTrain(cmds)
    fixed_stars = sim_tp.stellar.stars(filter_name="H",
                                       amplitudes=tbl['mag'] * u.mag,  # [u.mag, u.ABmag, u.Jy]
                                       spec_types=np.full(len(idxs), 'A0V'),
                                       x=tbl['x'], y=tbl['y'])  # [u.arcsec]
    micado.observe(fixed_stars)
    hdus = micado.readout()
    sim_image = hdus[0][1].data #numpy array
    return sim_image

def findStars(star_name=None):
    '''
    Run source detection on simulated image

    :param star_name:
    :return: (measured_x, measured_y)
    '''
    # get position in arcsec from pixels
    idxs, label = resolveStars(star_name)
    if idxs is None:
        return None, None
    tbl = orbit_table[idxs]

    im = simulate(star_name)

    if not im.dtype.isnative:
        im = im.byteswap(inplace=True)
        im = im.view(im.dtype.newbyteorder('='))

    threshold = 1.0
    while threshold <= SEP_THRESHOLD_MAX:
        bkg = sep.Background(im)
        sources = sep.extract(im - bkg, threshold, err=bkg.globalrms)
        threshold += 1
        if len(sources) == len(idxs):
            print(f"Required threshold: {threshold})")
            break
    else:
        print(f"Warning: could not isolate exactly {len(idxs)} source(s) "
              f"within SEP threshold {SEP_THRESHOLD_MAX}; got {len(sources)}.")

    # Get positions
    print(f"Found {len(sources)} stars with SEP threshold: {threshold - 1}")

    # Convert measured pixel positions to arcsec
    x_ref_pix = im.shape[1] / 2
    y_ref_pix = im.shape[0] / 2

    measured_x = (sources['x'] - x_ref_pix) * PIXEL_SCALE
    measured_y = (sources['y'] - y_ref_pix) * PIXEL_SCALE

    # Compare with calculated positions
    calc_x = np.array(tbl['x'])
    calc_y = np.array(tbl['y'])

    dx_list = []
    dy_list = []
    sep_list = []

    # For each measured star, find nearest calculated
    for mx, my in zip(measured_x, measured_y):
        distances = np.sqrt((calc_x - mx) ** 2 + (calc_y - my) ** 2)
        nearest_idx = np.argmin(distances)

        dx_list.append(mx - calc_x[nearest_idx])
        dy_list.append(my - calc_y[nearest_idx])
        sep_list.append(distances[nearest_idx])

    dx = np.array(dx_list)
    dy = np.array(dy_list)
    sep_ = np.array(sep_list)


    for i, (mx, my) in enumerate(zip(measured_x, measured_y)):
        print(f"{tbl['ref'][i]}:")
        print(f"  true:     ({tbl['x'][i]: .5f}, {tbl['y'][i]: .5f}) arcsec")
        print(f"  measured: ({mx: .5f}, {my: .5f}) arcsec")
        print(f"  error:    ({np.abs(mx - tbl['x'][i]): .5f}, {np.abs(my - tbl['y'][i]): .5f}) arcsec")
        print()

    print(f"Matched {len(dx)} stars")
    print(f"Mean offset: ΔX = {np.mean(dx):.4f} ± {np.std(dx):.4f} arcsec")
    print(f"             ΔY = {np.mean(dy):.4f} ± {np.std(dy):.4f} arcsec")
    print(f"RMS error: {np.sqrt(np.mean(sep_ ** 2)):.4f} arcsec")
    print(f"Median separation: {np.median(sep_):.4f} arcsec")

    return measured_x, measured_y

def positionPolt(star_name=None, t_start=None, t_end=None):
    # position plot
    idxs, label = resolveStars(star_name)
    if idxs is None:
        return

    t_start = t_start if t_start is not None else 2030
    t_end   = t_end   if t_end   is not None else 2050
    t_plot  = np.linspace(t_start, t_end, 200)

    fig, ax = plt.subplots()

    scatter_objects = []

    for val in idxs:
        x, y, z = orbitalPosition(t_plot, a_val[val], e_val[val], i_val[val], Omega_val[val], w_val[val], Tp_val[val],
                                  Per_val[val])
        sc = ax.scatter(x, y, marker='x', s=1, label=names_val[val])
        scatter_objects.append(sc)

    ax.scatter(0, 0, color='black', marker='+', s=80, zorder=5)

    ax.set_title(f'Calculated orbits  [{t_start}–{t_end}]')
    ax.set_xlabel('R.A. ["]')
    ax.set_ylabel('Dec. ["]')
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)

    leg = ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.0, -0.05, 1.0, 0.1),
        ncol=len(idxs),
        mode='expand',
        frameon=False,
        fontsize=8,
        handlelength=0,
        handletextpad=0,
        borderaxespad=0
    )
    for handle in leg.legend_handles:
        handle.set_visible(False)
    for sc, text in zip(scatter_objects, leg.get_texts()):
        text.set_color(sc.get_facecolor()[0])
    ax.add_artist(leg)

    return plt.show()

def velocityPlot(star_name=None, t_start=None, t_end=None):
    idxs, label = resolveStars(star_name)
    if idxs is None:
        return

    t_start = t_start if t_start is not None else 1992
    t_end   = t_end   if t_end   is not None else 2012
    t_plot  = np.linspace(t_start, t_end, 200)

    for val in idxs:
        v = orbitalVelocity(t_plot, a_val[val], e_val[val], i_val[val], Omega_val[val], w_val[val], Tp_val[val],
                            Per_val[val])
        plt.plot(t_plot, v, label=names_val[val])

    plt.xlabel('t [yr]')
    plt.ylabel(r'$v$ [km/s]')
    plt.title(f'Orbital velocities [{t_start}–{t_end}]')
    if star_name and not (len(star_name) == 1 and star_name[0].lower() == "all"):
        plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.show()
    return

def comparePlot(star_name=None):
    idxs, label = resolveStars(star_name)
    if idxs is None:
        return
    tbl = orbit_table[idxs]

    measured_x, measured_y = findStars(star_name)
    if measured_x is None:
        return

    # Print results
    fig, ax = plt.subplots()

    line1_objects = []

    for i, (mx, my) in enumerate(zip(measured_x, measured_y)):
        ax1 = ax.scatter(tbl['x'][i], tbl['y'][i], marker='+', label=tbl['ref'][i], color='grey')
        ax2 = ax.scatter(mx, my, marker='x')
        line1_objects.append(ax2)

    ax.set_xlabel('R.A. ["]')
    ax.set_ylabel('Dec. ["]')
    ax.set_title(f'Calculated (+) vs Simulated (×)')
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)

    legend1 = ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.0, -0.05, 1.0, 0.1),
        ncol=len(idxs),
        mode='expand',
        frameon=False,
        fontsize=8,
        handlelength=0,
        handletextpad=0,
        borderaxespad=0
    )
    for handle in legend1.legend_handles:
        handle.set_visible(False)

    for line, text in zip(line1_objects, legend1.get_texts()):
        curr_color = line.get_facecolor()[0]
        text.set_color(curr_color)
    ax.add_artist(legend1)

    marker_handles = [
        ax.scatter([], [], marker='+', linestyle="None", label="Calculated", color='black'),
        ax.scatter([], [], marker='x', linestyle="None", label="Simulated", color='black')
    ]
    ax.legend(handles=marker_handles, loc='upper right')
    plt.show()
    return

def spectralPlotSim(star_name=None):
    idxs, label = resolveStars(star_name)
    if idxs is None:
        return

    im = simulate(star_name)
    if im is None:
        return

    plt.figure(figsize=(12, 12))
    ax = plt.gca()
    ax.set_title(f'Simulated MICADO image')

    plt.imshow(im, norm='log', cmap='gray')
    #plt.colorbar()
    plt.show()
    return

def findPass(star_name, t_peri, r_peri, a, e, i, Omega, w, Tp, Per, passage_num=None):
    '''
    Helper function used by pericentrePass. Computes the 3D weekly displacement vector
    by evaluating the position half a week before and after, prints a formatted summary.

    :param star_name:
    :param t_peri:
    :param r_peri:
    :param a:
    :param e:
    :param i:
    :param Omega:
    :param w:
    :param Tp:
    :param Per:
    :param passage_num:
    :return: (t_peri, r_peri, weekly_dist)
    '''
    week = 1 / 52.1775  # 1 week in years
    x_before, y_before, z_before = orbitalPosition(t_peri - week / 2, a, e, i, Omega, w, Tp, Per)
    x_after, y_after, z_after = orbitalPosition(t_peri + week / 2, a, e, i, Omega, w, Tp, Per)
    dx = x_after - x_before
    dy = y_after - y_before
    dz = z_after - z_before
    weekly_dist = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    year = int(t_peri)
    day_of_year = (t_peri - year) * 365.25
    peri_date = datetime.date(year, 1, 1) + datetime.timedelta(days=day_of_year)

    label = f"Pericentre passage: {star_name}"
    if passage_num is not None:
        label += f"  (passage {passage_num})"
    print(f"\n--- {label} ---")
    print(f"  Time of closest approach  : {t_peri:.4f} yr  ({peri_date.strftime('%Y-%m-%d')})")
    print(f"  Closest distance to centre: {r_peri:.5f} arcsec")
    print(f"  Distance moved in +-1/2 week : {weekly_dist:.5f} arcsec")
    print(f"  (dX={dx:.5f}, dY={dy:.5f}, dZ={dz:.5f}) arcsec")

    return t_peri, r_peri, weekly_dist

def pericentrePass(star_name, t_start=None, t_end=None):
    '''
    Without t_start/t_end: finds the next pericentre passage to the Tp out of the data set.
    With t_start/t_end: scans every orbital period in interval.

    :param star_name:
    :param t_start:
    :param t_end:
    :return: results from findPass
    '''


    matches = np.where(names_val == star_name)[0]
    if len(matches) == 0:
        print(f"Star '{star_name}' not found. Available stars: {list(names_val)}")
        return

    idx = matches[0]
    a, e, i, Omega, w, Tp, Per = (
        a_val[idx], e_val[idx], i_val[idx],
        Omega_val[idx], w_val[idx], Tp_val[idx], Per_val[idx]
    )

    #find single pericentre passage
    if t_start is None and t_end is None:
        t_fine = np.linspace(Tp - Per / 2, Tp + Per / 2, 100000)
        x, y, z = orbitalPosition(t_fine, a, e, i, Omega, w, Tp, Per)
        r = np.sqrt(x**2 + y**2 + z**2)
        min_idx = np.argmin(r)
        return findPass(star_name, t_fine[min_idx], r[min_idx], a, e, i, Omega, w, Tp, Per)

    #find all passages in given interval
    if t_start >= t_end:
        print("Error: t_start must be less than t_end")
        return

    # Find the first period window where end exceeds t_start
    n_periods_before = np.floor((t_start - Tp) / Per)
    t_win = Tp + n_periods_before * Per

    results = []
    passage_num = 1

    while t_win < t_end:
        t_win_end = t_win + Per
        n_samples = max(10000, int(Per * 2000))
        t_fine = np.linspace(t_win, t_win_end, n_samples)
        x, y, z = orbitalPosition(t_fine, a, e, i, Omega, w, Tp, Per)
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        min_idx = np.argmin(r)
        t_peri = t_fine[min_idx]
        r_peri = r[min_idx]

        # Only report if the pericentre falls inside the requested interval
        if t_start <= t_peri <= t_end:
            result = findPass(star_name, t_peri, r_peri, a, e, i, Omega, w, Tp, Per, passage_num=passage_num)
            results.append(result)
            passage_num += 1

        t_win = t_win_end

    if not results:
        print(f"\nNo pericentre passages found for {star_name} "
              f"between {t_start:.2f} and {t_end:.2f}.")
    else:
        print(f"\n{len(results)} pericentre passage(s) found for {star_name} "
              f"between {t_start:.2f} and {t_end:.2f}.")

    return results

def bestObserving(star_name, t_start, t_end):
    '''
    Finds the first pericentre pass for given star in given interval. Print date and results.
    :param star_name:
    :param t_start:
    :param t_end:
    :return: (t_best, r_best, weekly_dist)
    '''
    # find star index by name
    matches = np.where(names_val == star_name)[0]
    if len(matches) == 0:
        print(f"Star '{star_name}' not found. Available stars: {list(names_val)}")
        return

    idx = matches[0]
    a, e, i, Omega, w, Tp, Per = (
        a_val[idx], e_val[idx], i_val[idx],
        Omega_val[idx], w_val[idx], Tp_val[idx], Per_val[idx]
    )

    week = 1 / 52.1775  # 1 week in years

    # high-resolution sampling over the requested window
    n_samples = max(100000, int((t_end - t_start) * 50000))
    t_fine = np.linspace(t_start, t_end, n_samples)

    x, y, z = orbitalPosition(t_fine, a, e, i, Omega, w, Tp, Per)
    r = np.sqrt(x**2 + y**2 + z**2)

    min_idx = np.argmin(r)
    t_best = t_fine[min_idx]
    r_best = r[min_idx]

    # angular velocity proxy: displacement over one week centred on best time
    t_lo = max(t_start, t_best - week / 2)
    t_hi = min(t_end,   t_best + week / 2)
    x0, y0, z0 = orbitalPosition(t_lo, a, e, i, Omega, w, Tp, Per)
    x1, y1, z1 = orbitalPosition(t_hi, a, e, i, Omega, w, Tp, Per)
    weekly_dist = np.sqrt((x1-x0)**2 + (y1-y0)**2 + (z1-z0)**2)

    # calendar date
    year = int(t_best)
    day_of_year = (t_best - year) * 365.25
    best_date = datetime.date(year, 1, 1) + datetime.timedelta(days=day_of_year)

    # also report distance profile: min / mean / max over window
    r_mean = np.mean(r)
    r_max  = np.max(r)

    print(f"\n--- Best observing window: {star_name}  [{t_start:.2f} – {t_end:.2f}] ---")
    print(f"  Best time               : {t_best:.4f} yr  ({best_date.strftime('%Y-%m-%d')})")
    print(f"  Distance at best time   : {r_best:.5f} arcsec")
    print(f"  Distance moved in 1 week: {weekly_dist:.5f} arcsec")
    print(f"  Distance range in window: min={r_best:.5f}  mean={r_mean:.5f}  max={r_max:.5f} arcsec")

    return t_best, r_best, weekly_dist

def orbitFit(star_name=None):
    '''
    Calculates the coefficients for the stars fit using SVD
    currently set to: # points = 5, t_0 = 2032, dt = 1

    :param star_name:
    :return:
    '''
    idxs, label = resolveStars(star_name)
    if idxs is None:
        return
    # orbitFit works on a single star; use first resolved index
    star_idx = idxs[0]
    print(f"Fitting orbit for: {names_val[star_idx]}")

    cmds = sim.UserCommands(use_instrument="MICADO", set_modes=["SCAO", "IMG_1.5mas"])
    point = 5
    points = np.zeros((point, 2))

    for dt in range(point):
        t_ = 2032 + dt
        table = orbitTable(t_, names_val, a_val, e_val, i_val, Omega_val, w_val, Tp_val, Per_val)

        # EXPTIME = 3600 = ndit * dit
        cmds["!DET.dit"] = 30
        cmds["!DET.ndit"] = 120
        micado = sim.OpticalTrain(cmds)
        fixed_stars = sim_tp.stellar.stars(filter_name="H",
                                           amplitudes=np.atleast_1d(float(table['mag'][star_idx])),
                                           spec_types=np.full(1, 'A0V'),
                                           x=np.atleast_1d(float(table['x'][star_idx])),
                                           y=np.atleast_1d(float(table['y'][star_idx])))
        micado.observe(fixed_stars)
        hdus = micado.readout()
        im = hdus[0][1].data  # numpy array

        if not im.dtype.isnative:
            im = im.byteswap(inplace=True)
            im = im.view(im.dtype.newbyteorder('='))

        threshold = 1.0
        while threshold <= SEP_THRESHOLD_MAX:
            bkg = sep.Background(im)
            sources = sep.extract(im - bkg, threshold, err=bkg.globalrms)
            threshold += 1
            if len(sources) == 1:
                break
        else:
            print(f"Warning: could not isolate exactly 1 source at t={t_}; got {len(sources)}.")

        x_ref_pix = im.shape[1] / 2
        y_ref_pix = im.shape[0] / 2
        points[dt][0] = (sources['x'][0] - x_ref_pix) * PIXEL_SCALE
        points[dt][1] = (sources['y'][0] - y_ref_pix) * PIXEL_SCALE

    x = points[:, 0]
    y = points[:, 1]

    design_mat = np.stack([x ** 2, x * y, y ** 2, x, y, np.ones_like(x)], axis=1)

    _, _, Vh = np.linalg.svd(design_mat)
    coefficients = Vh[-1, :]

    residuals = design_mat @ coefficients
    rmse = np.sqrt(np.mean(residuals ** 2))

    A, B, C, D_coeff, E, F = coefficients

    result = {
        "coeffs": {"A": A, "B": B, "C": C, "D": D_coeff, "E": E, "F": F},
        "rmse": rmse
    }
    print(result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orbital analysis tools for S-stars")
    parser.add_argument("--data", default="J_ApJ_837_30_table3.dat.fits",
                        help="Path to FITS star catalogue file (default: J_ApJ_837_30_table3.dat.fits)")
    subparsers = parser.add_subparsers(dest="command")

    star_help     = "One or more star names (default: all). E.g. --star S2 S4 S14"
    star_pos_help = "One or more star names, e.g. S2 S4"

    # pericentre — one or more stars required; optional time window
    p_peri = subparsers.add_parser("pericentre",
        help="Find pericentre passage(s). Without --t_start/--t_end finds the single passage "
             "near the catalogued Tp; with both flags finds all passages within [t_start, t_end].")
    p_peri.add_argument("--star", type=str, nargs='+', help=star_pos_help)
    p_peri.add_argument("--t_start", type=float, default=None, help="Start of time interval [yr], e.g. 2000")
    p_peri.add_argument("--t_end",   type=float, default=None, help="End of time interval [yr],   e.g. 2050")

    # best observing window — one or more stars required
    p_obs = subparsers.add_parser("bestobs", help="Find best time to observe star(s) within a given time frame")
    p_obs.add_argument("--star",    type=str,   nargs='+', help=star_pos_help)
    p_obs.add_argument("--t_start", type=float, help="Start of time frame [yr], e.g. 2020")
    p_obs.add_argument("--t_end",   type=float, help="End of time frame [yr],   e.g. 2030")

    # plot / analysis commands — star optional, accepts multiple
    p_pos = subparsers.add_parser("position", help="Plot calculated orbits")
    p_pos.add_argument("--star",    type=str,   nargs='+', default=None, help=star_help)
    p_pos.add_argument("--t_start", type=float, default=None, help="Start of time frame [yr], e.g. 1992")
    p_pos.add_argument("--t_end",   type=float, default=None, help="End of time frame [yr],   e.g. 2025")

    p_vel = subparsers.add_parser("velocity", help="Plot orbital velocities")
    p_vel.add_argument("--star",    type=str,   nargs='+', default=None, help=star_help)
    p_vel.add_argument("--t_start", type=float, default=None, help="Start of time frame [yr], e.g. 1992")
    p_vel.add_argument("--t_end",   type=float, default=None, help="End of time frame [yr],   e.g. 2025")

    p_cmp = subparsers.add_parser("compare", help="Compare calculated vs simulated positions")
    p_cmp.add_argument("--star", type=str, nargs='+', default=None, help=star_help)

    p_sim = subparsers.add_parser("sim", help="Show simulated MICADO image")
    p_sim.add_argument("--star", type=str, nargs='+', default=None, help=star_help)

    p_fit = subparsers.add_parser("orbitfit", help="Fit ellipse to simulated orbit points")
    p_fit.add_argument("--star", type=str, nargs='+', default=None, help=star_help)

    args = parser.parse_args()

    load_data(args.data)

    if args.command == "pericentre":
        if (args.t_start is None) != (args.t_end is None):
            print("Error: provide both --t_start and --t_end, or neither.")
        elif args.t_start is not None and args.t_start >= args.t_end:
            print("Error: t_start must be less than t_end")
        else:
            for s in args.star:
                pericentrePass(s, args.t_start, args.t_end)
    elif args.command == "bestobs":
        if args.t_start >= args.t_end:
            print("Error: t_start must be less than t_end")
        else:
            for s in args.star:
                bestObserving(s, args.t_start, args.t_end)
    elif args.command == "position":
        positionPolt(args.star, args.t_start, args.t_end)
    elif args.command == "velocity":
        velocityPlot(args.star, args.t_start, args.t_end)
    elif args.command == "compare":
        comparePlot(args.star)
    elif args.command == "sim":
        spectralPlotSim(args.star)
    elif args.command == "orbitfit":
        orbitFit(args.star)
    else:
        parser.print_help()