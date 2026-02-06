import datetime

import matplotlib.pyplot as plt
import numpy as np
import scopesim as sim
import scopesim_templates as sim_tp
from astropy import units as u
from astropy.table import Table
from scopesim.source import source
import sep
from scipy.interpolate import splprep, splev

filename = "table3.dat.txt"
full_data = np.loadtxt(filename, usecols=range(1,15))
eligible_stars = [0, 1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 15, 17, 19, 22, 24]
full_names_val = np.loadtxt(filename, usecols=0, dtype=str)
full_spectral_val = np.loadtxt(filename, usecols=15, dtype=str)
full_kmag_val = np.loadtxt(filename, usecols=16)
data = full_data[eligible_stars, :]

names_val = full_names_val[eligible_stars]
spectral_val = full_spectral_val[eligible_stars]
kmag_val = full_kmag_val[eligible_stars]
a_val = data[:,0] # semi-major axis a [arcsec]
a_unc = data[:,1]
e_val = data[:,2] # eccentricity
e_unc = data[:,3]
i_val = data[:,4] # inclination [deg]
i_unc = data[:,5]
Omega_val = data[:,6] # angle of ascending node W [deg]
Omega_unc = data[:,7]
w_val = data[:,8] # longitude of pericenter w [deg]
w_unc = data[:,9]
Tp_val = data[:,10] # epoch of pericentre passage [yr]
Tp_unc = data[:,11]
Per_val = data[:,12] # period [yr]
Per_unc = data[:,13]

t0 = 2017
t = np.linspace(2000, 2015, 200)
t_obs = 2022 # float(input("time: "))
dt = t - t0

# Newton-Raphson Method für kepler glg: M = E - e*sin(E)
def kepler(M, e, epsilon=1e-9, max_it=100):
    M = np.mod(M, 2 * np.pi) # 0 <= M <= 2pi
    E = np.copy(M)

    for i in range(max_it):
        delta = (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
        E -= delta
        if np.all(np.abs(delta) < epsilon):
            break

    return E

def orbitalPosition(t, a, e, i, Omega, w, Tp, Per):
    i, Omega, w = np.radians([i, Omega, w])
    M = (2 * np.pi / Per) * (t - Tp) # mean anomaly
    E = kepler(M, e) # eccentric anomaly
    f = 2 * np.arctan(np.sqrt((1 + e) / (1-e)) * np.tan(E / 2)) # true anomaly
    r = a * (1 - e * np.cos(E))
    x_ = r * (np.cos(Omega) * np.cos(w + f) - np.sin(Omega) * np.sin(w + f) * np.cos(i))
    y_ = r * (np.sin(Omega) * np.cos(w + f) + np.cos(Omega) * np.sin(w + f) * np.cos(i))
    z_ = r * (np.sin(w + f) * np.sin(i))

    return x_, y_, z_

def orbitalVelocity(t, a, e, i, Omega, w, Tp, Per):
    n = 2 * np.pi / Per # mean motion
    M = 2 * np.pi / Per * (t - Tp)  # mean anomaly
    E = kepler(M, e)  # eccentric anomaly
    f = 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2))  # true anomaly

    r = a * (1 - e * np.cos(E))
    vx = - n * a**2 * np.sin(E) / r
    vy = n * a**2 * np.sqrt(1-e**2) * np.cos(E) / r
    v = np.sqrt(vx**2 + vy**2)
    return v

def orbitTable(t_obs, names_arr, a_arr, e_arr, i_arr, Omega_arr, w_arr, Tp_arr, Per_arr):
    x_pos = []
    y_pos = []

    for id in range(len(names_arr)):
        x, y, z = orbitalPosition(t_obs, a_arr[id], e_arr[id], i_arr[id], Omega_arr[id], w_arr[id], Tp_arr[id], Per_arr[id])

        x_pos.append(x)
        y_pos.append(y)

    table = Table(
        names=["x", "y", "ref", "mag", "type"],
        data=[x_pos, y_pos, names_arr, kmag_val, spectral_val],
        units=[u.arcsec, u.arcsec, None, None, None]
    )
    table['x'].unit = 'arcsec'
    table['y'].unit = 'arcsec'

    print(table)
    return table
orbit_table = orbitTable(t_obs, names_val, a_val, e_val, i_val, Omega_val, w_val, Tp_val, Per_val)

def simulate():
    '''
    -- create scopesim simulation
    -- compare with calculation results
    '''

    #sim.download_packages(["Armazones", "ELT", "MICADO"])

    cmds = sim.UserCommands(use_instrument="MICADO", set_modes=["SCAO", "IMG_1.5mas"])

    # EXPTIME = 3600 = ndit * dit
    cmds["!DET.dit"] = 30
    cmds["!DET.ndit"] = 120
    micado = sim.OpticalTrain(cmds)
    fixed_stars = sim_tp.stellar.stars(filter_name="H",
                                       amplitudes=orbit_table['mag'] * u.mag,  # [u.mag, u.ABmag, u.Jy]
                                       spec_types=np.full(len(eligible_stars), 'A0V'),
                                       x=orbit_table['x'], y=orbit_table['y'])  # [u.arcsec]
    micado.observe(fixed_stars)
    hdus = micado.readout()
    sim_image = hdus[0][1].data #numpy array
    return sim_image

def findStars():
    # get position in arcsec fom pixels
    pixel_scale = 0.0015  #
    measured_pos = []
    im = simulate()

    if not data.dtype.isnative:
        im = im.byteswap(inplace=True)
        im = im.view(data.dtype.newbyteorder('='))

    i = 1.0
    while (True):
        # Extract
        bkg = sep.Background(im)
        sources = sep.extract(im - bkg, i, err=bkg.globalrms)
        i += 1
        if (len(sources) == len(eligible_stars)):
            break

    # Get positions
    print(f"Found {len(sources)} stars with a SEP threshold: {i}")
    x, y = sources['x'], sources['y']

    # Step 1: Convert measured positions
    measured_x = []
    measured_y = []

    x_ref_pix = im.shape[1] / 2  # image center X
    y_ref_pix = im.shape[0] / 2  # image center Y

    for x, y in zip(sources['x'], sources['y']):
        x_arc = (x - x_ref_pix) * pixel_scale
        y_arc = (y - y_ref_pix) * pixel_scale
        measured_x.append(x_arc)
        measured_y.append(y_arc)

    measured_x = np.array(measured_x)
    measured_y = np.array(measured_y)

    # Compare with calculated positions
    calc_x = np.array(orbit_table['x'])
    calc_y = np.array(orbit_table['y'])

    dx_list = []
    dy_list = []
    sep_list = []

    # For each measured star, find nearest calculated
    for mx, my in zip(measured_x, measured_y):
        distances = np.sqrt((calc_x - mx) ** 2 + (calc_y - my) ** 2)
        nearest_idx = np.argmin(distances)

        dx_list.append(mx - calc_x[nearest_idx])
        dy_list.append(my - calc_y[nearest_idx])
        sep_list.append(distances[nearest_idx])  # ADDED!

    dx = np.array(dx_list)
    dy = np.array(dy_list)
    sep_ = np.array(sep_list)

    # Print results
    for i, (mx, my) in enumerate(zip(measured_x, measured_y)):
        print(f"{orbit_table['ref'][i]}:")
        print(f"  true:     ({orbit_table['x'][i]: .5f}, {orbit_table['y'][i]: .5f}) arcsec")
        print(f"  measured: ({mx: .5f}, {my: .5f}) arcsec")
        print(f"  error:    ({np.abs(mx - orbit_table['x'][i]): .5f}, {np.abs(my - orbit_table['y'][i]): .5f}) arcsec")
        print()

    print(f"Matched {len(dx)} stars")
    print(f"Mean offset: ΔX = {np.mean(dx):.4f} ± {np.std(dx):.4f} arcsec")
    print(f"             ΔY = {np.mean(dy):.4f} ± {np.std(dy):.4f} arcsec")
    print(f"RMS error: {np.sqrt(np.mean(sep_ ** 2)):.4f} arcsec")
    print(f"Median separation: {np.median(sep_):.4f} arcsec")

    return measured_x, measured_y

def positionPolt():
    # position plot
    fig, ax = plt.subplots()

    line_objects = []

    for val in range(len(names_val)):
        x, y, z = orbitalPosition(t, a_val[val], e_val[val], i_val[val], Omega_val[val], w_val[val], Tp_val[val], Per_val[val])
        line, = ax.plot(x,y, label=names_val[val], linewidth=1.5)
        line_objects.append(line)

    ax.scatter(0, 0, color='black', marker='+')

    ax.set_title('Calculated orbits')
    ax.set_xlabel('R.A. ["]')
    ax.set_ylabel('Dec. ["]')

    ax.invert_xaxis()
    ax.grid(True)

    leg = ax.legend(
        loc = 'upper center',
        bbox_to_anchor=(0.0, -0.05, 1.0, 0.1),
        ncol = len(names_val),
        mode = 'expand',
        frameon = False,
        fontsize = 8,
        handlelength = 0,
        handletextpad = 0,
        borderaxespad=0
    )

    for line, text in zip(line_objects, leg.get_texts()):
        curr_color = line.get_color()
        text.set_color(curr_color)



    return plt.show()

def velocityPlot():
    for val in range(len(names_val)):
        v = orbitalVelocity(t, a_val[val], e_val[val], i_val[val], Omega_val[val], w_val[val], Tp_val[val],
                            Per_val[val])
        plt.plot(t, v)

    plt.xlabel('t [yr]')
    plt.ylabel(r'$V_{LSR}$ [km/s]')
    plt.tight_layout()
    plt.grid(True)
    plt.show()
    return

def comparePlot():
    measured_x, measured_y = findStars()

    # Print results
    fig, ax = plt.subplots()
    line1_objects = []

    for i, (mx, my) in enumerate(zip(measured_x, measured_y)):
        ax1 = ax.scatter(orbit_table['x'][i], orbit_table['y'][i], marker='x', label=orbit_table['ref'][i])
        color = ax1.get_facecolor()
        ax2 = ax.scatter(mx, my, marker='+', color=color)
        line1_objects.append(ax1)

    ax.scatter(0, 0, color='black', marker='+')
    ax.set_title('Calculated (x) vs Measured (+)')
    ax.set_xlabel('R.A. ["]')
    ax.set_ylabel('Dec. ["]')
    ax.invert_xaxis()
    ax.grid(True)

    legend1 = ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.0, -0.05, 1.0, 0.1),
        ncol=len(names_val),
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
        ax.scatter([], [], marker='x', linestyle="None", label="Calculated", color='black'),
        ax.scatter([], [], marker='+', linestyle="None", label="Measured", color='black')
    ]
    ax.legend(handles=marker_handles,
              loc='upper right',
              )
    plt.show()
    return

def spectralPlotCalc():

    # spectral plot
    color_map = {'e': 'blue', 'l': 'red'}
    star_colors = [color_map[t] for t in orbit_table["type"]]
    min_mag = np.min(orbit_table["mag"])
    max_mag = np.max(orbit_table["mag"])
    size_scale_factor = 1200
    star_sizes = size_scale_factor * (max_mag - orbit_table["mag"] + 1) / (max_mag - min_mag + 1)

    plt.figure(figsize=(9, 9), facecolor='black')
    ax = plt.gca()
    ax.set_facecolor('black')

    plt.scatter(orbit_table['x'], orbit_table['y'], s=star_sizes, c='white')
    plt.show()

    return

def spectralPlotSim():
    im = simulate()

    plt.figure(figsize=(12, 12))
    plt.imshow(im, norm='log', cmap='gray')
    plt.colorbar()
    plt.show()
    return

def orbitFit():
    cmds = sim.UserCommands(use_instrument="MICADO", set_modes=["SCAO", "IMG_1.5mas"])
    point = 5
    t_ = 1990
    dt = 0
    measured_x = []
    measured_y = []
    points = np.zeros((point, 2))

    while dt < point:
        t_ += dt
        table = orbitTable(t_, names_val, a_val, e_val, i_val, Omega_val, w_val, Tp_val, Per_val)

        # EXPTIME = 3600 = ndit * dit
        cmds["!DET.dit"] = 30
        cmds["!DET.ndit"] = 120
        micado = sim.OpticalTrain(cmds)
        fixed_stars = sim_tp.stellar.stars(filter_name="H",
                                           amplitudes=table['mag'],  # [u.mag, u.ABmag, u.Jy]
                                           spec_types=np.full(len(eligible_stars), 'A0V'),
                                           x=table['x'], y=table['y'])  # [u.arcsec]
        micado.observe(fixed_stars)
        hdus = micado.readout()
        im = hdus[0][1].data  # numpy array

        pixel_scale = 0.0015  #

        if not data.dtype.isnative:
            im = im.byteswap(inplace=True)
            im = im.view(data.dtype.newbyteorder('='))

        i = 1.0
        while (True):
            # Extract
            bkg = sep.Background(im)
            sources = sep.extract(im - bkg, i, err=bkg.globalrms)
            i += 1
            if len(sources) == 1:
                break

        # Get positions

        x, y = sources['x'][0], sources['y'][0]

        # Step 1: Convert measured positions


        x_ref_pix = im.shape[1] / 2  # image center X
        y_ref_pix = im.shape[0] / 2  # image center Y

        for x, y in zip(sources['x'], sources['y']):
            x_arc = (x - x_ref_pix) * pixel_scale
            y_arc = (y - y_ref_pix) * pixel_scale
            points[dt][0] = x_arc
            points[dt][1] = y_arc

        dt += 1

    print(points)
    measured_x = np.array(measured_x)
    measured_y = np.array(measured_y)


    # Transpose for splprep
    x, y = points.T

    # Parametric spline
    tck, u = splprep([x, y], s=0, k=3)

    # Interpolated orbit
    u_fine = np.linspace(0, 1, 500)
    x_i, y_i = splev(u_fine, tck)

comparePlot()

