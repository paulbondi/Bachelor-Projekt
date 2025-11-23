import datetime
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy import units as u
import scopesim as sim
from scopesim.source import source


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
'''
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
'''

def orbitTable(t_obs, names_arr, a_arr, e_arr, i_arr, Omega_arr, w_arr, Tp_arr, Per_arr, ):
    x_pos = []
    y_pos = []

    for id in range(len(names_arr)):
        x, y, z = orbitalPosition(t_obs, a_arr[id], e_arr[id], i_arr[id], Omega_arr[id], w_arr[id], Tp_arr[id], Per_arr[id])

        x_pos.append(x)
        y_pos.append(y)

    table = Table(
        names=["x", "y", "ref", "weight", "type"],
        data=[x_pos, y_pos, names_arr, kmag_val, spectral_val],
        units=[u.arcsec, u.arcsec, None, None, None]
    )
    table['x'].unit = 'arcsec'
    table['y'].unit = 'arcsec'
    return table
orbit_table = orbitTable(t_obs, names_val, a_val, e_val, i_val, Omega_val, w_val, Tp_val, Per_val)
print(orbit_table)

# position plot
for val in range(len(names_val)):
    x, y, z = orbitalPosition(t, a_val[val], e_val[val], i_val[val], Omega_val[val], w_val[val], Tp_val[val], Per_val[val])
    plt.plot(x, y)
    #plt.plot(x, y, marker='+', label=f'S{val+1}') # for legend containing every name and scatter

#plt.scatter(x, y, label='S 1-39', s=5, color='steelblue')
#plt.plot(x, y, label='S 1-39', color='steelblue')
plt.scatter(0, 0, color='black', marker='+', label='Sgr A*')

plt.title('Calculated orbits')
plt.xlabel('R.A. ["]')
plt.ylabel('Dec. ["]')
plt.tight_layout()
plt.gca().invert_xaxis()
plt.grid(True)
plt.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=len(names_val), fontsize=7)
plt.show()


'''
-- create scopesim simulation
-- compare with calculation results
'''

source = sim.source.source.Source(table=orbit_table)

#sim.download_packages(["Armazones", "ELT", "MICADO"])

# from irdb/MICADO/docs/example_notebooks/2_scopesim_SCAO_1.5mas_astrometry.ipynb
observation_dict = {
    "!OBS.filter_name_pupil": "open",
    "!OBS.filter_name_fw1": "J",
    "!OBS.filter_name_fw2": "open",
    "!INST.filter_name": "Pa-beta",
    "!OBS.ndit": 1,
    "!OBS.dit": 3600,

    "!OBS.instrument": "MICADO",
    "!OBS.catg": "SCIENCE",
    "!OBS.tech": "IMAGE",
    "!OBS.type": "OBJECT",
    "!OBS.mjdobs": datetime.datetime(2022, 1, 1, 2, 30, 0)
}

cmd = sim.UserCommands(
    use_instrument="MICADO",
    set_modes=["SCAO", "IMG_1.5mas"],
    properties=observation_dict,
)
cmd["!DET.width"] = 256     # pixel
cmd["!DET.height"] = 256

cmd["!SIM.sub_pixel.flag"] = True

micado = sim.OpticalTrain(cmd)
micado.observe(source)
hdus = micado.readout()
sim_image = hdus[0][1].data #numpy array

# get position in arcsec fom pixels
pixel_scale = 0.0015 #
measrued_pos = []

for x_in, y_in in zip(orbit_table['x'], orbit_table['y']):
    # Convert arcsec → pixels
    x_pix = int(sim_image.shape[1] / 2 + x_in / pixel_scale)
    y_pix = int(sim_image.shape[0] / 2 + y_in / pixel_scale)

    # Convert to arcsec relative to center
    x_arc = (x_pix - sim_image.shape[1] / 2) * pixel_scale
    y_arc = (y_pix - sim_image.shape[0] / 2) * pixel_scale

    measrued_pos.append((x_arc, y_arc))

# print measured vs calculated
for i, (mx, my) in enumerate(measrued_pos):
    print(f"{orbit_table['ref'][i]}:")
    print(f"  true:     ({orbit_table['x'][i]: .5f}, {orbit_table['y'][i]: .5f}) arcsec")
    print(f"  measured: ({mx: .5f}, {my: .5f}) arcsec")
    print(f"  error:    ({np.abs(mx - orbit_table['x'][i]): .5f}, {np.abs(my - orbit_table['y'][i]): .5f}) arcsec")
    print()

    plt.scatter(orbit_table['x'][i], orbit_table['y'][i], marker='x')
    plt.scatter(mx, my, marker='+')

plt.scatter(0, 0, color='black', marker='+', label='Sgr A*')

plt.title('Calculated (x) vs Measured (+)')
plt.xlabel('R.A. ["]')
plt.ylabel('Dec. ["]')
plt.tight_layout()
plt.gca().invert_xaxis()
plt.grid(True)
plt.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=len(names_val), fontsize=7)
plt.show()


# spectral plot
color_map = {'e': 'blue', 'l': 'red'}
star_colors = [color_map[t] for t in orbit_table["type"]]
min_mag = np.min(orbit_table["weight"])
max_mag = np.max(orbit_table["weight"])
size_scale_factor = 1200
star_sizes = size_scale_factor * (max_mag - orbit_table["weight"] + 1) / (max_mag - min_mag + 1)

plt.figure(figsize=(9, 9), facecolor='black')
ax = plt.gca()
ax.set_facecolor('black')

plt.scatter(orbit_table['x'], orbit_table['y'], s=star_sizes * 0.2, c='white', alpha=1.0, zorder=10)
plt.show()

'''
# velocity plot
for val in range(len(names_val)):
    v = orbitalVelocity(t, a_val[val], e_val[val], i_val[val], Omega_val[val], w_val[val], Tp_val[val], Per_val[val])
    plt.plot(t, v)

plt.xlabel('t [yr]')
plt.ylabel(r'$V_{LSR}$ [km/s]')
plt.tight_layout()
plt.grid(True)
plt.show()
'''

