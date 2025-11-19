import numpy as np
import matplotlib.pyplot as plt
import scopesim as ss
from astropy.table import Table
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
t = np.linspace(1992, 2016, 200)
t_obs = float(input("time: "))
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
    M = 2 * np.pi / Per * (t - Tp) # mean anomaly
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

def scopesimTable(t_obs, names_arr, a_arr, e_arr, i_arr, Omega_arr, w_arr, Tp_arr, Per_arr):
    x_pos = []
    y_pos = []

    for id in range(len(names_arr)):
        x, y, z = orbitalPosition(t_obs, a_arr[id], e_arr[id], i_arr[id], Omega_arr[id], w_arr[id], Tp_arr[id], Per_arr[id])

        x_pos.append(x)
        y_pos.append(y)

    table = Table({
        'name': names_arr,
        'x': x_pos,
        'y': y_pos,
        'magnitude': kmag_val,
        'spec_type': spectral_val
    })
    table['x'].unit = 'arcsec'
    table['y'].unit = 'arcsec'
    return table
scope_table = scopesimTable(t_obs, names_val, a_val, e_val, i_val, Omega_val, w_val, Tp_val, Per_val)

'''
TO-DO: 
-- create scopesim simulation
-- compare with calculation results
'''



# position plot
for val in range(len(names_val)):
    x, y, z = orbitalPosition(t, a_val[val], e_val[val], i_val[val], Omega_val[val], w_val[val], Tp_val[val], Per_val[val])
    plt.plot(x, y)
    plt.scatter(scope_table['x'][val], scope_table['y'][val], marker='o')
    #plt.plot(x, y, marker='+', label=f'S{val+1}') # for legend containing every name and scatter

#plt.scatter(x, y, label='S 1-39', s=5, color='steelblue')
#plt.plot(x, y, label='S 1-39', color='steelblue')
plt.scatter(0, 0, color='black', marker='+', label='Sgr A*')

plt.xlabel('R.A. ["]')
plt.ylabel('Dec. ["]')
plt.tight_layout()
plt.gca().invert_xaxis()
plt.grid(True)
plt.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=len(names_val), fontsize=7)
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

