import numpy as np
import matplotlib.pyplot as plt

filename = "../../Documents/Studium/Astro/BAccArbeit/table3.dat.txt"
full_data = np.loadtxt(filename, usecols=range(1,15))
eligible_stars = [0, 1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 15, 17, 19, 22, 24]
full_names_val = np.loadtxt(filename, usecols=0, dtype=str)
data = full_data[eligible_stars, :]
t0 = 2017
t = np.linspace(2000, 2100, 200)
dt = t - t0

class MyClass:
    def __init__(self, filename, data, t):

        self.data = data
        self.names_val = full_names_val[eligible_stars]
        self.a_val = self.data[:, 0]  # semi-major axis a [arcsec]
        self.a_unc = self.data[:, 1]
        self.e_val = self.data[:, 2]  # eccentricity
        self.e_unc = self.data[:, 3]
        self.i_val = self.data[:, 4]  # inclination [deg]
        self.i_unc = self.data[:, 5]
        self.Omega_val = self.data[:, 6]  # angle of ascending node W [deg]
        self.Omega_unc = self.data[:, 7]
        self.w_val = self.data[:, 8]  # longitude of pericenter w [deg]
        self.w_unc = self.data[:, 9]
        self.Tp_val = self.data[:, 10]  # epoch of pericentre passage [yr]
        self.Tp_unc = self.data[:, 11]
        self.Per_val = self.data[:, 12]  # period [yr]
        self.Per_unc = self.data[:, 13]
        self.t = t

    # Newton-Raphson Method für kepler glg: M = E - e*sin(E)
    def kepler(self, M, e, epsilon=1e-9, max_it=100):
        M = np.mod(M, 2 * np.pi)  # 0 <= M <= 2pi
        E = np.copy(M)

        for i in range(max_it):
            delta = (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
            E -= delta
            if np.all(np.abs(delta) < epsilon):
                break

        return E

    def orbitalPosition(self, t, a, e, i, Omega, w, Tp, Per):
        i, Omega, w = np.radians([i, Omega, w])
        M = 2 * np.pi / Per * (t - Tp)  # mean anomaly
        E = self.kepler(M, e)  # eccentric anomaly
        f = 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2))  # true anomaly
        r = a * (1 - e * np.cos(E))
        x_ = r * (np.cos(Omega) * np.cos(w + f) - np.sin(Omega) * np.sin(w + f) * np.cos(i))
        y_ = r * (np.sin(Omega) * np.cos(w + f) + np.cos(Omega) * np.sin(w + f) * np.cos(i))
        z_ = r * (np.sin(w + f) * np.sin(i))

        return x_, y_, z_

    def plotter(self):
        for val in range(len(self.names_val)):
            x, y, z = self.orbitalPosition(self.t, self.a_val[val], self.e_val[val], self.i_val[val], self.Omega_val[val], self.w_val[val], self.Tp_val[val],
                                      self.Per_val[val])
            plt.plot(x, y)
            #plt.plot(x, y, label=f'S{val+1}') # for legend containing every name

        plt.plot(x, y, label='S 1-39', color='steelblue')
        plt.scatter(0, 0, color='black', marker='+', label='Sgr A*')
        plt.xlabel('R.A. ["]')
        plt.ylabel('Dec. ["]')
        plt.tight_layout()
        plt.gca().invert_xaxis()
        plt.grid(True)
        plt.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=len(self.names_val), fontsize=7)
        return plt.show()


x = MyClass(filename, data, t)
x.plotter()