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

t_obs = 2022
t = np.linspace(2000, 2020, 200)

class MyClass:
    def __init__(self, filename, data, t):

        self.data = data
        self.names_val = full_names_val[eligible_stars]
        self.spectral_val = full_spectral_val[eligible_stars]
        self.kmag_val = full_kmag_val[eligible_stars]
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

    def orbitTable(self):
        x_pos = []
        y_pos = []

        for val in range(len(self.names_val)):
            x, y, z = self.orbitalPosition(t_obs, self.a_val[val], self.e_val[val], self.i_val[val], self.Omega_val[val], self.w_val[val], self.Tp_val[val],
                                           self.Per_val[val])
            plt.plot(x, y)

            x_pos.append(x)
            y_pos.append(y)

        table = Table(
            names=["x", "y", "ref", "weight", "type"],
            data=[x_pos, y_pos, self.names_val, self.kmag_val, self.spectral_val],
            units=[u.arcsec, u.arcsec, None, None, None]
        )
        table['x'].unit = 'arcsec'
        table['y'].unit = 'arcsec'
        return table

    def simulate(self):
        '''
        - Simulates MICADO observation
        - converts pixel data to RA and DEC

        - returns (x,y)
        '''
        source = sim.source.source.Source(table=orbit_table)

        # sim.download_packages(["Armazones", "ELT", "MICADO"])

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
        cmd["!DET.width"] = 256  # pixel
        cmd["!DET.height"] = 256

        cmd["!SIM.sub_pixel.flag"] = True

        micado = sim.OpticalTrain(cmd)
        micado.observe(source)
        hdus = micado.readout()
        sim_image = hdus[0][1].data  # numpy array

        # get position in arcsec fom pixels
        pixel_scale = 0.0015  #
        measrued_pos = []

        for x_in, y_in in zip(orbit_table['x'], orbit_table['y']):
            # Convert arcsec → pixels
            x_pix = int(sim_image.shape[1] / 2 + x_in / pixel_scale)
            y_pix = int(sim_image.shape[0] / 2 + y_in / pixel_scale)

            # Convert to arcsec relative to center
            x_arc = (x_pix - sim_image.shape[1] / 2) * pixel_scale
            y_arc = (y_pix - sim_image.shape[0] / 2) * pixel_scale

            measrued_pos.append((x_arc, y_arc))

        return measrued_pos

    def plotter(self):
        for val in range(len(self.names_val)):
            x, y, z = self.orbitalPosition(self.t, self.a_val[val], self.e_val[val], self.i_val[val],
                                           self.Omega_val[val], self.w_val[val], self.Tp_val[val],
                                           self.Per_val[val])
            plt.plot(x, y)
            # plt.plot(x, y, label=f'S{val+1}') # for legend containing every name

        plt.scatter(0, 0, color='black', marker='+', label='Sgr A*')

        plt.title('Calculated orbits')
        plt.xlabel('R.A. ["]')
        plt.ylabel('Dec. ["]')
        plt.tight_layout()
        plt.gca().invert_xaxis()
        plt.grid(True)
        plt.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=len(self.names_val), fontsize=7)
        return plt.show()

    def comparePlot(self):
        measured_pos = self.simulate()
        for i, (mx, my) in enumerate(measured_pos):
            print(f"{orbit_table['ref'][i]}:")
            print(f"  true:     ({orbit_table['x'][i]: .5f}, {orbit_table['y'][i]: .5f}) arcsec")
            print(f"  measured: ({mx: .5f}, {my: .5f}) arcsec")
            print(f"  error:    ({mx - orbit_table['x'][i]: .5f}, {my - orbit_table['y'][i]: .5f}) arcsec")
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
        plt.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=len(self.names_val), fontsize=7)

        return plt.show()




x = MyClass(filename, data, t)
x.plotter()
orbit_table = x.orbitTable()
print(orbit_table)
x.comparePlot()