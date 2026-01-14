from manim import *
import numpy as np
import os

os.environ["PATH"] += os.pathsep + r'C:\Users\aupau\ffmpeg\bin'

filename = "table3.dat.txt"
full_data = np.loadtxt(filename, usecols=range(1,15))
eligible_stars = [0, 1, 2, 4, 5, 6, 8, 9, 10]
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
t = np.linspace(1992, 2022, 200)
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


from manim import *
import numpy as np


class SStarOrbits(ThreeDScene):
    def construct(self):
        # 1. Scale and Axes
        # Since your data is in the range of ~0.5, we scale by 10
        # so that 0.5 arcseconds = 5 Manim units.
        SCALE_FACTOR = 10

        axes = ThreeDAxes(
            axis_config={
                "stroke_width": 1,  # Makes the lines thinner
                "include_tip": False,  # Removes the arrows at the ends
                "include_ticks": True,  # Optional: keeps the measurement ticks
            }
        )
        self.add(axes)

        '''
        Bounding Box
        
        # Create a bounding box
        # 'side_length' should match the reach of your orbits (e.g., 0.5 * SCALE_FACTOR * 2)
        bounding_box = Cube(side_length=10, fill_opacity=0, stroke_width=1, stroke_color=GRAY)

        # Position it at the origin (Sagittarius A*)
        bounding_box.move_to(ORIGIN)
        self.add(bounding_box)

        # Create a grid floor
        grid = NumberPlane(
            x_range=[-5, 5, 1],
            y_range=[-5, 5, 1],
            background_line_style={"stroke_color": GRAY, "stroke_width": 1, "stroke_opacity": 0.3}
        )
        # Move it to the bottom of your bounding box (z = -5)
        grid.move_to(IN * 5)
        self.add(grid)
        '''

        '''
        Camer Angle
        '''
        # 2. Initial Camera Angle
        self.set_camera_orientation(phi=75 * DEGREES, theta=-45 * DEGREES)

        self.camera.background_color = "#000000"  # Deep black
        center_glow = Dot3D(radius=0.05, color=WHITE).set_glow_factor(2)
        self.add(center_glow)

        '''
        Timer
        
        year_start = 1992.0
        time_speed = 5
        # Create the decimal number (2D)
        year_num = DecimalNumber(year_start, num_decimal_places=1, color=YELLOW)
        year_label = Text("Year: ", font_size=36)
        timer_group = VGroup(year_label, year_num).arrange(RIGHT)

        # This is the magic line: it stays flat on the screen (HUD)
        self.add_fixed_orientation_mobjects(timer_group)

        # Position it in the top-left corner of the screen
        timer_group.to_corner(UL, buff=0.5)

        # Simple updater
        year_num.add_updater(lambda d: d.set_value(year_start + self.renderer.time * time_speed))
        '''


        ''''''
        # 3. Create the stars based on your list
        s_star_colors = [
            RED, BLUE, GREEN, YELLOW, ORANGE,
            PURPLE, PINK, TEAL, GOLD, MAROON,
            LIGHT_PINK, MAROON_A, RED_E, WHITE
        ]

        for idx in range(len(names_val)):
            # Pick a color (Manim cycles through these if you use a list)
            star_color = s_star_colors[idx % len(s_star_colors)]

            star = Dot3D(radius=0.03, color=star_color)

            # trail ensures the lines look like your matplotlib plot
            trail = TracedPath(
                star.get_center,
                stroke_color=star_color,
                stroke_width=2,
                dissipating_time=None  # This keeps the lines permanent
            )

            # Label that follows the star
            label = Text(str(names_val[idx]), font_size=12, color=star_color)
            self.add_fixed_orientation_mobjects(label)
            label.add_updater(lambda m, s=star: m.move_to(s.get_center() + OUT * 0.1))

            self.add(star, trail, label)

            # 4. The Physics Updater
            def update_star(mobj, dt, i=idx):
                # Speed up time if the stars move too slowly (e.g., years to seconds)
                t_curr = self.renderer.time * 5

                pos = orbitalPosition(
                    t_curr, a_val[i], e_val[i], i_val[i],
                    Omega_val[i], w_val[i], Tp_val[i], Per_val[i]
                )

                # Apply scaling and axis inversion (matplotlib vs manim)
                # x is inverted in your plot, so we use -pos[0]
                mobj.move_to(np.array([-pos[0], pos[1], pos[2]]) * SCALE_FACTOR)

            star.add_updater(update_star)

        # 5. The "Cinematic" Rotation
        self.begin_ambient_camera_rotation(rate=0.15)

        # Adjust this wait time to see more or less of the orbit
        self.wait(10)