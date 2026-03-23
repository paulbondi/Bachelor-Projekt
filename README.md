# orbit_kepler.py — Orbital Analysis of S-Stars

A Python toolkit for computing, visualising, and simulating the Keplerian orbits of S-stars around the Galactic Centre supermassive black hole (Sgr A*). It reads a standard FITS orbital-element catalogue, solves Kepler's equation via Newton–Raphson, and provides a command-line interface for plotting, astrometric prediction, and synthetic MICADO/ELT imaging.

---

## Table of Contents

- [Requirements](#requirements)
- [Input Data](#input-data)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
- [Function Reference](#function-reference)
- [Configuration Constants](#configuration-constants)
- [Examples](#examples)

---

## Requirements

| Package | Role |
|---|---|
| `numpy` | Numerical core |
| `matplotlib` | Plotting |
| `astropy` | FITS I/O, units, tables |
| `scopesim` | ELT/MICADO instrument simulation |
| `scopesim_templates` | Synthetic stellar field generation |
| `sep` | Source extraction & photometry |

Install all dependencies with:

```bash
pip install numpy matplotlib astropy scopesim scopesim_templates sep
```

The first time `simulate` or `orbitfit` is run, ScopeSim may need to download instrument packages. Uncomment and run this line once:

```python
sim.download_packages(["Armazones", "ELT", "MICADO"])
```

---

## Input Data

The script expects a FITS binary table in the format of **Gillessen et al. 2017** (*ApJ* 837, 30), Table 3. The default filename is:

```
J_ApJ_837_30_table3.dat.fits
```

Pass a different path with `--data`. The table must contain the following columns:

| Column | Description | Unit |
|---|---|---|
| `Star` | Star identifier (e.g. `S2`) | — |
| `SpT` | Spectral type | — |
| `Kmag` | K-band magnitude | mag |
| `a` / `e_a` | Semi-major axis ± uncertainty | arcsec |
| `e` / `e_e` | Eccentricity ± uncertainty | — |
| `i` / `e_i` | Inclination ± uncertainty | deg |
| `Omega` / `e_Omega` | Longitude of ascending node ± | deg |
| `w` / `e_w` | Argument of periapsis ± | deg |
| `Tp` / `e_Tp` | Time of pericentre passage ± | yr |
| `Per` / `e_Per` | Orbital period ± | yr |

---

## Installation

No installation is needed beyond the dependencies above. Simply place `orbit_kepler.py` and the FITS catalogue in the same directory, then run directly with Python.

---

## Quick Start

```bash
# Plot orbits of all stars from 2030 to 2050
python orbit_kepler.py position

# Plot the orbit of S2 from 1992 to 2025
python orbit_kepler.py position --star S2 --t_start 1992 --t_end 2025

# Find the next pericentre passage for S2
python orbit_kepler.py pericentre --star S2

# Find the best observing window for S14 between 2025 and 2035
python orbit_kepler.py bestobs --star S14 --t_start 2025 --t_end 2035
```

---

## CLI Reference

All subcommands accept a global `--data` flag:

```bash
python orbit_kepler.py --data /path/to/catalogue.fits <subcommand> [options]
```

### `position` — Plot calculated orbits

```bash
python orbit_kepler.py position [--star <name> ...] [--t_start <yr>] [--t_end <yr>]
```

Plots the sky-projected (R.A., Dec.) orbital tracks for one or more stars. Defaults to all stars, 2030–2050. The Galactic Centre is marked at the origin.

### `velocity` — Plot orbital speeds

```bash
python orbit_kepler.py velocity [--star <name> ...] [--t_start <yr>] [--t_end <yr>]
```

Plots the orbital speed (mas/yr) as a function of time. Defaults to all stars, 1992–2012.

### `pericentre` — Find pericentre passages

```bash
# Next single passage near the catalogued Tp
python orbit_kepler.py pericentre --star S2

# All passages within a time window
python orbit_kepler.py pericentre --star S2 --t_start 2000 --t_end 2050
```

Prints the time (year and calendar date), closest approach distance, and weekly angular displacement at pericentre. Multiple star names may be given.

### `bestobs` — Find best observing window

```bash
python orbit_kepler.py bestobs --star S14 --t_start 2025 --t_end 2035
```

Identifies the moment within `[t_start, t_end]` when the star is closest to Sgr A*, and reports the weekly angular displacement as a measure of observability. `--t_start` and `--t_end` are required.

### `compare` — Compare calculated vs simulated positions

```bash
python orbit_kepler.py compare [--star <name> ...]
```

Runs a MICADO simulation at the observation epoch (`t_obs = 2030`) and overlays SEP-detected positions against Keplerian predictions. Prints per-star position errors and overall RMS.

### `sim` — Display a synthetic MICADO image

```bash
python orbit_kepler.py sim [--star <name> ...]
```

Shows the raw simulated MICADO detector image (log-scaled, greyscale) for the selected stars.

### `orbitfit` — Fit an ellipse to simulated orbit points

```bash
python orbit_kepler.py orbitfit [--star <name>]
```

Runs 5 simulations at annual epochs (2032–2036), detects the star in each image, and fits a conic section (ellipse) to the resulting pixel positions using Singular Value Decomposition. Prints the six conic coefficients (A–F) and RMSE.

---

## Function Reference

### Data loading

**`load_data(filename)`**
Reads the FITS catalogue and populates all global orbital-element arrays. Also pre-computes `orbit_table` at the global `t_obs` epoch. Called automatically by the CLI.

### Core orbital mechanics

**`kepler(M, e, epsilon=1e-9, max_it=1000)`**
Solves Kepler's equation M = E − e·sin(E) for the eccentric anomaly E using Newton–Raphson iteration. Accepts scalar or NumPy array mean anomalies.

**`orbitalPosition(t, a, e, i, Omega, w_, Tp, Per)`**
Returns `(ra, dec, z)` in arcsec at time `t` (years). Angles are taken in degrees and converted internally.

**`orbitalVelocity(t, a, e, i, Omega, w, Tp, Per)`**
Returns the instantaneous orbital speed in arcsec/yr using the vis-viva equation in the orbital plane.

### Table & star resolution

**`orbitTable(t_obs, names, a, e, i, Omega, w, Tp, Per, kmag, spectral)`**
Builds an Astropy `Table` with columns `[x, y, ref, mag, type]` (positions in arcsec) for all stars at epoch `t_obs`.

**`resolveStars(star_name)`**
Maps a name string, list, or `"all"` to index arrays into the global catalogue. Returns `(indices, label)`, or `(None, None)` if no match is found.

### Simulation & detection

**`simulate(star_name)`**
Creates a synthetic MICADO/ELT image using ScopeSim (SCAO mode, 1.5 mas/pixel, 1 hour total exposure). Returns the raw detector array.

**`findStars(star_name)`**
Runs `simulate`, extracts sources with SEP (iterating the detection threshold until the expected number of sources is found), and returns measured `(x, y)` positions in arcsec.

### Analysis & pericentre tools

**`pericentrePass(star_name, t_start, t_end)`**
Finds one or all pericentre passages. Without a time window, returns the passage nearest to the catalogued Tp. With a window, scans every orbital period and reports each passage within it. Calls `findPass` for formatted output.

**`findPass(star_name, t_peri, r_peri, a, e, i, Omega, w, Tp, Per, passage_num)`**
Helper that computes and prints a formatted pericentre summary: date, closest distance (mas), weekly displacement vector (mas).

**`bestObserving(star_name, t_start, t_end)`**
Scans the interval at high resolution to find the moment of minimum separation from Sgr A*, and reports the observing window statistics.

**`orbitFit(star_name)`**
Fits a general conic Ax² + Bxy + Cy² + Dx + Ey + F = 0 to 5 annual simulation detections via SVD. Returns a dict with coefficients and RMSE.

### Plotting

| Function | Description |
|---|---|
| `positionPolt(star_name, t_start, t_end)` | Sky-plane orbit track scatter plot |
| `velocityPlot(star_name, t_start, t_end)` | Orbital speed vs. time line plot |
| `comparePlot(star_name)` | Calculated (+) vs. simulated (×) position overlay |
| `spectralPlotSim(star_name)` | Raw MICADO image display |

---

## Configuration Constants

These are set at the top of the script and can be adjusted directly:

| Constant | Default | Description |
|---|---|---|
| `PIXEL_SCALE` | `0.0015` arcsec/pixel | MICADO 1.5 mas plate scale |
| `SEP_THRESHOLD_MAX` | `250` | Maximum SEP detection threshold before issuing a warning |
| `t_obs` | `2030` | Default observation epoch used for `orbit_table` and simulations |

---

## Examples

```bash
# Multi-star orbit plot, custom epoch range
python orbit_kepler.py position --star S2 S14 S38 --t_start 2020 --t_end 2040

# All pericentre passages of S2 across 50 years
python orbit_kepler.py pericentre --star S2 --t_start 2000 --t_end 2050

# Best observation epoch for S38 in the next decade
python orbit_kepler.py bestobs --star S38 --t_start 2026 --t_end 2036

# Velocity profile for a single star
python orbit_kepler.py velocity --star S2 --t_start 2018 --t_end 2026

# Ellipse fit from simulated astrometry
python orbit_kepler.py orbitfit --star S2

# Use a different data file
python orbit_kepler.py --data /data/my_catalogue.fits position --star S2
```
