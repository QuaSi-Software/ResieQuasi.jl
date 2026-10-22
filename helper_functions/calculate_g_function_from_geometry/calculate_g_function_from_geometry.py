"""Calculate and export the g-function of a borehole field from a shapefile or x/y coordinates.

Outputs in the 'output_g_function' folder:
    - g_function.csv
    - g_function.prf (ReSiE format)
    - probe_field.png
    - g_function_time.png
    - g_function_resie.png

Shapefile mode requires one point geometry per borehole and a correctly
defined CRS file (.prj). Shapefile paths are relative to this script.
Coordinate mode takes (x, y) pairs in metres; no Shapefile is required.
"""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygfunction as gt


# --------------------------------------------------
# 1. Input parameters
# --------------------------------------------------

# Borehole position input: use a Shapefile or manual x/y coordinates
input_mode = "shapefile"  # "shapefile" or "coordinates"
shapefile = "probe_field_shape.shp"

# Only used when input_mode = "coordinates".
# One (x, y) pair per borehole, in metres in a common planar coordinate system.
probe_coordinates = [
    (0.0, 0.0),
    (6.0, 0.0),
    (12.0, 0.0),
    (0.0, 6.0),
    (6.0, 6.0),
    (12.0, 6.0),
]

# Borehole
probe_depth = 150.0                   # [m] Total depth below ground level
probe_head_depth = 0.0                # [m] Depth of borehole head
borehole_diameter = 0.135             # [m] Borehole diameter

# Ground
soil_density = 1500.0                 # [kg/m³]
soil_specific_heat_capacity = 1450.0  # [J/(kg K)]
soil_heat_conductivity = 2.9          # [W/(m K)]

# Time range and calculation
start_time_hours = 24.0               # [h] Note: The short-term function is calculated internally by ReSiE and is therefore not required here. 
end_time_years = 100                  # [years] Time range of the g-function
n_time_points = 120                   # [-] Number of g-function data points
# Note: If ReSiE produces errors such as "Duplicated knots were deduplicated", 
#       the number of data points is too high at the beginning of the simulation.

# Calculation method
# "detailed":     Explicitly calculates thermal interactions of all segment pairs. Computational effort: High
# "similarities": Uses geometric similarities to speed up the calculation. Computational effort: Medium
# "equivalent":   Groups thermally similar boreholes into equivalent groups. Computational effort: Low
method = "similarities"

# Boundary condition
# "UBWT": Uniform Borehole Wall Temperature
# "UHTR": Uniform Heat Transfer Rate per unit borehole length
# "MIFT": Mixed Inlet Fluid Temperatures, accounts for the hydraulic borehole network and fluid temperatures
boundary_condition = "UBWT"

# Borehole discretization
n_segments = 12                     # Number of segments per borehole

# Output
output_folder = "output_g_function"
show_plots = True                   # Display in addition to saving


# --------------------------------------------------
# 2. Derived parameters and validation
# --------------------------------------------------

if min(
    probe_depth, borehole_diameter, soil_density,
    soil_specific_heat_capacity, soil_heat_conductivity,
    start_time_hours, end_time_years,
) <= 0:
    raise ValueError("Lengths, ground parameters, and times must be positive.")
if probe_head_depth < 0 or probe_head_depth >= probe_depth:
    raise ValueError("probe_head_depth must be between 0 and probe_depth.")
if n_time_points < 2:
    raise ValueError("n_time_points must be at least 2.")

H = probe_depth - probe_head_depth  # [m] thermally active borehole length
D = probe_head_depth                # [m] Depth of borehole head
r_b = borehole_diameter / 2.0       # [m] Borehole radius
alpha = soil_heat_conductivity / (soil_density * soil_specific_heat_capacity)  # [m²/s]
t_s = H**2 / (9.0 * alpha)          # [s] characteristic time according to Eskilson
seconds_per_year = 365.25 * 24 * 3600

script_dir = Path(__file__).resolve().parent
output_dir = script_dir / output_folder

if input_mode not in ("shapefile", "coordinates"):
    raise ValueError('input_mode must be "shapefile" or "coordinates".')
if input_mode == "shapefile":
    shapefile_path = script_dir / shapefile
    if not shapefile_path.is_file():
        raise FileNotFoundError(f"Shapefile not found: {shapefile_path}")
output_dir.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 3. Determine borehole positions
# --------------------------------------------------

if input_mode == "shapefile":
    gdf = gpd.read_file(shapefile_path)
    if gdf.empty:
        raise ValueError("The shapefile contains no boreholes.")
    if gdf.crs is None:
        raise ValueError("The coordinate reference system (CRS) is missing.")
    if not gdf.geom_type.eq("Point").all():
        raise ValueError("The shapefile must contain only point geometries.")
    if gdf.geometry.is_empty.any() or gdf.geometry.isna().any():
        raise ValueError("The shapefile contains empty or missing geometries.")

    # Reproject to a metric UTM CRS.
    gdf = gdf.to_crs(gdf.estimate_utm_crs())
    x = gdf.geometry.x.to_numpy()
    y = gdf.geometry.y.to_numpy()
else:
    # Coordinates are already planar, in metres; do not transform the CRS.
    try:
        coordinates = np.asarray(probe_coordinates, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("probe_coordinates must contain numeric (x, y) pairs.") from exc
    if coordinates.ndim != 2 or coordinates.shape[1] != 2 or len(coordinates) == 0:
        raise ValueError("probe_coordinates must be a non-empty list of (x, y) pairs.")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("probe_coordinates must contain only finite numbers.")
    x = coordinates[:, 0].copy()
    y = coordinates[:, 1].copy()

# Shift to a local origin without changing the borehole spacing.
x = x - x.mean()
y = y - y.mean()

borefield = [
    gt.boreholes.Borehole(H=H, D=D, r_b=r_b, x=float(xi), y=float(yi))
    for xi, yi in zip(x, y)
]

duplicates = gt.boreholes.find_duplicates(borefield)
if duplicates:
    raise ValueError(f"Duplicate or overlapping boreholes: {duplicates}")

number_of_probes = len(borefield)
print(f"Number of boreholes: {number_of_probes}")
print(f"Thermal diffusivity alpha: {alpha:.6e} m²/s")
print(f"Characteristic time t_s: {t_s:.6e} s")


# --------------------------------------------------
# 4. Calculation of the g-function
# --------------------------------------------------

time = np.geomspace(
    start_time_hours * 3600,
    end_time_years * seconds_per_year,
    n_time_points,
)

if not np.all(np.diff(time) > 0):
    raise ValueError("The calculation time range is invalid.")

gfunc = gt.gfunction.gFunction(
    borefield,
    alpha=alpha,
    time=time,
    method=method,
    boundary_condition=boundary_condition,
    options={"nSegments": n_segments},
)

g_values = np.asarray(gfunc.gFunc).ravel()
log_time = np.log(time / t_s)       # ReSiE x-axis: ln(t / t_s)

if len(g_values) != len(time) or not np.all(np.isfinite(g_values)):
    raise ValueError("The g-function returned invalid or missing values.")


# --------------------------------------------------
# 5. Data export as CSV and ReSiE file
# --------------------------------------------------

results = pd.DataFrame({
    "time_s": time,
    "time_years": time / seconds_per_year,
    "log_time": log_time,
    "g_function": g_values,
})
results.to_csv(output_dir / "g_function.csv", index=False)

output_file = output_dir / "g_function.prf"
with output_file.open("w", encoding="utf-8", newline="\n") as f:
    f.write("G-function calculated with pygfunction\n")
    f.write(f"number_of_probes: {number_of_probes}\n")
    f.write("***\n")
    for log_t, g_val in zip(log_time, g_values):
        f.write(f"{log_t:.8f}; {g_val:.8f}\n")


# --------------------------------------------------
# 6. Create and save plots
# --------------------------------------------------

# 6a. Borehole field (metric x/y coordinates)
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(x, y, s=35)
ax.set_aspect("equal", adjustable="datalim")
ax.grid(True, alpha=0.4)
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_title(f"Borehole field ({number_of_probes} boreholes)")
fig.tight_layout()
fig.savefig(output_dir / "probe_field.png", dpi=300, bbox_inches="tight")

# 6b. g-function over time in years (logarithmic x-axis)
fig, ax = plt.subplots(figsize=(8, 5))
ax.semilogx(time / seconds_per_year, g_values)
ax.grid(True, which="both", alpha=0.4)
ax.set_xlabel("Time [years] (logarithmic)")
ax.set_ylabel("g-function [-]")
ax.set_title("g-function over time")
fig.tight_layout()
fig.savefig(output_dir / "g_function_time.png", dpi=300, bbox_inches="tight")

# 6c. g-function over normalized logarithmic time (linear ReSiE x-axis)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(log_time, g_values)
ax.grid(True, alpha=0.4)
ax.set_xlabel(r"Normalized logarithmic time $\ln(t/t_s)$ [-]")
ax.set_ylabel("g-function [-]")
ax.set_title("g-function – ReSiE representation")
fig.tight_layout()
fig.savefig(output_dir / "g_function_resie.png", dpi=300, bbox_inches="tight")

# Display all plots or close the figures without displaying them.
if show_plots:
    plt.show()
else:
    plt.close("all")

print(f"Results saved to: {output_dir}")
print("  - g_function.csv")
print("  - g_function.prf")
print("  - probe_field.png")
print("  - g_function_time.png")
print("  - g_function_resie.png")
