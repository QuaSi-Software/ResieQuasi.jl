# Geothermal borefield g-function helper

This standalone Python script calculates the **g-function of a geothermal borehole field** from borehole coordinates stored in an ESRI Shapefile. It uses [pygfunction](https://pygfunction.readthedocs.io/) and exports a g-function file that can be imported by the **ReSiE `GeothermalProbes` component**.

The tool supports irregular borehole layouts. It is an **optional preprocessing utility**, not a required Python dependency of ReSiE, and does not need a running Julia environment.

## Installation

Python 3 and the packages listed in `requirements.txt` are required:

```bash
python -m pip install -r requirements.txt
```

Run this command from `helper_functions/calculate_g_function_from_shape_file/`, or pass the path to its `requirements.txt` file.

## Input: borehole coordinates

Supply an ESRI Shapefile with **one point feature per borehole**. The following files must share the same base name and be placed alongside the Python script (unless `shapefile` is changed to a different relative or absolute path):

| File | Purpose |
| --- | --- |
| `probe_field_shape.shp` | Point geometries (required) |
| `probe_field_shape.shx` | Geometry index (required) |
| `probe_field_shape.dbf` | Feature attributes (required) |
| `probe_field_shape.prj` | Coordinate reference system (required by this script) |
| `probe_field_shape.cpg` | Character encoding (optional) |

The Shapefile must have a valid coordinate reference system. The script estimates an appropriate UTM CRS, reprojects the points into metric coordinates, and shifts them to a local origin. It rejects empty or missing point geometries and overlapping/duplicate boreholes.

All boreholes are currently assigned the **same depth, borehole diameter, and buried depth** from the script configuration; attributes in the Shapefile are not used to set individual borehole dimensions.

## Configuration

Edit the input parameter block near the top of `calculate_g_function_from_shape_file.py`:

| Setting | Meaning | Included example |
| --- | --- | --- |
| `shapefile` | Shapefile name/path | `probe_field_shape.shp` |
| `probe_depth` | Total drilled depth below ground level, m | `150.0` |
| `probe_head_depth` | Buried depth of borehole head, m | `0.0` |
| `borehole_diameter` | Borehole diameter, m | `0.135` |
| `soil_density` | Ground density, kg/m³ | `1500.0` |
| `soil_specific_heat_capacity` | Specific heat capacity, J/(kg K) | `1450.0` |
| `soil_heat_conductivity` | Ground thermal conductivity, W/(m K) | `2.9` |
| `start_time_hours` | Earliest calculation time, hours | `24.0` |
| `end_time_years` | Latest calculation time, years | `100` |
| `n_time_points` | Number of logarithmically distributed time points | `120` |
| `method` | pygfunction solution method | `similarities` |
| `boundary_condition` | Borehole boundary condition | `UBWT` |
| `n_segments` | Number of vertical segments per borehole | `12` |
| `output_folder` | Folder used for generated files | `output_g_function` |
| `show_plots` | Also display the figures interactively | `True` |

The code computes the derived quantities:

- **Active borehole length:** `H = probe_depth - probe_head_depth` (m)
- **Buried head depth:** `D = probe_head_depth` (m)
- **Borehole radius:** `r_b = borehole_diameter / 2` (m)
- **Ground thermal diffusivity:** `alpha = soil_heat_conductivity / (soil_density * soil_specific_heat_capacity)` (m²/s)
- **Eskilson characteristic time:** `t_s = H**2 / (9 * alpha)` (s)

The included parameters are illustrative and should be replaced with project-specific values. `UBWT` assumes a uniform borehole wall temperature; the utility does not model fluid hydraulics or detailed borehole thermal resistance.

## Running the script

From the helper directory:

```bash
python calculate_g_function_from_shape_file.py
```

The script resolves `shapefile` and `output_folder` relative to its own file location, not the shell's current working directory. All outputs are written to `output_g_function/` (unless changed).

## Output files

| Output | Content |
| --- | --- |
| `g_function.prf` | ReSiE-compatible custom g-function file |
| `g_function.csv` | Calculation times in seconds and years, normalized log time, g-function values |
| `probe_field.png` | Borehole layout (metric x/y coordinates) |
| `g_function_time.png` | g-function vs. years (logarithmic x-axis) |
| `g_function_resie.png` | g-function vs. `ln(t/t_s)` (linear ReSiE x-axis) |

The `g_function.prf` file has this structure (illustrative values only):

```text
G-function calculated with pygfunction
number_of_probes: 42
***
-8.84569000; 1.07147000
-7.90838000; 1.38002000
...
```

The **first column** is the dimensionless normalized logarithmic time `ln(t/t_s)`; the **second** is the dimensionless g-function value. The number of boreholes in the file header is inferred from the Shapefile.

## Import into ReSiE

Point the `GeothermalProbes` component to the generated file by setting `g_function_file_path` in your ReSiE input configuration, for example:

```json
"g_function_file_path": "path/to/output_g_function/g_function.prf"
```

Use a path that can be resolved by your ReSiE run. See the [ReSiE component parameter documentation](https://quasi-software.readthedocs.io/en/latest/resie_component_parameters/#geothermal-probes) for other required component settings and the full import format.

**Consistency is essential:** ReSiE must use the same *thermally active borehole length* as the value `H` used by pygfunction when calculating `t_s`. With the included example (`probe_depth = 150.0`, `probe_head_depth = 0.0`), set ReSiE `probe_depth` to `150.0 m`. With a nonzero `probe_head_depth`, set ReSiE `probe_depth` to `probe_depth - probe_head_depth` from this script. Keep the ground thermal conductivity consistent with the g-function calculation.

## Time sampling and ReSiE interpolation

The script evaluates `n_time_points` geometrically spaced times from `start_time_hours` to `end_time_years`. ReSiE may round converted time coordinates to the simulation time-step grid. If two points collapse onto the same time after rounding, Julia's `Interpolations.jl` may emit a **`Duplicated knots were deduplicated`** warning.

For an hourly simulation, the included `start_time_hours = 24.0` and `n_time_points = 120` are practical starting values, **not a guarantee** that no rounded time points will coincide under every setup. Adjust the sampling range or density if ReSiE reports duplicate knots, and ensure the imported curve covers the time range required by your simulation.

## Model assumptions and limitations

- A homogeneous ground with constant effective thermophysical properties is assumed.
- All boreholes are modeled as vertical and assigned common geometry parameters.
- The point locations must represent borehole heads in a valid CRS; check coordinates and distances before use.
- The numerical output is sensitive to the boundary condition, time range, solver, and vertical discretization; refine `n_segments` and compare results when accuracy requirements are high.
- This script only **generates** the g-function input; it does not run ReSiE or validate a geothermal system design.

## References

- [pygfunction documentation](https://pygfunction.readthedocs.io/)
- [ReSiE geothermal probe parameter documentation](https://quasi-software.readthedocs.io/en/latest/resie_component_parameters/#geothermal-probes)
- [ReSiE sample custom g-function](https://quasi-software.readthedocs.io/en/latest/data/validation_geothermal_probe/custom_g-function_geothermal_probe.txt)
