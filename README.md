# 2D C-Type Airfoil Mesh Generation

This project focuses on the generation of high-quality 2D C-type meshes for airfoil geometries, starting from a given airfoil contour and using as few user-defined parameters as possible. The generated meshes are designed such that a depth-extruded version is suitable for Delayed Detached Eddy Simulation (DDES) ([meshing guideline](https://www.aiaa-dpw.org/ref/gridding_guidelines_v3_07012024.pdf)).

**Background and Reference**:
- [pyAero](https://github.com/chiefenne/PyAero); the package was evaluated as a reference. Although, it implements many of the required features; however, initial tests revealed the following issues:
  - loading the contour data is not very robust
  - the implementation can't deal with contours that already come with a trailing edge
  - the contour refinement is not smooth enough for DDES-type simulations; the algorithms introduces 1.5-ratio jumps in the cell edge length along the contour

The current implementation addresses the above limitations through the following:

- Handling contours blunt trailing edge, by replacing it with a smooth curve.
- A smooth point-distribution algorithm for the airfoil surface based on:
  - mesh-size ratio,
  - weighting functions to concentrate mesh points in regions of interest (e.g. leading edge or shock buffet regions).

**Mesh topology**:

The mesh is generated using a hybrid structured–unstructured strategy:

- Structured mesh blocks are created in critical regions:
  - the airfoil boundary layer,
  - buffet region.
- The remaining domain is filled with **unstructured triangular elements**, generated using the [Gmsh](https://github.com/live-clones/gmsh/tree/master) library.
Options are provided to make the wake region finer.  

<p align="center">
  <img src="data/images/base/airfoil_main.png" width="600">
</p>
<p align="center">
  <img src="data/images/base/airfoil_main_zoom_1.png" width="600">
</p>
<p align="center">
  <img src="data/images/base/airfoil_main_zoom_2.png" width="45%">
  <img src="data/images/base/airfoil_main_zoom_3.png"  width="45%">
</p>

The mesh settings used to generate this mesh are provided in the default "mesh_config.yaml" file.

## Dependencies
To clone the repository:
```
git clone https://github.com/tanujravi/airfoil_meshing.git
cd airfoil_meshing
```
Setting up a suitable Python environment:
```
python3 -m venv aero
source aero/bin/activate
pip install -U pip
pip install -r requirements.txt
```
The OAT15A airfoil contour data can be downloaded [here](https://www.aiaa-dpw.org/geometry.html).

## How to run

To generate the mesh, run:
```
source aero/bin/activate
python main.py mesh_config.yaml

```
Mesh parameters can be adjusted in "mesh_config.yaml" file.

This generates a 2D mesh in obj format. The mesh can be extruded and converted to OpenFOAM format with boundary conditions included, using
```
cd openFOAM_mesh
cp ../mesh.obj .
./Allrun
```

Replace the mesh file name as required.

## Parameter variation
1. n_points (airfoil_boundary) - Increases the number of mesh points allocated to the surface of the airfoil contour
<table align="center">
  <tr>
    <td align="center">
      <img src="data/images/parameter_variation/n_points_airfoil_base.png" width="400"><br>
      <em>(a) n_points = 500</em>
    </td>
    <td align="center">
      <img src="data/images/parameter_variation/n_points_airfoil_act.png" width="400"><br>
      <em>(b) n_points = 700</em>
    </td>
  </tr>
</table>

2. weight_upper - Increases the number of mesh nodes distributed to the suction side of airfoil
<table align="center">
  <tr>
    <td align="center">
      <img src="data/images/parameter_variation/weight_upper_base.png" width="400"><br>
      <em>(a) weight_upper = 1</em>
    </td>
    <td align="center">
      <img src="data/images/parameter_variation/weight_upper_act.png" width="400"><br>
      <em>(b) weight_upper = 4</em>
    </td>
  </tr>
</table>
It is observed that the horizontal mesh length of one element in the buffet region is reduced from 0.0092 to 0.0085. Further refinement can be achieved by allocating fewer points in the trailing-edge region using the weight_te parameter, so that more points are distributed in the buffet zone.

3. weight_curvature - Increases the distibution of points where the curvature is high i.e. Nose of the airfoil. 
<table align="center">
  <tr>
    <td align="center">
      <img src="data/images/parameter_variation/weight_curvature_base.png" width="400"><br>
      <em>(a) weight_curvature = 4</em>
    </td>
    <td align="center">
      <img src="data/images/parameter_variation/weight_curvature_act.png" width="400"><br>
      <em>(b) weight_curvature = 10</em>
    </td>
  </tr>
</table>

Note: When multiple weighting functions are applied simultaneously, their sensitivities can differ significantly. In some cases, one weighting parameter must be varied by orders of magnitude before its effect becomes noticeable. This behavior was observed when adjusting weight_upper and weight_curvature at the same time. In particular, a value of weight_curvature = 100 was required before additional points appeared near the leading edge (nose) when increasing weight_upper.

4. weight_te - Increases points distribution towards the trailing edge region.
<table align="center">
  <tr>
    <td align="center">
      <img src="data/images/parameter_variation/weight_te_base.png" width="400"><br>
      <em>(a) weight_te = 4</em>
    </td>
    <td align="center">
      <img src="data/images/parameter_variation/weight_te_act.png" width="400"><br>
      <em>(b) weight_te = 10</em>
    </td>
  </tr>
</table>
The trailing edge region where this bias is applied is controlled using "fraction_te".

5. Airfoil boundary layer dimensions - Controls the dimensions of the airfoil boundary layer, namely the first-layer thickness (cell_thickness, t), the growth ratio (growth, r) and extrusion length (extrusion_distance, L).

<table align="center">
  <tr>
    <td align="center">
      <img src="data/images/parameter_variation/airfoil_bl_base.png" width="400"><br>
      <em>(a) t = 1e-5, r = 1.03, L = 0.3</em>
    </td>
    <td align="center">
      <img src="data/images/parameter_variation/airfoil_bl_act.png" width="400"><br>
      <em>(b) t = 2e-5, r = 1.04, L = 0.5</em>
    </td>
  </tr>
</table>

6. Shock box dimensions - Controls the dimensions of the structured mesh region where shock buffet is expected to happen.
<table align="center">
  <tr>
    <td align="center">
      <img src="data/images/parameter_variation/shock_box_base.png" width="400"><br>
      <em>(a) xmin = 0.05, xmax = 0.7, height = 0.6</em>
    </td>
    <td align="center">
      <img src="data/images/parameter_variation/shock_box_act.png" width="400"><br>
      <em>(b) xmin = 0.2, xmax = 0.6, height = 0.8</em>
    </td>
  </tr>
</table>

7. n_points (wake_tunnel) - Controls number of points used to make the boundary at the trailing edge.
<table align="center">
  <tr>
    <td align="center">
      <img src="data/images/parameter_variation/n_points_base.png" width="400"><br>
      <em>(a) n_points = 70</em>
    </td>
    <td align="center">
      <img src="data/images/parameter_variation/n_points_act.png" width="400"><br>
      <em>(b) n_points = 40</em>
    </td>
  </tr>
</table>

8. make_curve (wake) - Provides an option to enable or disable a curved trailing edge.
<table align="center">
  <tr>
    <td align="center">
      <img src="data/images/parameter_variation/make_curve_base.png" width="400"><br>
      <em>(a) make_curve = True</em>
    </td>
    <td align="center">
      <img src="data/images/parameter_variation/make_curve_act.png" width="400"><br>
      <em>(b) make_curve = False</em>
    </td>
  </tr>
</table>

9. Wake region dimensions - Controls the region of refinement in the wake region (trapezium-shaped).
<table align="center">
  <tr>
    <td align="center">
      <img src="data/images/parameter_variation/wake_dim_base.png" width="400"><br>
      <em>(a) length = 5, angle = 5 deg</em>
    </td>
    <td align="center">
      <img src="data/images/parameter_variation/wake_dim_act.png" width="400"><br>
      <em>(b) length = 8, angle = 10 deg</em>
    </td>
  </tr>
</table>

10. Minimum interface mesh size - Defines the target mesh size for unstructured triangular elements at the interface between the structured and unstructured regions.
It acts as a lower bound that prevents excessively small triangles from forming at the transition. In practice, it allows smaller structured elements near the interface to be merged into larger triangular elements, according to the specified threshold. 
<table align="center">
  <tr>
    <td align="center">
      <img src="data/images/parameter_variation/interface_min_size_base.png" width="400"><br>
      <em>(a) interface_min_size = 1e-6</em>
    </td>
    <td align="center">
      <img src="data/images/parameter_variation/interface_min_size_act.png" width="400"><br>
      <em>(b) interface_min_size = 1e-3</em>
    </td>
  </tr>
</table>

11. wake region mesh size - Controls the triangular element size in the wake region.
<table align="center">
  <tr>
    <td align="center">
      <img src="data/images/parameter_variation/wake_size_base.png" width="400"><br>
      <em>(a) wake_size = 0.02</em>
    </td>
    <td align="center">
      <img src="data/images/parameter_variation/wake_size_act.png" width="400"><br>
      <em>(b) wake_size = 0.01</em>
    </td>
  </tr>
</table>
Note: No. of elements for wake_size = 0.02 is 222,346 and for wake_size = 0.01 is 354,396.

12. Farfield mesh size - Controls the mesh sizing for filling remaining region.
<table align="center">
  <tr>
    <td align="center">
      <img src="data/images/parameter_variation/farfield_mesh_base.png" width="400"><br>
      <em>(a) min_size = 0.02, max_size = 3.2</em>
    </td>
    <td align="center">
      <img src="data/images/parameter_variation/farfield_mesh_act.png" width="400"><br>
      <em>(b) min_size = 0.02, max_size = 6</em>
    </td>
  </tr>
</table>

13. Fafield mesh grading distance - Specifies the spatial distance over which the mesh transitions from min_size to max_size. A larger value produces a smoother and more gradual grading, allowing the fine mesh region to extend farther into the domain. A smaller value results in a sharper transition and a more rapid increase in element size away from the body. 
<table align="center">
  <tr>
    <td align="center">
      <img src="data/images/parameter_variation/grading_base.png" width="400"><br>
      <em>(a) grading = 0.01</em>
    </td>
    <td align="center">
      <img src="data/images/parameter_variation/grading_act.png" width="400"><br>
      <em>(b) grading = 100</em>
    </td>
  </tr>
</table>
