# Create a 2D C-type airfoil mesh

The goal is creating 2D high-quality airfoil mesh given a contour and as few as possible parameters to define the mesh. A depth-extruded version of the mesh should be suitable for DDES ([meshing guideline](https://www.aiaa-dpw.org/ref/gridding_guidelines_v3_07012024.pdf)).

Next steps:
- create near-airfoil mesh by extruding contour points in normal direction; prescribed parameters should be:
  - first-layer thickness
  - extrusion distance in normal direction
  - growth ratio
  - separate parameters for trailing edge (to be determined)
- create remaining farfield blocks to create C-type mesh; prescribed parameters should be:
  - C-shape: radius, wake length in chord length (e.g., r=50c, wl=100c)
  - number of cells
  - first cell layer should be a smooth extension of the inner blocks' last layer 
- save mesh in *.obj* format; should include boundary definition to create patches
- implement smoothing for farfield mesh

Reference implementation:
- [pyAero](https://github.com/chiefenne/PyAero); the package implements many of the required features; however, initial tests revealed the following issues:
  - loading the contour data is not very robust
  - the implementation can't deal with contours that already come with a tailing edge
  - the contour refinement is not smooth enough for DDES-type simulations; the algorithms introduces 1.5-ratio jumps in the cell edge length along the contour

## Dependencies

Setting up a suitable Python environment:
```
python3 -m venv aero
source aero/bin/activate
pip install -U pip
pip install -r requirements.txt
```
The OAT15A airfoil contour data can be downloaded [here](https://aiaa-dpw.larc.nasa.gov/geometry.html).

## How to run

To test the contour loading and point distribution, run:
```
source aero/bin/activate
python airfoil.py
```