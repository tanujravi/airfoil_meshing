## OpenFOAM mesh creation
This folder contains the files necessary to extrude the mesh from an obj file and also to create the patches required to assign the boundary conditions. The following steps are to be carried out:

1. Copy the mesh in .obj format to the top level of this folder. The obj file should be named "mesh.obj"
2. Run the Allrun script
```
./Allrun
```
3. Mesh is generated and can be viewed in paraview using,
```
touch para.foam
paraview para.foam
```