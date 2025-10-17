from Assemble import Assemble
from Connect import Connect
from yaml import safe_load
from mesh import Mesh
import meshio
import sys
import numpy as np



if __name__ == "__main__":

    config_file = sys.argv[1] if len(sys.argv) > 1 else "mesh_config.yaml"
    try:
        with open(config_file, "r") as cf:
            config = safe_load(cf)
    except Exception as e:
        print(e)

    assemble = Assemble(config)
    assemble.assemble()
    connect = Connect()
    vertices, connectivity = connect.connectAllBlocks(assemble.blocks)
    mesh = Mesh(vertices, connectivity)
    mesh.write_obj("mesh.obj")
    mesh.extrudeTo3d(thickness=0.1, zmax_name="front", zmin_name="back")
    mesh.write_gmsh22_ascii(msh_name = "mesh_new.msh")