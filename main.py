from Assemble import Assemble
from Connect import Connect
from yaml import safe_load
import sys

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
    connect.write_obj(vertices,connectivity, "mesh.obj")
