from Assemble import Assemble
from Connect import Connect
from yaml import safe_load
from mesh import Mesh
import meshio
import sys
import numpy as np

def extrude_mesh(mesh_name, length = 0.01):
    m = meshio.read(mesh_name)  # surface: triangles/quads

    pts = m.points
    tris = m.cells_dict.get("triangle", [])
    quads = m.cells_dict.get("quad", [])

    th = length  # small thickness
    top = pts.copy(); top[:,2] += th
    points = np.vstack([pts, top])
    n = len(pts)

    #prisms = np.array([[a,b,c,a+n,b+n,c+n] for a,b,c in (tris or [])], dtype=int)
    hexes  = np.array([[a,b,c,d,a+n,b+n,c+n,d+n] for a,b,c,d in quads], dtype=int)

    cells = {}
    #if len(prisms): cells["wedge"] = prisms
    if len(hexes):  cells["hexahedron"] = hexes


    mesh3d = meshio.Mesh(points, cells)
    meshio.write("mesh3d.vtk", mesh3d, file_format="vtk", binary=False)


def extrude_to_3d_gmsh_quads_only(mesh_obj,
                                  thickness=0.01,
                                  msh_name="mesh3d.msh",
                                  zmin_name="front",
                                  zmax_name="back",
                                  write_vtu_preview=False):
    """
    Extrude a 2D quad mesh into 3D hex mesh and write Gmsh v2.2 with physical tags.
    Requires:
      mesh_obj.mesh -> (vertices, connectivity) with ONLY quads in connectivity (0-based)
      mesh_obj.boundary_tags -> dict[str] = list[(i,j)] edges (0-based, boundary-only)

    Physical tags:
      - each side patch uses the SAME name as in boundary_tags and is a surface (dim=2)
      - bottom (z=0) surface named zmin_name (dim=2)
      - top    (z=th) surface named zmax_name (dim=2)
      - volume named "Domain" (dim=3)
    Physical and geometrical tags are set to the SAME integers.
    """
    # --- vertices (inject z if needed) ---
    V2, quads2d = mesh_obj.mesh
    V = np.asarray(V2, float)
    if V.shape[1] == 2:
        V = np.c_[V, np.zeros(len(V))]  # z=0 plane
    quads2d = np.asarray(quads2d, dtype=int)
    if quads2d.size == 0 or (quads2d.ndim != 2 or quads2d.shape[1] != 4):
        raise ValueError("This function expects ONLY quad cells in the 2D domain.")

    n2d = V.shape[0]
    th = float(thickness)

    # --- extrude points ---
    top = V.copy()
    top[:, 2] += th
    points3d = np.vstack([V, top])

    # --- volume hexes from quads ---
    # (a,b,c,d) -> (a,b,c,d,a+n,b+n,c+n,d+n)
    hexes = np.c_[quads2d,
                  quads2d + n2d]

    # --- z-planes (bottom/top) as quads ---
    quad_bottom = quads2d
    quad_top    = quads2d + n2d

    # --- side faces from boundary edge tags (each edge -> a quad) ---
    side_faces_by_name = {}
    btags = getattr(mesh_obj, "boundary_tags", {}) or {}
    for nm, edges in btags.items():
        if not edges:
            continue
        e = np.asarray(edges, dtype=int)
        # consistent ordering: (i, j, j+n, i+n)
        side_faces_by_name[nm] = np.c_[e[:, 0], e[:, 1], e[:, 1] + n2d, e[:, 0] + n2d]

    # --- assign physical (== geometrical) tags ---
    field_data = {}
    next_tag = 1

    name_to_tag = {}
    # side patches (dim=2)
    for nm in side_faces_by_name.keys():
        name_to_tag[nm] = next_tag
        field_data[nm] = np.array([2, next_tag], dtype=np.int32)
        next_tag += 1

    # z planes (dim=2)
    tag_zmin = next_tag; field_data[zmin_name] = np.array([2, tag_zmin], dtype=np.int32); next_tag += 1
    tag_zmax = next_tag; field_data[zmax_name] = np.array([2, tag_zmax], dtype=np.int32); next_tag += 1

    # volume (dim=3)
    tag_domain = next_tag; field_data["Domain"] = np.array([3, tag_domain], dtype=np.int32)

    print(field_data)
    # --- build meshio cells + cell_data (physical == geometrical) ---
    cells = []
    phys = []
    geom = []

    # volume first
    cells.append(("hexahedron", hexes))
    phys.append(np.full(len(hexes), tag_domain, dtype=np.int32))
    geom.append(np.full(len(hexes), tag_domain, dtype=np.int32))

    # z surfaces
    cells.append(("quad", quad_bottom))
    phys.append(np.full(len(quad_bottom), tag_zmin, dtype=np.int32))
    geom.append(np.full(len(quad_bottom), tag_zmin, dtype=np.int32))

    cells.append(("quad", quad_top))
    phys.append(np.full(len(quad_top), tag_zmax, dtype=np.int32))
    geom.append(np.full(len(quad_top), tag_zmax, dtype=np.int32))

    # side patches, one block per patch name (keeps one physical per block)
    for nm, faces in side_faces_by_name.items():
        if len(faces):
            cells.append(("quad", faces))
            tag = name_to_tag[nm]
            phys.append(np.full(len(faces), tag, dtype=np.int32))
            geom.append(np.full(len(faces), tag, dtype=np.int32))

    cell_data = {"gmsh:physical": phys, "gmsh:geometrical": geom}
    write.gmsh22_ascii("mesh_new.msh", points3d, cells, cell_data=cell_data, field_data=field_data)

    """
    mesh3d = meshio.Mesh(points3d, cells, cell_data=cell_data, field_data=field_data)

    # write gmsh v2.2 ASCII (gmshToFoam-friendly)
    meshio.write(msh_name, mesh3d, file_format="gmsh22", binary = False)

    if write_vtu_preview:
        meshio.write(msh_name.replace(".msh", ".vtu"), mesh3d)
    """
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
    #mesh.extrude_mesh()
    mesh.write_obj("mesh.obj")
    #mesh.writeVTK_legacy("mesh.vtk")
    #extrude_mesh("mesh.vtk", 0.01)
    mesh.extrudeTo3d(thickness=0.1, zmax_name="front", zmin_name="back")
    mesh.write_gmsh22_ascii(msh_name = "mesh_new.msh")