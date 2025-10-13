import numpy as np
import os
import logging
logger = logging.getLogger(__name__)


class Mesh:
    def __init__(self, vertices, connectivity):
        # add mesh to Wind-tunnel instance
        self.mesh = vertices, connectivity

        # generate cell to vertex connectivity from mesh
        self.makeLCV()

        # generate cell to edge connectivity from mesh
        self.makeLCE()

        # generate boundaries from mesh connectivity
        self.makeBoundaries()
    
    def makeLCV(self):
        """Make cell to vertex connectivity for the mesh
           LCV is identical to connectivity
        """
        _, connectivity = self.mesh
        self.LCV = connectivity


    def makeLCE(self):
        """Make cell to edge connectivity for the mesh"""
        _, connectivity = self.mesh
        self.LCE = dict()
        self.edges = list()

        for i, cell in enumerate(connectivity):
            # example for quadrilateral:
            # cell: [0, 1, 5, 4]
            # edges: [(0,1), (1,5), (5,4), (4,0)]
            edges = [(cell[j], cell[(j + 1) % len(cell)])
                           for j in range(len(cell))]

            # all edges for cell i
            self.LCE[i] = edges

            # all edges in one list
            self.edges += [tuple(sorted(edge)) for edge in edges]

    def makeBoundaries(self):
        """A boundary edge is an edge that belongs only to one cell"""

        vertices, _ = self.mesh
        vertices = np.array(vertices)

        edges = self.edges

        seen = set()
        unique = list()
        doubles = set()
        for edge in edges:
            if edge not in seen:
                seen.add(edge)
                unique.append(edge)
            else:
                doubles.add(edge)

        self.boundary_edges = [edge for edge in unique if edge not in doubles]

        # tag edges for boundary definitions
        # FIXME
        # FIXME here it's done the dirty way
        # FIXME at least try to make it faster later
        # FIXME
        self.boundary_tags = {'airfoil': [],
                              'inlet': [],
                              'outlet': [],
                              'top': [],
                              'bottom': []}
        
        ### FIXME
        ### FIXME too dirty below (do not work with toplerances!!!)
        ### FIXME

        xmax = np.max(vertices[:,0])
        y_vals = vertices[:, 1]

        unique_y, counts = np.unique(y_vals, return_counts=True)

        sorted_idx = np.argsort(counts)[::-1]

        top_two_y = unique_y[sorted_idx[:2]]

        ymax = np.max(top_two_y)
        ymin = np.min(top_two_y)

        for edge in self.boundary_edges:
            x1 = vertices[edge[0]][0]
            y1 = vertices[edge[0]][1]
            x2 = vertices[edge[1]][0]
            y2 = vertices[edge[1]][1]
            tol = 1e-6  # tolerance for coordinate comparison
            if x1 > -0.1 and x1 < 1.1 and y1 < 0.5 and y1 > -0.5:
                self.boundary_tags['airfoil'].append(edge)
            elif abs(x1 - xmax) < tol and abs(x2 - xmax) < tol:
                self.boundary_tags['outlet'].append(edge)
            elif abs(y1 - ymax) < tol and abs(y2 - ymax) < tol:
                self.boundary_tags['top'].append(edge)
            elif abs(y1 - ymin) < tol and abs(y2 - ymin) < tol:
                self.boundary_tags['bottom'].append(edge)
            else:
                self.boundary_tags['inlet'].append(edge)

        return
    
    def write_obj(self, filename):
        with open(filename, 'w') as f:
            # Write vertices
            vertices, _ = self.mesh
            for v in vertices:
                if len(v) == 2:
                    f.write(f"v {v[0]} {v[1]} 0.0\n")  # Add z=0 for 2D vertices
                else:
                    f.write(f"v {v[0]} {v[1]} {v[2]}\n")

            # Write faces (convert 0-based to 1-based indices)
            for face in self.LCV:
                indices_str = ' '.join(str(i + 1) for i in face)
                f.write(f"f {indices_str}\n")

      



    def writeVTK_nolib(self, filename):
        """
        Write a VTU file (UnstructuredGrid)."""

        mesh = self.mesh
        vertices, connectivity = mesh
        tags = self.boundary_tags

        vertices = [v + (0.0,) for v in vertices]
        # Determine number of points
        num_vertices = len(vertices)

        # Convert connectivity (list of lists) to a consistent format internally
        # without changing the external data model.
        polygon_cells = [np.array(cell, dtype=int) for cell in connectivity]


        # Function to determine VTK cell type based on number of vertices in a cell
        # Add more mappings if you have other cell types.
        def cell_type_from_length(n):
            if n == 2:
                return 3  # VTK_LINE
            elif n == 3:
                return 5  # VTK_TRIANGLE
            elif n == 4:
                return 9  # VTK_QUAD
            else:
                raise ValueError(f"No VTK cell type defined for {n}-node cells.")

        # Process main polygonal cells
        # For these cells, we assign boundary_id=0 as a default.
        # Flatten their connectivity

        # Process main polygonal cells
        polygon_cell_lengths = [len(cell) for cell in polygon_cells]
        polygon_connectivity_flat = np.concatenate([cell for cell in polygon_cells]) if polygon_cells else np.array([], dtype=int)
        polygon_cell_types = np.array([cell_type_from_length(l) for l in polygon_cell_lengths], dtype=np.uint8)
        polygon_boundary_ids = np.zeros(len(polygon_cells), dtype=np.int32)  # default boundary_id=0

        # Process boundary edges (line cells)
        # Assign each boundary name a unique ID starting from 1
        boundary_names = list(tags.keys())
        boundary_id_map = {name: i+1 for i, name in enumerate(boundary_names)}

        # Flatten boundary edges into a single connectivity array
        # Each edge is a 2-vertex line cell
        boundary_edges = []
        boundary_edge_lengths = []
        boundary_edge_types = []
        boundary_edge_ids = []

        for bname, edges in tags.items():
            for edge in edges:
                edge = np.array(edge, dtype=int)  # ensure numpy array
                if len(edge) != 2:
                    raise ValueError("Boundary edges must have exactly 2 vertices.")
                boundary_edges.append(edge)
                boundary_edge_lengths.append(2)
                boundary_edge_types.append(cell_type_from_length(2))
                boundary_edge_ids.append(boundary_id_map[bname])

        if len(boundary_edges) > 0:
            boundary_connectivity_flat = np.concatenate(boundary_edges)
            boundary_cell_types = np.array(boundary_edge_types, dtype=np.uint8)
            boundary_ids_array = np.array(boundary_edge_ids, dtype=np.int32)
        else:
            boundary_connectivity_flat = np.array([], dtype=int)
            boundary_cell_types = np.array([], dtype=np.uint8)
            boundary_ids_array = np.array([], dtype=np.int32)

        # Combine polygonal cells and boundary line cells
        all_connectivity_flat = np.concatenate([polygon_connectivity_flat, boundary_connectivity_flat])
        all_cell_types = np.concatenate([polygon_cell_types, boundary_cell_types])
        all_boundary_ids = np.concatenate([polygon_boundary_ids, boundary_ids_array])
        all_cell_lengths = polygon_cell_lengths + boundary_edge_lengths

        num_cells = len(all_cell_lengths)
        offsets = np.cumsum(all_cell_lengths)

        # Write VTU file in ASCII format
        with open(filename, "w") as f:
            f.write('<?xml version="1.0"?>\n')
            f.write('<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n')
            f.write('  <UnstructuredGrid>\n')
            f.write(f'    <Piece NumberOfPoints="{num_vertices}" NumberOfCells="{num_cells}">\n')

            # Cell Data: boundary_ids
            f.write('      <CellData Scalars="BoundaryID">\n')
            f.write('        <DataArray type="Int32" Name="BoundaryID" format="ascii">\n')
            f.write('          ' + ' '.join(map(str, all_boundary_ids)) + '\n')
            f.write('        </DataArray>\n')
            f.write('      </CellData>\n')

            # vertices
            f.write('      <Points>\n')
            f.write('        <DataArray type="Float32" NumberOfComponents="3" format="ascii">\n')
            for p in vertices:
                f.write(f'          {p[0]} {p[1]} {p[2]}\n')
            f.write('        </DataArray>\n')
            f.write('      </Points>\n')

            # Cells
            f.write('      <Cells>\n')
            # connectivity
            f.write('        <DataArray type="Int32" Name="connectivity" format="ascii">\n')
            f.write('          ' + ' '.join(map(str, all_connectivity_flat)) + '\n')
            f.write('        </DataArray>\n')

            # offsets
            f.write('        <DataArray type="Int32" Name="offsets" format="ascii">\n')
            f.write('          ' + ' '.join(map(str, offsets)) + '\n')
            f.write('        </DataArray>\n')

            # types
            f.write('        <DataArray type="UInt8" Name="types" format="ascii">\n')
            f.write('          ' + ' '.join(map(str, all_cell_types)) + '\n')
            f.write('        </DataArray>\n')

            f.write('      </Cells>\n')

            f.write('    </Piece>\n')
            f.write('  </UnstructuredGrid>\n')
            f.write('</VTKFile>\n')

        basename = os.path.basename(filename)
        logger.info('VTK type mesh saved as {}'.
                    format(basename))
    def writeVTK_legacy(self, filename):
        """
        Write a legacy VTK (.vtk) UnstructuredGrid file (ASCII).
        - Main polygon cells get BoundaryID=0
        - Boundary edges (2-node line cells) are appended with BoundaryID>0
        """
        import os
        import numpy as np

        # data
        vertices, connectivity = self.mesh
        tags = getattr(self, "boundary_tags", {}) or {}

        # 3D points (z=0)
        pts = [(float(x), float(y), 0.0) if len(p) == 2 else tuple(map(float, p))
            for p in vertices for x, y in [p[:2]]]

        # polygons (tri/quad) from connectivity
        poly_cells = [np.asarray(c, dtype=np.int32) for c in connectivity]

        # boundary edges (each must be length-2 indices)
        boundary_names = list(tags.keys())
        boundary_id_map = {name: i + 1 for i, name in enumerate(boundary_names)}

        edge_cells = []
        edge_ids = []
        for bname, edges in tags.items():
            for e in edges:
                e = np.asarray(e, dtype=np.int32)
                if e.size != 2:
                    raise ValueError("Boundary edges must have exactly 2 vertices.")
                edge_cells.append(e)
                edge_ids.append(boundary_id_map[bname])

        # VTK cell types (legacy numeric codes)
        VTK_LINE = 3
        VTK_TRI  = 5
        VTK_QUAD = 9
        # If you want to support generic polygons (n>=3), you can use VTK_POLYGON=7.

        def vtk_type(cell_len: int) -> int:
            if cell_len == 2:
                return VTK_LINE
            elif cell_len == 3:
                return VTK_TRI
            elif cell_len == 4:
                return VTK_QUAD
            else:
                raise ValueError(f"No VTK legacy type defined for {cell_len}-node cells.")

        # Build final cell lists (polygons first, then edges)
        all_cells = []
        all_types = []
        boundary_ids = []

        # polygons → BoundaryID = 0
        for c in poly_cells:
            all_cells.append(c)
            all_types.append(vtk_type(len(c)))
            boundary_ids.append(0)

        # edges → BoundaryID > 0
        for c, bid in zip(edge_cells, edge_ids):
            all_cells.append(c)
            all_types.append(vtk_type(len(c)))
            boundary_ids.append(int(bid))

        num_pts = len(pts)
        num_cells = len(all_cells)

        # CELLS section wants: numberOfCells  totalIntCount
        # totalIntCount = sum(1 + len(cell) for each cell)
        cell_size_list = [len(c) for c in all_cells]
        total_ints = sum(1 + n for n in cell_size_list)

        # write file
        if not filename.endswith(".vtk"):
            filename = filename + ".vtk"

        with open(filename, "w") as f:
            # header
            f.write("# vtk DataFile Version 2.0\n")
            f.write("mesh\n")
            f.write("ASCII\n")
            f.write("DATASET UNSTRUCTURED_GRID\n")

            # POINTS
            f.write(f"POINTS {num_pts} float\n")
            for x, y, z in pts:
                f.write(f"{x} {y} {z}\n")

            # CELLS
            f.write(f"CELLS {num_cells} {total_ints}\n")
            for c in all_cells:
                f.write(str(len(c)))
                f.write(" ")
                f.write(" ".join(str(int(i)) for i in c))
                f.write("\n")

            # CELL_TYPES
            f.write(f"CELL_TYPES {num_cells}\n")
            for t in all_types:
                f.write(f"{t}\n")

            # CELL_DATA with BoundaryID
            f.write(f"CELL_DATA {num_cells}\n")
            f.write("SCALARS BoundaryID int 1\n")
            f.write("LOOKUP_TABLE default\n")
            for bid in boundary_ids:
                f.write(f"{bid}\n")

        # log
        try:
            basename = os.path.basename(filename)
            logger.info(f"Legacy VTK mesh saved as {basename}")
        except Exception:
            print(f"[info] Legacy VTK mesh saved as {filename}")

    def extrudeTo3d(self, thickness = 0.01, zmin_name = "front", zmax_name = "back"):
        V2, quads2d = self.mesh
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
        btags = getattr(self, "boundary_tags", {}) or {}
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

        self.cell_data = {"gmsh:physical": phys, "gmsh:geometrical": geom}
        self.cells = cells
        self.points3d = points3d
        self.field_data = field_data

    def write_gmsh22_ascii(self, msh_name):
        """
        Write mesh to Gmsh v2.2 ASCII (.msh) similar to meshio.write(file_format='gmsh22', binary=False).

        Parameters
        ----------
        msh_name : str
            Output file path, e.g. "mesh.msh".
        points3d : (N,3) array_like
            Node coordinates.
        cells : dict[str, ndarray] | list[tuple[str, ndarray]]
            Cell connectivity by type. Each array is shape (M, k) with 0-based node indices.
            Accepts either:
                - {'triangle': tri_conn, 'quad': quad_conn, ...}
                - [('triangle', tri_conn), ('quad', quad_conn), ...]
        cell_data : dict[str, list[np.ndarray]] | None
            Per-block data arrays in the same order as the `cells` blocks (meshio-like).
            Expected keys for Gmsh tags:
                - "gmsh:physical"   -> list of 1D arrays (ints)
                - "gmsh:geometrical"-> list of 1D arrays (ints)
            If not provided, elements will have zero tags.
        field_data : dict[str, array_like(int)]
            Mapping of physical name to (tag, dim) or (dim, tag). We try both orders.
            Example: {"inlet": [1, 1], "fluid": [2, 2]}
        """
        # Normalize points
        pts = np.asarray(self.points3d, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError("points3d must be (N,3)")

        # Normalize cells to a list of (type, data)
        if isinstance(self.cells, dict):
            cell_blocks = [(ct, np.asarray(conn, dtype=int)) for ct, conn in self.cells.items()]
        else:
            cell_blocks = [(ct, np.asarray(conn, dtype=int)) for ct, conn in self.cells]

        # Map meshio cell types to Gmsh 2.2 element type IDs
        gmsh_type_map = {
            "vertex": 15,         # 1-node point
            "line": 1,            # 2-node line
            "line2": 8,           # 3-node second-order line
            "triangle": 2,        # 3-node triangle
            "triangle6": 9,       # 6-node second-order triangle
            "quad": 3,            # 4-node quadrangle
            "quad8": 16,          # 8-node second-order quad
            "tetra": 4,           # 4-node tetra
            "tetra10": 11,        # 10-node second-order tetra
            "hexahedron": 5,      # 8-node hexahedron
            "hexahedron20": 17,   # 20-node second-order hex
            "wedge": 6,           # 6-node prism
            "pyramid": 7,         # 5-node pyramid
        }

        # Prepare cell tags from cell_data (optional)
        # meshio layout: cell_data[name] -> list aligned with cell_blocks
        phys_lists = None
        geom_lists = None
        if self.cell_data:
            phys_lists = self.cell_data.get("gmsh:physical")
            geom_lists = self.cell_data.get("gmsh:geometrical")
            # Sanity: if provided, lengths should match number of blocks
            for lst in (phys_lists, geom_lists):
                if lst is not None and len(lst) != len(cell_blocks):
                    raise ValueError("cell_data lists must align with number of cell blocks")

        # Field data → $PhysicalNames
        # meshio typically stores as name -> [tag, dim]; sometimes [dim, tag].

        with open(msh_name, "w") as f:
            # --- MeshFormat ---
            f.write("$MeshFormat\n")
            f.write("2.2 0 8\n")   # version 2.2, ascii=0, data-size=8
            f.write("$EndMeshFormat\n")

            # --- PhysicalNames (optional) ---
            if self.field_data:
                items = list(self.field_data.items())
                f.write("$PhysicalNames\n")
                f.write(f"{len(items)}\n")
                for name, val in items:
                    dim, tag = val
                    # name must be quoted
                    f.write(f"{dim} {tag} \"{name}\"\n")
                f.write("$EndPhysicalNames\n")

            # --- Nodes ---
            f.write("$Nodes\n")
            f.write(f"{pts.shape[0]}\n")
            # Node IDs are 1-based
            for i, (x, y, z) in enumerate(pts, start=1):
                # Use repr-like float to preserve precision without scientific notation issues
                f.write(f"{i} {x:.16g} {y:.16g} {z:.16g}\n")
            f.write("$EndNodes\n")

            # --- Elements ---
            # Count total elements
            total_elems = sum(b[1].shape[0] for b in cell_blocks)
            f.write("$Elements\n")
            f.write(f"{total_elems}\n")

            eid = 1  # global element id (1-based)
            for bi, (ctype, conn) in enumerate(cell_blocks):
                if ctype not in gmsh_type_map:
                    raise ValueError(f"Unsupported cell type for Gmsh 2.2: {ctype}")
                etype = gmsh_type_map[ctype]

                # Optional per-element tags
                phys = None
                geom = None
                if phys_lists is not None and phys_lists[bi] is not None:
                    phys = np.asarray(phys_lists[bi], dtype=int).ravel()
                    if phys.size == 1:
                        phys = phys * np.ones(conn.shape[0], dtype=int)
                if geom_lists is not None and geom_lists[bi] is not None:
                    geom = np.asarray(geom_lists[bi], dtype=int).ravel()
                    if geom.size == 1:
                        geom = geom * np.ones(conn.shape[0], dtype=int)

                for ei in range(conn.shape[0]):
                    nodes = conn[ei] + 1  # 1-based node indices
                    tags = []

                    # Gmsh expects: number-of-tags, then tags (physical, geometrical, ...).
                    # We'll write at most physical+geometrical if present.
                    if phys is not None:
                        tags.append(int(phys[ei]))
                    if geom is not None:
                        tags.append(int(geom[ei]))

                    f.write(f"{eid} {etype} {len(tags)}")
                    if tags:
                        f.write(" " + " ".join(str(t) for t in tags))
                    f.write(" " + " ".join(str(int(n)) for n in nodes) + "\n")
                    eid += 1

            f.write("$EndElements\n")
