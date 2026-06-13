from airfoil import Airfoil
from BlockMesh import BlockMesh
import numpy as np
from LineDistribution import LineDistribution
import gmsh


class Assemble:
    """Assemble multi-block geometry and generate a structured-unstructured hybrid 2D 
    triangular mesh.

    This assembler constructs:
    - a near-airfoil boundary-layer block by extruding the airfoil contour along
      surface normals with geometric growth,
    - an upper-side shock-box block extruded from a selected outer-layer segment,
    - a wake “tunnel” (trapezium-shaped) downstream of the trailing edge (optionally 
      with a curved TE-cap block),
    - an unstructured farfield mesh.

    """

    def __init__(self, config):
        """Initialize assembler.

        :param config: configuration dictionary (typically parsed from YAML)
        :type config: dict
        """
        self.blocks = list()
        self.unstructured = list()
        self.config = config
    def _gmsh_extract_2d_mesh(self):
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        P3 = np.asarray(node_coords, dtype=float).reshape(-1, 3)

        types, elem_tags, node_tags_elem = gmsh.model.mesh.getElements(2)

        TRI3_TYPE = 2
        QUAD4_TYPE = 3

        tri_tags = None
        quad_tags = None

        for etype, e_tags, conn in zip(types, elem_tags, node_tags_elem):
            if len(e_tags) == 0:
                continue

            if etype == TRI3_TYPE:
                n_elems = len(e_tags)
                nper = len(conn) // n_elems
                tri_tags = np.asarray(conn, dtype=np.int64).reshape(-1, nper)

            elif etype == QUAD4_TYPE:
                n_elems = len(e_tags)
                nper = len(conn) // n_elems
                quad_tags = np.asarray(conn, dtype=np.int64).reshape(-1, nper)

        tag_to_idx = {int(tag): i for i, tag in enumerate(node_tags.tolist())}

        T_tri = None
        if tri_tags is not None:
            T_tri = np.vectorize(tag_to_idx.get, otypes=[np.int64])(tri_tags)

        T_quad = None
        if quad_tags is not None:
            T_quad = np.vectorize(tag_to_idx.get, otypes=[np.int64])(quad_tags)

        return {
            "P": P3,
            "tri_connectivity": T_tri,
            "quad_connectivity": T_quad,
        }


    def _clean_closed_polyline(self, pts, tol=1e-12):
        clean = []

        for p in pts:
            q = (float(p[0]), float(p[1]))

            if len(clean) == 0:
                clean.append(q)
            else:
                if np.linalg.norm(np.asarray(q) - np.asarray(clean[-1])) > tol:
                    clean.append(q)

        if np.linalg.norm(np.asarray(clean[0]) - np.asarray(clean[-1])) < tol:
            clean.pop()

        return clean


    def _add_gmsh_wire_from_points(self, pts):
        pts = self._clean_closed_polyline(pts)

        p_tags = [
            gmsh.model.occ.addPoint(float(x), float(y), 0.0)
            for x, y in pts
        ]

        l_tags = [
            gmsh.model.occ.addLine(p_tags[i], p_tags[(i + 1) % len(p_tags)])
            for i in range(len(p_tags))
        ]

        wire = gmsh.model.occ.addWire(l_tags)

        return p_tags, l_tags, wire


    def _mesh_wake_trapezium(
        self,
        poly_pts,
        x_left,
        x_right,
        lc_left,
        lc_right,
        fill_shape="tria",
    ):
        if fill_shape == "tria":
            algo = 6
        elif fill_shape == "quad":
            algo = 8
        else:
            raise ValueError(f"Invalid fill_shape '{fill_shape}'.")

        gmsh.initialize()

        try:
            gmsh.model.add("wake_trapezium")

            gmsh.option.setNumber("Mesh.Algorithm", algo)

            if fill_shape == "quad":
                gmsh.option.setNumber("Mesh.RecombineAll", 1)

            _, _, w = self._add_gmsh_wire_from_points(poly_pts)

            surf = gmsh.model.occ.addPlaneSurface([w])

            gmsh.model.occ.synchronize()

            if fill_shape == "quad":
                gmsh.model.mesh.setRecombine(2, surf)

            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

            if abs(x_right - x_left) < 1e-14:
                raise ValueError("Wake trapezium has zero x-extent.")

            f_grad = gmsh.model.mesh.field.add("MathEval")

            expr = (
                f"{lc_left} + ({lc_right} - {lc_left}) * "
                f"((x - {x_left}) / ({x_right - x_left}))"
            )

            gmsh.model.mesh.field.setString(f_grad, "F", expr)
            gmsh.model.mesh.field.setAsBackgroundMesh(f_grad)

            gmsh.model.mesh.generate(2)

            return self._gmsh_extract_2d_mesh()

        finally:
            gmsh.finalize()


   
    def _mesh_farfield_with_airfoil_circle_threshold_grading(
        self,
        outer_airfoil_line,
        inner_airfoil_line,
        circle_center=(0.5, 0.0),
        circle_radius=3.0,
        lc_inner=0.03,
        lc_circle=0.12,
        lc_farfield_start=None,
        lc_outer=3.2,
        circle_distmax=2.0,
        outer_distmax=20.0,
        fill_shape="tria",
        curve_detect_tol=None,
    ):
        """
        Farfield mesh with a circular refinement region around the airfoil only.

        Important:
        The circle does NOT need to contain the whole inner_airfoil_line.
        It only needs to overlap the actual farfield region.

        Geometry:
            1. Create farfield annulus:
                outer_airfoil_line with inner_airfoil_line as hole

            2. Create circular disk around the airfoil

            3. Fragment farfield annulus with the disk

            4. Keep only surfaces that belong to the original farfield annulus

            5. Classify:
                circle_surfaces = farfield pieces inside circular disk
                outer_surfaces  = remaining farfield pieces outside disk

            6. Apply Gmsh Distance + Threshold grading:
                inside circle:
                    inner boundary -> lc_inner
                    away from inner boundary -> lc_circle

                outside circle:
                    circular interface -> lc_circle
                    farfield -> lc_outer
        """

        if fill_shape == "tria":
            algo = 6
        elif fill_shape == "quad":
            algo = 8
        else:
            raise ValueError(
                f"Invalid fill_shape '{fill_shape}'. Expected 'tria' or 'quad'."
            )

        cx, cy = circle_center
        
        if lc_farfield_start is None:
            lc_farfield_start = lc_circle

        # --------------------------------------------------
        # Helper functions
        # --------------------------------------------------

        def point_segment_distance(p, a, b):
            p = np.asarray(p, dtype=float)
            a = np.asarray(a, dtype=float)
            b = np.asarray(b, dtype=float)

            ab = b - a
            denom = np.dot(ab, ab)

            if denom < 1.0e-300:
                return np.linalg.norm(p - a)

            t = np.dot(p - a, ab) / denom
            t = max(0.0, min(1.0, t))

            q = a + t * ab

            return np.linalg.norm(p - q)

        def point_polyline_distance(p, polyline):
            pts = list(polyline)

            dmin = np.inf

            for i in range(len(pts)):
                a = pts[i]
                b = pts[(i + 1) % len(pts)]

                dmin = min(
                    dmin,
                    point_segment_distance(p, a, b),
                )

            return dmin

        def get_surface_boundary_curves(surface_tags):
            curves = set()

            for s in surface_tags:
                boundary = gmsh.model.getBoundary(
                    [(2, s)],
                    oriented=False,
                    recursive=False,
                )

                for dim, tag in boundary:
                    if dim == 1:
                        curves.add(abs(tag))

            return curves

        def set_threshold_sizes(field_id, size_min, size_max):
            """
            Some Gmsh versions use SizeMin/SizeMax,
            older scripts often use LcMin/LcMax.
            """
            try:
                gmsh.model.mesh.field.setNumber(
                    field_id,
                    "SizeMin",
                    float(size_min),
                )
                gmsh.model.mesh.field.setNumber(
                    field_id,
                    "SizeMax",
                    float(size_max),
                )
            except Exception:
                gmsh.model.mesh.field.setNumber(
                    field_id,
                    "LcMin",
                    float(size_min),
                )
                gmsh.model.mesh.field.setNumber(
                    field_id,
                    "LcMax",
                    float(size_max),
                )

        gmsh.initialize()

        try:
            gmsh.model.add("farfield_airfoil_circle_threshold_grading")

            gmsh.option.setNumber("Mesh.Algorithm", algo)

            if fill_shape == "quad":
                gmsh.option.setNumber("Mesh.RecombineAll", 1)

            # --------------------------------------------------
            # Outer farfield boundary
            # --------------------------------------------------
            _, _, w_out = self._add_gmsh_wire_from_points(
                outer_airfoil_line
            )

            # --------------------------------------------------
            # Inner boundary / hole
            # This can include shock-box, wake, structured outer edge, etc.
            # The circular disk does NOT need to contain all of this.
            # --------------------------------------------------
            _, _, w_in = self._add_gmsh_wire_from_points(
                inner_airfoil_line
            )

            # --------------------------------------------------
            # Full farfield annulus
            # --------------------------------------------------
            surf_farfield = gmsh.model.occ.addPlaneSurface(
                [w_out, w_in]
            )

            # --------------------------------------------------
            # Circular disk around airfoil
            # --------------------------------------------------
            target_circle_spacing = float(lc_circle)

            if target_circle_spacing <= 0.0:
                raise ValueError("lc_circle / circle_size must be positive.")

            # Chord-based spacing:
            # chord length = 2 R sin(dtheta / 2)
            # choose dtheta so chord length is approximately target_circle_spacing
            dtheta = 2.0 * np.arcsin(
                min(1.0, target_circle_spacing / (2.0 * circle_radius))
            )

            n_circle = int(np.ceil(2.0 * np.pi / dtheta))

            # Keep a minimum number of points so the circle does not become too polygonal
            n_circle = max(n_circle, 32)

            theta = np.linspace(
                0.0,
                2.0 * np.pi,
                n_circle,
                endpoint=False,
            )

            circle_pts = [
                (
                    cx + circle_radius * np.cos(t),
                    cy + circle_radius * np.sin(t),
                )
                for t in theta
            ]
            
            _, _, w_circle = self._add_gmsh_wire_from_points(
                circle_pts
            )

            surf_disk = gmsh.model.occ.addPlaneSurface(
                [w_circle]
            )

            gmsh.model.occ.synchronize()

            # --------------------------------------------------
            # Fragment farfield with circular disk.
            #
            # out_map[0] gives the surfaces created from surf_farfield.
            # out_map[1] gives the surfaces created from surf_disk.
            #
            # Their intersection gives the part of the farfield that lies
            # inside the circular disk.
            # --------------------------------------------------
            out_dimtags, out_map = gmsh.model.occ.fragment(
                [(2, surf_farfield)],
                [(2, surf_disk)],
                removeObject=True,
                removeTool=True,
            )

            gmsh.model.occ.synchronize()

            all_surfaces = {
                tag for dim, tag in gmsh.model.getEntities(2)
            }

            if len(out_map) < 2:
                raise RuntimeError(
                    "Gmsh fragment did not return the expected object/tool map."
                )

            farfield_from_map = {
                tag for dim, tag in out_map[0]
                if dim == 2 and tag in all_surfaces
            }

            disk_from_map = {
                tag for dim, tag in out_map[1]
                if dim == 2 and tag in all_surfaces
            }

            if len(farfield_from_map) == 0:
                raise RuntimeError(
                    "Could not identify surfaces originating from the farfield annulus."
                )

            # --------------------------------------------------
            # Remove surfaces that are not part of the original farfield.
            # This removes circular disk pieces that may lie inside the hole.
            # --------------------------------------------------
            remove_surfaces = sorted(
                all_surfaces - farfield_from_map
            )

            if remove_surfaces:
                gmsh.model.occ.remove(
                    [(2, s) for s in remove_surfaces],
                    recursive=True,
                )

            gmsh.model.occ.synchronize()

            existing_surfaces = {
                tag for dim, tag in gmsh.model.getEntities(2)
            }

            # --------------------------------------------------
            # Classify final surfaces
            # --------------------------------------------------
            circle_surfaces = sorted(
                farfield_from_map.intersection(disk_from_map).intersection(existing_surfaces)
            )

            outer_surfaces = sorted(
                farfield_from_map.difference(disk_from_map).intersection(existing_surfaces)
            )

            keep_surfaces = sorted(
                circle_surfaces + outer_surfaces
            )

            if len(circle_surfaces) == 0:
                raise ValueError(
                    "No farfield surface was found inside the circular refinement region.\n"
                    "This means the circular disk does not overlap the farfield annulus.\n"
                    "Most likely the circle is completely inside the structured/wake hole.\n"
                    "Increase circle_radius or move circle_center."
                )

            if len(outer_surfaces) == 0:
                raise ValueError(
                    "No outer farfield surface was found outside the circle.\n"
                    "Decrease circle_radius or enlarge the outer farfield."
                )

            if fill_shape == "quad":
                for s in keep_surfaces:
                    gmsh.model.mesh.setRecombine(2, s)

            # --------------------------------------------------
            # Detect circular interface curves
            #
            # These are the curves shared by:
            #     circle_surfaces and outer_surfaces
            # --------------------------------------------------
            curves_circle_region = get_surface_boundary_curves(
                circle_surfaces
            )

            curves_outer_region = get_surface_boundary_curves(
                outer_surfaces
            )

            circle_interface_curves = sorted(
                curves_circle_region.intersection(curves_outer_region)
            )

            if len(circle_interface_curves) == 0:
                raise ValueError(
                    "Could not detect circular interface curves after fragmentation."
                )

            # --------------------------------------------------
            # Detect inner boundary curves after fragmentation.
            #
            # These are the boundary curves close to the original inner_airfoil_line.
            # --------------------------------------------------
            all_kept_boundary_curves = get_surface_boundary_curves(
                keep_surfaces
            )

            inner_polygon = [
                (float(x), float(y)) for x, y in inner_airfoil_line
            ]

            all_pts = np.asarray(
                list(outer_airfoil_line) + list(inner_airfoil_line),
                dtype=float,
            )

            domain_size = np.linalg.norm(
                np.max(all_pts[:, :2], axis=0)
                - np.min(all_pts[:, :2], axis=0)
            )

            if curve_detect_tol is None:
                curve_detect_tol = max(
                    1.0e-8 * domain_size,
                    1.0e-10,
                )

            inner_boundary_curves = []

            for c in all_kept_boundary_curves:
                xcm, ycm, _ = gmsh.model.occ.getCenterOfMass(1, c)

                dist_to_inner = point_polyline_distance(
                    (xcm, ycm),
                    inner_polygon,
                )

                if dist_to_inner < curve_detect_tol:
                    inner_boundary_curves.append(c)

            if len(inner_boundary_curves) == 0:
                raise ValueError(
                    "Could not detect inner boundary curves after fragmentation.\n"
                    "Try increasing curve_detect_tol, for example curve_detect_tol=1e-6."
                )

            # --------------------------------------------------
            # Mesh-size field options
            # --------------------------------------------------
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

            # ==================================================
            # Field 1:
            # Grading inside circular region
            #
            # Distance from inner boundary:
            #     d = 0              -> lc_inner
            #     d = circle_distmax -> lc_circle
            # ==================================================
            f_dist_inner = gmsh.model.mesh.field.add("Distance")

            gmsh.model.mesh.field.setNumbers(
                f_dist_inner,
                "CurvesList",
                inner_boundary_curves,
            )

            gmsh.model.mesh.field.setNumber(
                f_dist_inner,
                "NumPointsPerCurve",
                100,
            )

            f_thr_circle = gmsh.model.mesh.field.add("Threshold")

            gmsh.model.mesh.field.setNumber(
                f_thr_circle,
                "InField",
                f_dist_inner,
            )

            set_threshold_sizes(
                f_thr_circle,
                lc_inner,
                lc_circle,
            )

            gmsh.model.mesh.field.setNumber(
                f_thr_circle,
                "DistMin",
                0.0,
            )

            gmsh.model.mesh.field.setNumber(
                f_thr_circle,
                "DistMax",
                float(circle_distmax),
            )

            f_thr_circle_r = gmsh.model.mesh.field.add("Restrict")

            gmsh.model.mesh.field.setNumber(
                f_thr_circle_r,
                "InField",
                f_thr_circle,
            )

            gmsh.model.mesh.field.setNumbers(
                f_thr_circle_r,
                "SurfacesList",
                circle_surfaces,
            )

            # ==================================================
            # Field 2:
            # Grading outside circular region
            #
            # Distance from circular interface:
            #     d = 0             -> lc_circle
            #     d = outer_distmax -> lc_outer
            # ==================================================
            # ==================================================
            # Field 2A:
            # Outer farfield grading from circular interface
            #
            # Distance from circular interface:
            #     d = 0             -> lc_farfield_start
            #     d = outer_distmax -> lc_outer
            # ==================================================
            f_dist_circle = gmsh.model.mesh.field.add("Distance")

            gmsh.model.mesh.field.setNumbers(
                f_dist_circle,
                "CurvesList",
                circle_interface_curves,
            )

            gmsh.model.mesh.field.setNumber(
                f_dist_circle,
                "NumPointsPerCurve",
                100,
            )

            f_thr_outer_circle = gmsh.model.mesh.field.add("Threshold")

            gmsh.model.mesh.field.setNumber(
                f_thr_outer_circle,
                "InField",
                f_dist_circle,
            )

            set_threshold_sizes(
                f_thr_outer_circle,
                lc_farfield_start,
                lc_outer,
            )

            gmsh.model.mesh.field.setNumber(
                f_thr_outer_circle,
                "DistMin",
                0.0,
            )

            gmsh.model.mesh.field.setNumber(
                f_thr_outer_circle,
                "DistMax",
                float(outer_distmax),
            )

            f_thr_outer_circle_r = gmsh.model.mesh.field.add("Restrict")

            gmsh.model.mesh.field.setNumber(
                f_thr_outer_circle_r,
                "InField",
                f_thr_outer_circle,
            )

            gmsh.model.mesh.field.setNumbers(
                f_thr_outer_circle_r,
                "SurfacesList",
                outer_surfaces,
            )


            # ==================================================
            # Field 2B:
            # Outer farfield grading from inner boundary
            #
            # This is the important part for the wake.
            #
            # Distance from wake/inner boundary:
            #     d = 0             -> lc_farfield_start
            #     d = outer_distmax -> lc_outer
            # ==================================================
            f_dist_inner_outer = gmsh.model.mesh.field.add("Distance")

            gmsh.model.mesh.field.setNumbers(
                f_dist_inner_outer,
                "CurvesList",
                inner_boundary_curves,
            )

            gmsh.model.mesh.field.setNumber(
                f_dist_inner_outer,
                "NumPointsPerCurve",
                100,
            )

            f_thr_outer_inner = gmsh.model.mesh.field.add("Threshold")

            gmsh.model.mesh.field.setNumber(
                f_thr_outer_inner,
                "InField",
                f_dist_inner_outer,
            )

            set_threshold_sizes(
                f_thr_outer_inner,
                lc_farfield_start,
                lc_outer,
            )

            gmsh.model.mesh.field.setNumber(
                f_thr_outer_inner,
                "DistMin",
                0.0,
            )

            gmsh.model.mesh.field.setNumber(
                f_thr_outer_inner,
                "DistMax",
                float(outer_distmax),
            )

            f_thr_outer_inner_r = gmsh.model.mesh.field.add("Restrict")

            gmsh.model.mesh.field.setNumber(
                f_thr_outer_inner_r,
                "InField",
                f_thr_outer_inner,
            )

            gmsh.model.mesh.field.setNumbers(
                f_thr_outer_inner_r,
                "SurfacesList",
                outer_surfaces,
            )


            # ==================================================
            # Final background mesh field
            # ==================================================
            f_min = gmsh.model.mesh.field.add("Min")

            gmsh.model.mesh.field.setNumbers(
                f_min,
                "FieldsList",
                [
                    f_thr_circle_r,
                    f_thr_outer_circle_r,
                    f_thr_outer_inner_r,
                ],
            )

            gmsh.model.mesh.field.setAsBackgroundMesh(f_min)

            # --------------------------------------------------
            # Generate mesh
            # --------------------------------------------------
            gmsh.model.mesh.generate(2)

            return self._gmsh_extract_2d_mesh()

        finally:
            gmsh.finalize()

    def assemble(self):

        # --------------------------
        # Distribute points on airfoil contour
        # --------------------------

        af_cfg = self.config.get("airfoil", {})

        if "contour_file" not in af_cfg:
            raise KeyError("Missing required key 'contour_file' in Airfoil config.")
        kwargs_airfoil = {
            "filename": af_cfg.get("contour_file", None),
            "k": af_cfg.get("spline_degree", 2),
        }
        aero = Airfoil.from_contour_file(**kwargs_airfoil)

        bd = self.config.get("airfoil_boundary", {})
        n_points = bd.get("n_points", 500)

        dist_kwargs = {
            "n_points_te": bd.get("n_points_te"),
            "weight_upper": bd.get("weight_upper"),
            "weight_curvature": bd.get("weight_curvature"),
            "weight_te": bd.get("weight_te"),
            "fraction_te": bd.get("fraction_te"),
            "max_size_ratio": bd.get("max_size_ratio"),
            "n_points_high_res": bd.get("n_points_high_res"),
            "max_relax_iter": bd.get("max_relax_iter"),
        }

        xp, yp, _, _, surf_normals = aero.distribute_points(n_points, **dist_kwargs)

        x_chord_left = np.min(xp)
        chord_length = np.max(xp) - np.min(xp)
        # --------------------------
        # Boundary layer mesh by extrusion
        # --------------------------

        mesh = BlockMesh()
        ex = self.config.get("airfoil_extrusion", {})
        ex_kwargs = {
            "cell_thickness": ex.get("cell_thickness", 1e-4),
            "growth": ex.get("growth", 1.05),
            "extrusion_distance": ex.get("extrusion_distance", 0.05),
        }
        airfoil_bd = [(x, y) for x, y in zip(xp, yp)]
        mesh.extrudeLine_cell_thickness(airfoil_bd, surf_normals, **ex_kwargs)
        self.blocks.append(mesh)

        # --------------------------
        # Shock box structured mesh
        # --------------------------

        shock_box = BlockMesh()
        mesh_outer = mesh.getLine(number=-1, direction="u").copy()
        boundaries_lower = []
        surf_normals_lower = []

        ex_shock = self.config.get("shock_box", {})
        length = ex_shock.get("box_height", 0.6)*chord_length
        growth = ex_shock.get("growth", 1.05)
        xMin_shock = x_chord_left + ex_shock.get("xmin", 0.05)*chord_length
        xMax_shock = x_chord_left + ex_shock.get("xmax", 0.7)*chord_length

        for i, (x, y) in enumerate(mesh_outer):
            if xMin_shock < x < xMax_shock and y > 0:
                boundaries_lower.append((x, y))
                surf_normals_lower.append(surf_normals[i])

        boundaries_rest = [
            (x, y)
            for (x, y) in mesh_outer
            if not (xMin_shock < x < xMax_shock and y > 0)
        ]

        boundaries_right_upper = [
            (x, y) for (x, y) in boundaries_rest if x > xMax_shock and y > 0
        ]

        boundaries_rest_outer = [
            (x, y) for (x, y) in boundaries_rest if not (x > xMax_shock and y > 0)
        ]

        cell_thick = np.linalg.norm(
            np.array(mesh.getLine(number=-1, direction="u").copy()[0])
            - np.array(mesh.getLine(number=-2, direction="u").copy()[0])
        )

        dy = []
        s = 0.0
        h = float(cell_thick)

        while s + h <= length + 1e-12:
            dy.append(h)
            s += h
            h *= growth

        ex_kwargs = {
            "cell_thickness": cell_thick,
            "growth": growth,
            "extrusion_distance": length,
        }
        shock_box.extrudeLine_cell_thickness(
            boundaries_lower, surf_normals_lower, **ex_kwargs
        )
        self.blocks.append(shock_box)
        shock_right = shock_box.getLine(number=-1, direction="v")
        shock_left = shock_box.getLine(number=0, direction="v")
        
        shock_upper = shock_box.getLine(number=-1, direction="u")

        ex_tunnel = self.config.get("wake_tunnel", {})
        curve_bool = ex_tunnel.get("make_curve", True)

        line_airfoil_bound = mesh.getLine(number=0, direction="u")
        p0 = np.array(line_airfoil_bound[0])
        p1 = np.array(line_airfoil_bound[1])
        p2 = np.array(line_airfoil_bound[-1])

        n_po = ex_tunnel.get("n_points", 70)
        p_te_up = p2
        p_te_down = p0

        len_te = np.linalg.norm(np.array(p_te_up) - np.array(p_te_down))

        # --------------------------
        # Wake tunnel / TE-cap
        # --------------------------

        if curve_bool:

            te_circular_outer = BlockMesh()

            fraction_te_struct = ex_tunnel.get("fraction_structured", 0.5)    
            te_upper_line = mesh.getLine(number=-1, direction="v").copy()
            len_upper_te = np.linalg.norm(np.array(te_upper_line[0])-np.array(te_upper_line[-1]))            
            y_upper = p_te_up[1] + fraction_te_struct * len_upper_te

            
            boundaries_upper = [p for p in te_upper_line if p[1] <= y_upper]
            te_lower_line = mesh.getLine(number=0, direction="v").copy()
            boundaries_lower = te_lower_line[: len(boundaries_upper)]

            if fraction_te_struct == 1.0:
                te_line_remaining_upper = []
                te_line_remaining_lower = [boundaries_lower[-1]]
            else:

                te_line_remaining_upper = [p for p in te_upper_line if p[1] > y_upper]
                te_line_remaining_upper.insert(0, boundaries_upper[-1])

                te_line_remaining_lower = te_lower_line[len(boundaries_upper) :]
                te_line_remaining_lower.insert(0, boundaries_lower[-1])
            
            boundaries_left = LineDistribution.arc_like_spline_from_2pts_2tangents(
                p2,
                line_airfoil_bound[1],
                line_airfoil_bound[0],
                p0,
                line_airfoil_bound[-1],
                line_airfoil_bound[-2],
                n_points=n_po + 1,
                handle_scale=1.4,
            )

            pc1 = np.array(boundaries_upper[-1], float)
            pc2 = np.array(boundaries_lower[-1], float)
            T = np.array([1.0, 0.0])
            if np.allclose(pc1, pc2):
                raise ValueError("Points must be distinct.")
            M = 0.5 * (pc1 + pc2)
            v = pc2 - pc1
            n = np.array([-v[1], v[0]], float)
            n /= np.linalg.norm(n)
            s = n.dot(T - M)  # closest point on the perpendicular bisector to T
            centre_arc_te_skeleton = M + s * n  # center (h,k) near (1,0)
            R_arc_te_skeleton = np.linalg.norm(centre_arc_te_skeleton - pc1)  # radius

            normal_ref = LineDistribution.dist_arc_with_normals_spline(
                p1=boundaries_upper[-1],
                p2=boundaries_lower[-1],
                ref_polyline=boundaries_left,
                center=centre_arc_te_skeleton,
                radius=R_arc_te_skeleton,
            )
            boundaries_right = LineDistribution.dist_arc_with_ref_spacing(
                p1=boundaries_upper[-1],
                p2=boundaries_lower[-1],
                center=centre_arc_te_skeleton,
                radius=R_arc_te_skeleton,
                ref_polyline=normal_ref,
                direction="cw",
                alpha=0.0,
            )

            boundaries_left.reverse()
            boundaries_right.reverse()
            boundary = [
                boundaries_upper,
                boundaries_lower,
                boundaries_left,
                boundaries_right,
            ]

            te_circular_outer.transfinite(boundary=boundary)
            self.blocks.append(te_circular_outer)
            te_line = te_circular_outer.getLine(number=-1, direction="v").copy()
        else:
            p_te_up_arr = np.array(p_te_up, dtype=float)
            p_te_down_arr = np.array(p_te_down, dtype=float)

            t = np.linspace(0.0, 1.0, n_po)

            # down point first
            te_line = [
                tuple(p_te_down_arr + ti * (p_te_up_arr - p_te_down_arr)) for ti in t
            ]
            te_line_remaining_lower = mesh.getLine(number=0, direction="v").copy()
            te_line_remaining_upper = mesh.getLine(number=-1, direction="v").copy()

        # --------------------------
        # Unstructured mesh using Gmsh
        # --------------------------

        angle_deg = ex_tunnel.get("angle", 5.0)  # magnitude of slope angle (               deg)
        Lx = ex_tunnel.get(
            "length", 5.0
        )*chord_length  # horizontal length to the right (sets the vertical right boundary x)

        ex_mesh = self.config.get("tria_mesh_settings", {})
        fill_shape = ex_mesh.get("fill_shape", "tria")
        ds = ex_mesh.get("wake_size", 0.01)  # spacing
        lc_block1 = ex_mesh.get("wake_size", 0.01)
        ex_farfield = ex_mesh.get("farfield", "")
        lc_inner = ex_farfield.get("min_size", 0.3)
        lc_outer = ex_farfield.get("max_size", 3.2)
        d1 = ex_farfield.get("grading", 0.3)

        airfoil_line = te_line_remaining_lower.copy()
        airfoil_line.reverse()
        airfoil_line1 = te_line_remaining_upper.copy()
        airfoil_line.extend(te_line[1:])

        airfoil_line.extend(airfoil_line1[1:])

        # Ensure left boundary goes bottom -> top
        if airfoil_line[0][1] > airfoil_line[-1][1]:
            airfoil_line = airfoil_line[::-1]

        # Left boundary endpoints
        xL_bot, yL_bot = airfoil_line[0]
        xL_top, yL_top = airfoil_line[-1]

        ang = np.deg2rad(angle_deg)

        # Common right-boundary x (vertical right side)
        xR = xL_top + Lx

        yR_top = yL_top + np.tan(+ang) * (xR - xL_top)
        yR_bot = yL_bot + np.tan(-ang) * (xR - xL_bot)

        lc_left = ex_mesh.get("wake_size_left", 0.005)   # small cells near left
        lc_right = ex_mesh.get("wake_size_right", 0.03)  # larger cells near right
        x_left = min(xL_bot, xL_top)
        x_right = xR
        yR_bot = yL_bot + np.tan(-ang) * (xR - xL_bot)

        def wake_size_fun(x):
            return LineDistribution.size_x_linear(x, x_left, x_right, lc_left, lc_right)

        pL_top = (xL_top, yL_top)
        pR_top = (xR, yR_top)
        pL_bot = (xL_bot, yL_bot)
        pR_bot = (xR, yR_bot)

        # Check that the lower wake boundary does not intersect the TE circular arc.
        if curve_bool:
            x0, y0 = pL_bot
            cx_c, cy_c = centre_arc_te_skeleton
            R_c = float(R_arc_te_skeleton)
            A = cx_c - x0
            B = cy_c - y0
            AB = np.sqrt(A**2 + B**2)

            dist_to_line = abs(A * np.sin(ang) + B * np.cos(ang))
            proj_on_ray  = A * np.cos(ang) - B * np.sin(ang)

            if dist_to_line < R_c - 1e-10 and proj_on_ray > 0.0:
                if AB < R_c - 1e-10:
                    raise ValueError(
                        "pL_bot lies inside the trailing-edge circular arc region. "
                        "Check 'fraction_structured'."
                    )
                # Solve |A*sin(θ) + B*cos(θ)| = R_c analytically.
                # Equivalently: sqrt(A²+B²)*|sin(θ + φ)| = R_c, φ = arctan2(B, A)
                phi   = np.arctan2(B, A)
                alpha = np.arcsin(np.clip(R_c / AB, 0.0, 1.0))
                candidates = [
                    alpha - phi,
                    np.pi - alpha - phi,
                    -alpha - phi,
                    np.pi + alpha - phi,
                ]
                # Keep candidates that are positive and whose intersection lies ahead
                valid = [
                    c for c in candidates
                    if c > 1e-6
                    and A * np.cos(c) - B * np.sin(c) > 0.0
                ]
                if valid:
                    ang_min_deg = np.rad2deg(min(valid))
                else:
                    ang_min_deg = np.rad2deg(min(abs(c) for c in candidates))

                raise ValueError(
                    f"The wake trapezium lower boundary intersects the trailing-edge "
                    f"circular arc (TE arc radius ≈ {R_c:.4f}). "
                    f"Increase 'angle' in 'wake_tunnel' to at least "
                    f"{ang_min_deg:.1f} degrees "
                    f"(current value: {angle_deg:.1f} degrees)."
                )

        # top side: min size on left -> max size on right
        upper_line = LineDistribution.graded_straight_segment(
            pL_top, pR_top, wake_size_fun, include_start=False, include_end=True
        )

        # bottom side: min size on left -> max size on right
        lower_line = LineDistribution.graded_straight_segment(
            pL_bot, pR_bot, wake_size_fun, include_start=False, include_end=True
        )

        # right side: x = xR, so the x-based size law gives constant lc_right
        right_line = LineDistribution.graded_straight_segment(
            pR_bot, pR_top, wake_size_fun, include_start=False, include_end=False
        )
        poly_pts = airfoil_line + upper_line + right_line[::-1] + lower_line[::-1]
        inner_airfoil_line = boundaries_rest_outer.copy()

        inner_airfoil_line.extend(shock_left[:])

        inner_airfoil_line.extend(shock_upper[1:])

        shock_right.reverse()
        
        inner_airfoil_line.extend(shock_right[1:])
        inner_airfoil_line.extend(boundaries_right_upper)
        inner_airfoil_line.extend(upper_line)
        inner_airfoil_line.extend(right_line[::-1])
        inner_airfoil_line.extend(lower_line[::-1])
        ex_fardim = self.config.get("farfield", {})
        L_farfield = ex_fardim.get("length", 100)
        R_farfield = ex_fardim.get("radius", 50)
        x_airfoil_rb, y_airfoil_rb = max(
            airfoil_bd,
            key=lambda p: (p[0], -p[1])
        )
        bottom_right_farfield = (
            x_airfoil_rb + L_farfield,
            y_airfoil_rb - R_farfield,
        )

        L_tot = np.pi * R_farfield + 2 * R_farfield + 2 * L_farfield
        n_outer = int(np.ceil(L_tot / lc_outer))
        outer_airfoil_line = LineDistribution.closed_left_U(
            bottom_right=bottom_right_farfield,
            vert_len=2*R_farfield,
            horiz_len=L_farfield,
            n_segments=n_outer,
        )[1:]


        if fill_shape == "tria":
            algo = 6
        elif fill_shape == "quad":
            algo = 8  # Frontal-Delaunay for quads
        else:
            raise ValueError(f"Invalid fill_shape '{fill_shape}'. Expected 'tria' or 'quad'.")
        
        # --------------------------
        # Mesh 1: wake trapezium only
        # --------------------------

        wake_mesh = self._mesh_wake_trapezium(
            poly_pts=poly_pts,
            x_left=x_left,
            x_right=x_right,
            lc_left=lc_left,
            lc_right=lc_right,
            fill_shape=fill_shape,
        )

        self.unstructured.append(wake_mesh)


        # --------------------------
        # Mesh 2: farfield only
        # with circular refinement
        # --------------------------

        ff_cfg = ex_mesh.get("farfield", {})

        farfield_mesh = self._mesh_farfield_with_airfoil_circle_threshold_grading(
            outer_airfoil_line=outer_airfoil_line,
            inner_airfoil_line=inner_airfoil_line,

            circle_center=tuple(ff_cfg.get("circle_center", [0.5, 0.0])),
            circle_radius=ff_cfg.get("circle_radius", 3.0),
            lc_inner=ff_cfg.get("inner_size", 0.03),
            lc_circle=ff_cfg.get("circle_size", 0.12),
            lc_farfield_start=ff_cfg.get(
                "farfield_start_size",
                ff_cfg.get("circle_size", 0.12),
            ),
            lc_outer=ff_cfg.get("max_size", 3.2),

            circle_distmax=ff_cfg.get("circle_distmax", 1.5),
            outer_distmax=ff_cfg.get("outer_distmax", 20.0),
            fill_shape=fill_shape,
        )

        self.unstructured.append(farfield_mesh)