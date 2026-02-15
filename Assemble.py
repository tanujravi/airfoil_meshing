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
        self.trias = list()
        self.config = config

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
        length = ex_shock.get("box_height", 0.6)
        growth = ex_shock.get("growth", 1.05)
        xMin_shock = ex_shock.get("xmin", 0.05)
        xMax_shock = ex_shock.get("xmax", 0.7)

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

            y_upper = p_te_up[1] + 0.25 * len_te

            te_upper_line = mesh.getLine(number=-1, direction="v").copy()
            boundaries_upper = [p for p in te_upper_line if p[1] <= y_upper]

            te_line_remaining_upper = [p for p in te_upper_line if p[1] > y_upper]
            te_line_remaining_upper.insert(0, boundaries_upper[-1])

            te_lower_line = mesh.getLine(number=0, direction="v").copy()
            boundaries_lower = te_lower_line[: len(boundaries_upper)]

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
                alpha=0.9,
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
        def simplify_polyline_by_min_edge(points, h_min):
            """
            Remove intermediate points whose adjacent segments are too small to resolve.

            Rule: if ||A-B|| + ||B-C|| < h_min, drop B.
            Keeps endpoints always.

            :param points: polyline points [(x,y), ...]
            :type points: list[tuple[float,float]]
            :param h_min: target minimum mesh edge length
            :type h_min: float
            :return: simplified polyline
            :rtype: list[tuple[float,float]]
            """
            pts = [np.array(p, float) for p in points]
            if len(pts) <= 2:
                return points

            keep = [pts[0]]
            i = 1
            while i < len(pts) - 1:
                A = keep[-1]
                B = pts[i]
                C = pts[i + 1]
                if np.linalg.norm(B - A) + np.linalg.norm(C - B) < h_min:
                    # Drop B (do not advance keep), just skip it
                    i += 1
                    continue
                keep.append(B)
                i += 1

            keep.append(pts[-1])
            return [tuple(p) for p in keep]

        angle_deg = ex_tunnel.get("angle", 5.0)  # magnitude of slope angle (deg)
        Lx = ex_tunnel.get(
            "length", 5.0
        )  # horizontal length to the right (sets the vertical right boundary x)

        ex_mesh = self.config.get("tria_mesh_settings", {})

        ds = ex_mesh.get("wake_size", 0.01)  # spacing
        lc_block1 = ex_mesh.get("wake_size", 0.01)
        interface_size = ex_mesh.get("interface_min_size", 0.0001)
        ex_farfield = ex_mesh.get("farfield", "")
        lc_inner = ex_farfield.get("min_size", 0.3)
        lc_outer = ex_farfield.get("max_size", 3.2)
        d1 = ex_farfield.get("grading", 0.3)

        airfoil_line = te_line_remaining_lower.copy()
        airfoil_line.reverse()
        airfoil_line = simplify_polyline_by_min_edge(airfoil_line, interface_size)
        airfoil_line1 = te_line_remaining_upper.copy()
        airfoil_line1 = simplify_polyline_by_min_edge(airfoil_line1, interface_size)
        
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

        LU = np.hypot(xR - xL_top, yR_top - yL_top)
        nU = max(1, int(np.ceil(LU / ds)))
        tU = np.linspace(0.0, 1.0, nU + 1)[1:]

        upper_line = [
            (xL_top + tt * (xR - xL_top), yL_top + tt * (yR_top - yL_top)) for tt in tU
        ]

        yR_bot = yL_bot + np.tan(-ang) * (xR - xL_bot)

        LL = np.hypot(xR - xL_bot, yR_bot - yL_bot)
        nL = max(1, int(np.ceil(LL / ds)))
        tL = np.linspace(0.0, 1.0, nL + 1)[1:]

        lower_line = [
            (xL_bot + tt * (xR - xL_bot), yL_bot + tt * (yR_bot - yL_bot)) for tt in tL
        ]

        LR = abs(yR_top - yR_bot)
        nR = max(1, int(np.ceil(LR / ds)))
        tR = np.linspace(0.0, 1.0, nR + 1)[1:-1]

        right_line = [(xR, yR_bot + tt * (yR_top - yR_bot)) for tt in tR]

        poly_pts = airfoil_line + upper_line + right_line[::-1] + lower_line[::-1]
        inner_airfoil_line = boundaries_rest_outer.copy()
        shock_left = simplify_polyline_by_min_edge(shock_left, interface_size)

        inner_airfoil_line.extend(shock_left[:])

        inner_airfoil_line.extend(shock_upper[1:])

        shock_right.reverse()
        shock_right = simplify_polyline_by_min_edge(shock_right, interface_size)
        
        inner_airfoil_line.extend(shock_right[1:])
        inner_airfoil_line.extend(boundaries_right_upper)
        inner_airfoil_line.extend(upper_line)
        inner_airfoil_line.extend(right_line[::-1])
        inner_airfoil_line.extend(lower_line[::-1])
        #inner_airfoil_line = simplify_polyline_by_min_edge(inner_airfoil_line, lc_inner)
        ex_fardim = self.config.get("farfield", {})
        L_farfield = ex_fardim.get("length", 100)
        R_farfield = ex_fardim.get("radius", 50)

        L_tot = np.pi * R_farfield + 2 * R_farfield + 2 * L_farfield
        n_outer = int(np.ceil(L_tot / lc_outer))
        outer_airfoil_line = LineDistribution.closed_left_U(
            bottom_right=(L_farfield, -1 * R_farfield),
            vert_len=100,
            horiz_len=100,
            n_segments=n_outer,
        )[1:]

        algo = 6  # 2D meshing algorithm

        gmsh.initialize()
        gmsh.model.add("two_blocks_unstructured")
        gmsh.option.setNumber("Mesh.Algorithm", algo)

        # ---- Block 1: polygon surface ----
        p1 = [gmsh.model.occ.addPoint(x, y, 0.0) for (x, y) in poly_pts]
        l1 = [
            gmsh.model.occ.addLine(p1[i], p1[(i + 1) % len(p1)]) for i in range(len(p1))
        ]
        w1 = gmsh.model.occ.addWire(l1)
        surf1 = gmsh.model.occ.addPlaneSurface([w1])

        # ---- Block 2: annulus surface (outer with inner hole) ----
        p_out = [gmsh.model.occ.addPoint(x, y, 0.0) for (x, y) in outer_airfoil_line]
        l_out = [
            gmsh.model.occ.addLine(p_out[i], p_out[(i + 1) % len(p_out)])
            for i in range(len(p_out))
        ]
        w_out = gmsh.model.occ.addWire(l_out)

        p_in = [gmsh.model.occ.addPoint(x, y, 0.0) for (x, y) in inner_airfoil_line]
        l_in = [
            gmsh.model.occ.addLine(p_in[i], p_in[(i + 1) % len(p_in)])
            for i in range(len(p_in))
        ]
        w_in = gmsh.model.occ.addWire(l_in)

        surf2 = gmsh.model.occ.addPlaneSurface([w_out, w_in])

        gmsh.model.occ.synchronize()

        gmsh.model.occ.fragment([(2, surf1), (2, surf2)], [])
        gmsh.model.occ.synchronize()

        f_dist = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(f_dist, "CurvesList", l_in)
        gmsh.model.mesh.field.setNumber(f_dist, "NumPointsPerCurve", 50)

        f_thr = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(f_thr, "InField", f_dist)
        gmsh.model.mesh.field.setNumber(f_thr, "LcMin", float(lc_inner))
        gmsh.model.mesh.field.setNumber(f_thr, "LcMax", float(lc_outer))
        gmsh.model.mesh.field.setNumber(f_thr, "DistMin", 0.0)
        gmsh.model.mesh.field.setNumber(f_thr, "DistMax", float(d1))

        # --- Block 1 constant size restricted to surf1 ---
        f_c1 = gmsh.model.mesh.field.add("Constant")
        gmsh.model.mesh.field.setNumber(f_c1, "VIn", float(lc_block1))

        f_r1 = gmsh.model.mesh.field.add("Restrict")
        gmsh.model.mesh.field.setNumber(f_r1, "InField", f_c1)
        gmsh.model.mesh.field.setNumbers(f_r1, "SurfacesList", [surf1])

        # --- Combine both fields: smallest size wins ---
        f_min = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(f_min, "FieldsList", [f_thr, f_r1])
        gmsh.model.mesh.field.setAsBackgroundMesh(f_min)

        # Generate 2D mesh
        gmsh.model.mesh.generate(2)
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        P3 = np.asarray(node_coords, dtype=float).reshape(-1, 3)

        types, elem_tags, node_tags_elem = gmsh.model.mesh.getElements(2)
        TRI3_TYPE = 2

        T_tags = None
        for etype, e_tags, conn in zip(types, elem_tags, node_tags_elem):
            if etype == TRI3_TYPE:
                n_elems = len(e_tags)
                nper = len(conn) // n_elems
                T_tags = np.asarray(conn, dtype=np.int64).reshape(-1, nper)
                break

        tag_to_idx = {int(tag): i for i, tag in enumerate(node_tags.tolist())}
        T = np.vectorize(tag_to_idx.get, otypes=[np.int64])(T_tags)

        dict_tria = {"P": P3, "connectivity": T}
        self.trias.append(dict_tria)

        gmsh.finalize()
