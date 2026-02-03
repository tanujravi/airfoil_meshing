from airfoil import Airfoil
from BlockMesh import BlockMesh
from Connect import Connect
import numpy as np
import matplotlib.pyplot as plt
from LineDistribution import LineDistribution
import Elliptic
import sys
from Smooth_angle_based import SmoothAngleBased
from Smooth import Smooth
class Assemble:

    def __init__(self, config):
        self.blocks = list()
        self.trias = list()
        self.config = config

    def assemble(self):
        af_cfg = self.config.get("airfoil", {})

        if "contour_file" not in af_cfg:
            raise KeyError("Missing required key 'contour_file' in Airfoil config.")
        kwargs_airfoil = {
              "filename": af_cfg.get("contour_file", None),
              "k": af_cfg.get("spline_degree", 2)
            }
        aero = Airfoil.from_contour_file(**kwargs_airfoil)

        bd = self.config.get("airfoil_boundary", {})
        n_points = bd.get("n_points", 500)

        # Map YAML names -> function argument names (rename as your method expects)
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
        
        xp, yp, xp_te, yp_te, surf_normals = aero.distribute_points(n_points, **dist_kwargs)

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



        te_circular_outer = BlockMesh()
        line_airfoil_bound = mesh.getLine(number=0, direction = 'u')
        p0 = np.array(line_airfoil_bound[0])
        p1 = np.array(line_airfoil_bound[1])
        p2 = np.array(line_airfoil_bound[-1])
        p3 = np.array(line_airfoil_bound[-2])

        te_cfg = self.config.get("te_line_distribution", {})
        te_kwargs = {
            "h0": te_cfg.get("first_cell_thickness", 1e-4),
            "r": te_cfg.get("growth", 1.05),
            "even": te_cfg.get("even", False),
        }
        #line = LineDistribution.symmetric_decay_grow_line(p0, p2, **te_kwargs)

        line1 = LineDistribution.symmetric_decay_grow_line_fixed_N(p0, p2, te_kwargs["h0"], te_kwargs["r"], N = 60)
        p_te_up = p2
        p_te_down = p0

        len_te = np.linalg.norm(np.array(p_te_up)-np.array(p_te_down)) 

        """
        p_te_up = p0
        p_te_down = p2

        len_te = np.linalg.norm(np.array(p_te_up)-np.array(p_te_down)) 
        y_upper = p_te_up[1] + 0.1*len_te

        y_lower = p_te_down[1] - 0.1*len_te

        air_extrusion_up = mesh.getLine(number=-1, direction='v').copy()
        air_extrusion_up_filt = [p for p in air_extrusion_up if p[1] < y_upper]
        air_extrusion_up_remaining = [p for p in air_extrusion_up if p[1] >= y_upper]
        air_extrusion_up_remaining.insert(0, air_extrusion_up_filt[-1])

        air_extrusion_down = mesh.getLine(number=0, direction='v').copy()
        air_extrusion_down_filt = [p for p in air_extrusion_down if p[1] > y_lower]
        air_extrusion_down_remaining = [p for p in air_extrusion_down if p[1] <= y_lower]
        air_extrusion_down_remaining.insert(0, air_extrusion_down_filt[-1])
        """
        """
        boundaries_left = LineDistribution.arc_like_spline_from_2pts_2tangents(p2, line_airfoil_bound[1], line_airfoil_bound[0],
                                                                     p0, line_airfoil_bound[-1], line_airfoil_bound[-2])
        """
        
        boundaries_left = LineDistribution.arc_like_spline_from_2pts_2tangents_ref_spacing(p2, line_airfoil_bound[1], line_airfoil_bound[0],
                                                                     p0, line_airfoil_bound[-1], line_airfoil_bound[-2], ref_polyline=line1, handle_scale=1.5)
        
        #LineDistribution.plot_lines([p])
        #sys.exit()
        centre_te = 0.5*(p0+p2)
        #R_te = np.linalg.norm(centre_te-p0)
        """
        boundaries_left = LineDistribution.dist_arc_with_uniform_spacing(p1=p2, p2=p0, 
                                                                      center=centre_te, radius=R_te,
                                                                       segments = 50, direction="cw")
        """
        boundaries_upper = mesh.getLine(number=-1, direction='v').copy()
        boundaries_lower = mesh.getLine(number=0, direction='v').copy()

        pc1 = np.array(boundaries_upper[-1], float) 
        pc2 = np.array(boundaries_lower[-1], float) 
        T = np.array([1.0, 0.0])
        if np.allclose(pc1, pc2): raise ValueError("Points must be distinct.")
        M = 0.5*(pc1+pc2) 
        v = pc2-pc1
        n = np.array([-v[1], v[0]], float)
        n /= np.linalg.norm(n)
        s = n.dot(T - M)                      # closest point on the perpendicular bisector to T
        centre_arc_te_skeleton = M + s*n                           # center (h,k) near (1,0)
        R_arc_te_skeleton = np.linalg.norm(centre_arc_te_skeleton - pc1)            # radius

        """
        boundaries_right = LineDistribution.dist_arc_with_uniform_spacing(p1=boundaries_upper[-1], p2=boundaries_lower[-1], 
                                                                            center=centre_arc_te_skeleton, radius=R_arc_te_skeleton,
                                                                            segments = len(boundaries_left)-1, direction="cw")

        """
        L_circ = np.pi*R_arc_te_skeleton
        line2 = LineDistribution.triple_cluster_line(p0, (p0[0], p0[1]+L_circ), n_points = 41, r = te_kwargs["r"])
        line3 = LineDistribution.triple_cluster_line_smooth(p0, (p0[0], p0[1]+L_circ), n_points = 61, r_center=1.5, r_end=5, sigma_center= 0.3, sigma_end=0.1)
        
  
        boundaries_right = LineDistribution.dist_arc_with_ref_spacing(p1=boundaries_upper[-1], p2=boundaries_lower[-1], 
                                                                            center=centre_arc_te_skeleton, radius=R_arc_te_skeleton,
                                                                            ref_polyline=line3, direction="cw")

        #print(boundaries_lower)
        

        boundaries_left.reverse()
        boundaries_right.reverse()
        #boundary = [boundaries_lower, boundaries_upper, boundaries_left, boundaries_right]
        boundary = [ boundaries_upper, boundaries_lower, boundaries_left, boundaries_right]

        te_circular_outer.transfinite(boundary=boundary)
        self.blocks.append(te_circular_outer)

        """
        te_mesh = BlockMesh()

        line_airfoil_bound = mesh.getLine(number=0, direction = 'u')
        p0 = np.array(line_airfoil_bound[0])
        p1 = np.array(line_airfoil_bound[1])
        p2 = np.array(line_airfoil_bound[-1])
        p3 = np.array(line_airfoil_bound[-2])

        te_cfg = self.config.get("te_line_distribution", {})
        te_kwargs = {
            "h0": te_cfg.get("first_cell_thickness", 1e-4),
            "r": te_cfg.get("growth", 1.05),
            "even": te_cfg.get("even", False),
        }
        line = LineDistribution.symmetric_grow_decay_line(p0, p2, **te_kwargs)
        #line = LineDistribution.symmetric_grow_decay_line(p0, p2, h0=0.0001, r=1.05, even=False)
        te_ext = self.config.get("te_extrusion", {})
        if te_ext.get("cell_thickness") == "auto":

            thickness1 = np.linalg.norm(p1-p0)
            thickness2 = np.linalg.norm(p2-p3)
            te_mesh_cell_thickness = min(thickness1, thickness2)
        else:
            te_mesh_cell_thickness = float(te_ext.get("cell_thickness"))

        surf_normals_te = np.tile([1, 0], (len(line), 1))
        
        
        
        te_ex_kwargs = {
            "cell_thickness": te_mesh_cell_thickness,
            "growth": te_ext.get("growth", 1.005),
            "extrusion_distance": te_ext.get("extrusion_distance", 0.05),
        }
        line.reverse()
        te_mesh.extrudeLine_cell_thickness(line, surf_normals_te, **te_ex_kwargs)
        self.blocks.append(te_mesh)


        # finding top right point 




        angle = 65
        te_right_line = te_mesh.getLine(number=-1, direction="u").copy()
        p_te_up = te_right_line[0]
        p_te_down = te_right_line[-1]

        
        len_te = np.linalg.norm(np.array(p_te_up)-np.array(p_te_down)) 
        y_upper = p_te_up[1] + 2*len_te

        y_lower = p_te_down[1] - 2*len_te

        air_extrusion_up = mesh.getLine(number=-1, direction='v').copy()
        air_extrusion_up_filt = [p for p in air_extrusion_up if p[1] < y_upper]
        air_extrusion_up_remaining = [p for p in air_extrusion_up if p[1] >= y_upper]
        air_extrusion_up_remaining.insert(0, air_extrusion_up_filt[-1])
        p_top_right = (p_te_up[0]+(air_extrusion_up_filt[-1][1]-p_te_up[1])/np.tan(np.radians(angle)), air_extrusion_up_filt[-1][1])


        air_extrusion_down = mesh.getLine(number=0, direction='v').copy()
        air_extrusion_down_filt = [p for p in air_extrusion_down if p[1] > y_lower]
        air_extrusion_down_remaining = [p for p in air_extrusion_down if p[1] <= y_lower]
        air_extrusion_down_remaining.insert(0, air_extrusion_down_filt[-1])

        angle = 65
        p_bottom_right = (p_te_down[0]-(air_extrusion_down_filt[-1][1]-p_te_down[1])/np.tan(np.radians(angle)), air_extrusion_down_filt[-1][1])

        te_upper_mesh = BlockMesh()
        boundaries_lower = te_mesh.getLine(number=0, direction='v').copy()
        boundaries_left = air_extrusion_up_filt.copy()

        boundaries_right = LineDistribution.divide_line_by_reference(p_te_up, p_top_right, boundaries_left)
        #boundaries_upper = LineDistribution.divide_line_by_reference(air_extrusion_up_filt[-1], p_top_right, boundaries_lower)
        
        #boundaries_upper = LineDistribution.divide_line(air_extrusion_up_filt[-1], p_top_right, len(boundaries_lower)-1)
        boundaries_upper = LineDistribution.blended_proportional_uniform(air_extrusion_up_filt[-1], p_top_right, boundaries_lower, alpha=0.6 )
        
        boundary = [boundaries_lower, boundaries_upper, boundaries_left, boundaries_right]
        te_upper_mesh.transfinite(boundary=boundary)

        self.blocks.append(te_upper_mesh)


        te_lower_mesh = BlockMesh()

        boundaries_upper = te_mesh.getLine(number=-1, direction='v').copy()
        boundaries_left = air_extrusion_down_filt.copy()
        boundaries_right = LineDistribution.divide_line_by_reference(p_te_down, p_bottom_right, boundaries_left)
        #boundaries_lower = LineDistribution.divide_line_by_reference(air_extrusion_down_filt[-1], p_bottom_right, boundaries_upper)
        boundaries_lower = LineDistribution.blended_proportional_uniform(air_extrusion_down_filt[-1], p_bottom_right, boundaries_upper, alpha=0.6 )

        boundaries_left.reverse()
        boundaries_right.reverse()
        boundary = [boundaries_lower, boundaries_upper, boundaries_left, boundaries_right]
        #boundary = [boundaries_upper, boundaries_lower, boundaries_right, boundaries_left]
        #boundary = [boundaries_left, boundaries_right, boundaries_lower, boundaries_upper]
        #LineDistribution.plot_lines(boundary)
        te_lower_mesh.transfinite(boundary=boundary)

        self.blocks.append(te_lower_mesh)


        te_circular = BlockMesh()
        boundaries_left = te_mesh.getLine(number=-1, direction="u").copy()
        centre_arc_te = LineDistribution.line_intersection(p_te_up, p_top_right, p_te_down, p_bottom_right)

        R_arc_te = np.linalg.norm(np.asarray(p_top_right)-np.asarray(centre_arc_te))
        R_arc_te_1 = np.linalg.norm(np.asarray(p_bottom_right)-np.asarray(centre_arc_te))

        point3 = (centre_arc_te[0]+R_arc_te, centre_arc_te[1])

        centre_arc_te, R_arc_te = LineDistribution.circle_center_radius_from_3pts(p_top_right, p_bottom_right, point3)
        boundaries_right = LineDistribution.dist_arc_with_uniform_spacing(p1=p_top_right, p2=p_bottom_right, 
                                                                      center=centre_arc_te, radius=R_arc_te,
                                                                       segments = len(boundaries_left)-1, direction="cw")


        boundaries_upper = te_upper_mesh.getLine(number=-1, direction="v").copy()
        #print(boundaries_right)
        boundaries_lower = te_lower_mesh.getLine(number=-1, direction="v").copy()
        #boundaries_left.reverse()
        #boundaries_right.reverse()
        boundaries_lower.reverse()
        boundary = [boundaries_lower, boundaries_upper, boundaries_left, boundaries_right]
        

        te_circular.transfinite(boundary=boundary)
        self.blocks.append(te_circular)

        """





        """
        te_circular_outer = BlockMesh()


        boundaries_left = te_upper_mesh.getLine(number=-1, direction="u").copy()
        #te_circular.getLine(number=-1, direction="v")
        
        boundaries_left.extend(te_circular.getLine(number=-1, direction="v").copy()[1:])
        temp_line = te_lower_mesh.getLine(number=0, direction="u").copy()
        temp_line.reverse()
        boundaries_left.extend(temp_line[1:])

        boundaries_upper = air_extrusion_up_remaining.copy()
        boundaries_lower = air_extrusion_down_remaining.copy()


        p1 = np.array(boundaries_upper[-1], float) 
        p2 = np.array(boundaries_lower[-1], float) 
        T = np.array([1.0, 0.0])
        if np.allclose(p1, p2): raise ValueError("Points must be distinct.")
        M = 0.5*(p1+p2) 
        v = p2-p1
        n = np.array([-v[1], v[0]], float)
        n /= np.linalg.norm(n)
        s = n.dot(T - M)                      # closest point on the perpendicular bisector to T
        centre_arc_te_skeleton = M + s*n                           # center (h,k) near (1,0)
        R_arc_te_skeleton = np.linalg.norm(centre_arc_te_skeleton - p1)            # radius

        boundaries_right = LineDistribution.dist_arc_with_uniform_spacing(p1=boundaries_upper[-1], p2=boundaries_lower[-1], 
                                                                            center=centre_arc_te_skeleton, radius=R_arc_te_skeleton,
                                                                            segments = len(boundaries_left)-1, direction="cw")

        #print(boundaries_lower)
        

        boundary = [boundaries_lower, boundaries_upper, boundaries_left, boundaries_right]

        te_circular_outer.transfinite(boundary=boundary)
        self.blocks.append(te_circular_outer)

        """
        te_1 = BlockMesh()


        angle_1 = 3.5

        length_te_1 = 30


        y_upper = p_te_up[1] + 40*len_te

        y_lower = p_te_down[1] - 40*len_te

        p_top_right = p_te_up
        p_bottom_right = p_te_down

        te_circ_line = te_circular_outer.getLine(number= -1, direction="v").copy()
        boundaries_left = [p for p in te_circ_line if y_lower <= p[1] <= y_upper]

        te_1_remaining_upper = [p for p in te_circ_line if p[1] > y_upper]
        te_1_remaining_upper.insert(0, boundaries_left[-1])

        te_1_remaining_upper.reverse()

        te_1_remaining_lower = [p for p in te_circ_line if p[1] < y_lower]
        te_1_remaining_lower.append(boundaries_left[0])
        te_1_remaining_lower.reverse()

        #boundaries_left = 
        x_max_te_1 = np.max(np.asarray(boundaries_left.copy(), dtype = float)[:,0])
        start_point = (x_max_te_1, 0.5*(max(yp_te) + min(yp_te)))

        cell_thickness = np.linalg.norm(np.array(te_circular_outer.getLine(number= -1, direction="v").copy()[0]) - np.array(te_circular_outer.getLine(number= -2, direction="v").copy()[0]))
        center_ref = LineDistribution.grow_to_min_length_line(start_point, h0=cell_thickness, 
                                                    r=1.02, 
                                                    L= length_te_1, 
                                                    direction=(1, 0))

        length_te_1_corrected = center_ref[-1][0] - center_ref[0][0]
        
        p_te_1_top_right = (p_top_right[0] + length_te_1_corrected, p_top_right[1] + length_te_1_corrected*np.tan(np.radians(angle_1)))

        p_te_1_bottom_right = (p_bottom_right[0] + length_te_1_corrected, p_bottom_right[1] - length_te_1_corrected*np.tan(np.radians(angle_1)))


        x_te_1_max = centre_te[0] + np.linalg.norm(np.array(centre_te) - np.array(p_te_1_top_right))
        point3 = (x_te_1_max, 0.5*(p_te_1_top_right[1] + p_te_1_bottom_right[1]))
        centre_arc_te_1, R_arc_te_1 = LineDistribution.circle_center_radius_from_3pts(p_te_1_top_right, p_te_1_bottom_right, point3)

        
        #sys.exit()
        boundaries_right = LineDistribution.dist_arc_with_uniform_spacing(p1=p_te_1_top_right, p2=p_te_1_bottom_right, 
                                                                            center=centre_arc_te_1, radius=R_arc_te_1,
                                                                            segments = len(boundaries_left)-1, direction="cw")

        boundaries_upper = LineDistribution.divide_line_by_reference(p_top_right, p_te_1_top_right, center_ref)
        boundaries_lower = LineDistribution.divide_line_by_reference(p_bottom_right, p_te_1_bottom_right, center_ref)
        boundaries_right.reverse()
        boundary = [boundaries_lower, boundaries_upper, boundaries_left, boundaries_right]

        te_1.transfinite(boundary=boundary)
        self.blocks.append(te_1)


        
        inner_airfoil_line = mesh.getLine(number=-1, direction="u").copy()
        #inner_airfoil_line.extend(te_circular_outer.getLine(number=-1, direction="v").copy()[1:-1])
        inner_airfoil_line.extend(te_1_remaining_upper[1:])
        
        inner_airfoil_line.extend(te_1.getLine(number=-1, direction="u").copy()[1:])
        inner_airfoil_line.extend(te_1.getLine(number=-1, direction="v").copy()[::-1][1:])
        
        temp_line = te_1.getLine(number=0, direction="u").copy()

        temp_line.reverse()
        inner_airfoil_line.extend(temp_line[1:])
        inner_airfoil_line.extend(te_1_remaining_lower[1:-1])
        outer_airfoil_line = LineDistribution.closed_left_U(bottom_right=(100, -50), vert_len=100,horiz_len=100, n_segments=len(inner_airfoil_line)//6-1)[1:]

        import gmsh 
        inner = inner_airfoil_line
        outer = outer_airfoil_line
        lc_inner = 0.3   # small near inner boundary (airfoil layer)
        lc_outer = 3.2    # larger away from airfoil
        d1       = 0.3     # distance scale for growth
        p        = 1.5     # grading exponent

        gmsh.initialize()
        gmsh.model.add("airfoil_annulus_trias")

        def add_polyline_loop(pts):
            pids = [gmsh.model.geo.addPoint(x, y, 0) for x, y in pts]
            lids = [gmsh.model.geo.addLine(pids[i], pids[(i+1) % len(pids)])
                    for i in range(len(pids))]
            loop = gmsh.model.geo.addCurveLoop(lids)
            return pids, lids, loop

        # build outer surface with inner hole
        _, outer_lids, outer_loop = add_polyline_loop(outer)
        _, inner_lids, inner_loop = add_polyline_loop(inner)
        surf = gmsh.model.geo.addPlaneSurface([outer_loop, inner_loop])  # hole = inner_loop
        gmsh.model.geo.synchronize()

        # ---- size field: fine near inner boundary, coarser outward ----
        f_dist = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(f_dist, "CurvesList", inner_lids)  # grade from inner edge
        gmsh.model.mesh.field.setNumber(f_dist, "NumPointsPerCurve", 50)

        f_thr = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(f_thr, "InField", f_dist)
        gmsh.model.mesh.field.setNumber(f_thr, "LcMin", float(lc_inner))  # fine near inner
        gmsh.model.mesh.field.setNumber(f_thr, "LcMax", float(lc_outer))  # coarse far away
        gmsh.model.mesh.field.setNumber(f_thr, "DistMin", 0.0)            # where lc=lc_inner
        gmsh.model.mesh.field.setNumber(f_thr, "DistMax", float(d1))      # reach lc_outer by this distance

        gmsh.model.mesh.field.setAsBackgroundMesh(f_thr)
        gmsh.model.mesh.generate(2)           # triangles

        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        P3 = np.asarray(node_coords, dtype=float).reshape(-1, 3)   # (N,3): x,y,z


        types, elem_tags, node_tags_elem = gmsh.model.mesh.getElements(2)  # dim=2
        TRI3_TYPE = 2  # Gmsh element type ID for 3-node triangles

        T_tags = None

        for etype, e_tags, conn in zip(types, elem_tags, node_tags_elem):
            if etype == TRI3_TYPE:                     # only linear triangles
                n_elems = len(e_tags)
                if n_elems == 0:
                    continue
                nper = len(conn) // n_elems            # should be 3
                conn = np.asarray(conn, dtype=np.int64).reshape(-1, nper)
                T_tags = conn
                break                                  # stop after first TRI3 block

        if T_tags is None:
            raise RuntimeError("No TRI3 elements found in 2D mesh.")

        # Map Gmsh node tags (1-based, possibly non-contiguous) → compact 0-based indices
        tag_to_idx = {int(tag): i for i, tag in enumerate(node_tags.tolist())}
        T = np.vectorize(tag_to_idx.get, otypes=[np.int64])(T_tags)   # (M,3)

        dict_tria = {"P": P3, "connectivity" : T}
        self.trias.append(dict_tria)



        #latest
        """        
        te_circular = BlockMesh()

        line_airfoil_bound = mesh.getLine(number=0, direction = 'u')
        p0 = np.array(line_airfoil_bound[0])
        p1 = np.array(line_airfoil_bound[1])
        p2 = np.array(line_airfoil_bound[-1])
        p3 = np.array(line_airfoil_bound[-2])

        te_cfg = self.config.get("te_line_distribution", {})
        te_kwargs = {
            "h0": te_cfg.get("first_cell_thickness", 1e-4),
            "r": te_cfg.get("growth", 1.05),
            "even": te_cfg.get("even", False),
        }
        line = LineDistribution.symmetric_grow_decay_line(p0, p2, **te_kwargs)
        #line = LineDistribution.symmetric_grow_decay_line(p0, p2, h0=0.0001, r=1.05, even=False)
        te_ext = self.config.get("te_extrusion", {})
        if te_ext.get("cell_thickness") == "auto":

            thickness1 = np.linalg.norm(p1-p0)
            thickness2 = np.linalg.norm(p2-p3)
            te_mesh_cell_thickness = min(thickness1, thickness2)
        else:
            te_mesh_cell_thickness = float(te_ext.get("cell_thickness"))

        #surf_normals_te = np.tile([1, 0], (len(line), 1))
        
        
        boundaries_left = line.copy()

        boundaries_upper = mesh.getLine(number=-1, direction='v').copy()
        boundaries_lower = mesh.getLine(number=0, direction='v').copy()



        p1 = np.array(boundaries_upper[-1], float) 
        p2 = np.array(boundaries_lower[-1], float) 
        T = np.array([1.0, 0.0])
        if np.allclose(p1, p2): raise ValueError("Points must be distinct.")
        M = 0.5*(p1+p2) 
        v = p2-p1
        n = np.array([-v[1], v[0]], float)
        n /= np.linalg.norm(n)
        s = n.dot(T - M)                      # closest point on the perpendicular bisector to T
        centre_arc_te_skeleton = M + s*n                           # center (h,k) near (1,0)
        R_arc_te_skeleton = np.linalg.norm(centre_arc_te_skeleton - p1)            # radius

        boundaries_right = LineDistribution.dist_arc_with_uniform_spacing(p1=boundaries_upper[-1], p2=boundaries_lower[-1], 
                                                                            center=centre_arc_te_skeleton, radius=R_arc_te_skeleton,
                                                                            segments = len(boundaries_left)-1, direction="cw")

        boundaries_left.reverse()
        #print(boundaries_lower)
        

        boundary = [boundaries_lower, boundaries_upper, boundaries_left, boundaries_right]
        te_circular.transfinite(boundary=boundary)
        self.blocks.append(te_circular)

        """

        #latest


        #latest_but_one
        """        
        te_1 = BlockMesh()


        angle_1 = 4

        length_te_1 = 20


        y_upper = p_te_up[1] + 15*len_te

        y_lower = p_te_down[1] - 15*len_te

        te_circ_line = te_circular.getLine(number= -1, direction="v").copy()
        boundaries_left = [p for p in te_circ_line if y_lower <= p[1] <= y_upper]

        te_1_remaining_upper = [p for p in te_circ_line if p[1] > y_upper]
        te_1_remaining_upper.append(boundaries_left[0])

        te_1_remaining_lower = [p for p in te_circ_line if p[1] < y_lower]
        te_1_remaining_lower.insert(0, boundaries_left[-1])

        #boundaries_left = 
        x_max_te_1 = np.max(np.asarray(boundaries_left.copy(), dtype = float)[:,0])
        start_point = (x_max_te_1, 0.5*(max(yp_te) + min(yp_te)))

        
        cell_thickness = np.linalg.norm(np.array(te_circular.getLine(number= -1, direction="v").copy()[0]) - np.array(te_circular.getLine(number= -2, direction="v").copy()[0]))
        center_ref = LineDistribution.grow_to_min_length_line(start_point, h0=cell_thickness, 
                                                    r=1.04, 
                                                    L= length_te_1, 
                                                    direction=(1, 0))

        length_te_1_corrected = center_ref[-1][0] - center_ref[0][0]
        
        p_te_1_top_right = (p_top_right[0] + length_te_1_corrected, p_top_right[1] + length_te_1_corrected*np.tan(np.radians(angle_1)))

        p_te_1_bottom_right = (p_bottom_right[0] + length_te_1_corrected, p_bottom_right[1] - length_te_1_corrected*np.tan(np.radians(angle_1)))


        x_te_1_max = centre_arc_te[0] + np.linalg.norm(np.array(centre_arc_te) - np.array(p_te_1_top_right))
        point3 = (x_te_1_max, 0.5*(p_te_1_top_right[1] + p_te_1_bottom_right[1]))
        centre_arc_te_1, R_arc_te_1 = LineDistribution.circle_center_radius_from_3pts(p_te_1_top_right, p_te_1_bottom_right, point3)

        
        #sys.exit()
        boundaries_right = LineDistribution.dist_arc_with_uniform_spacing(p1=p_te_1_top_right, p2=p_te_1_bottom_right, 
                                                                            center=centre_arc_te_1, radius=R_arc_te_1,
                                                                            segments = len(boundaries_left)-1, direction="cw")

        boundaries_upper = LineDistribution.divide_line_by_reference(p_top_right, p_te_1_top_right, center_ref)
        boundaries_lower = LineDistribution.divide_line_by_reference(p_bottom_right, p_te_1_bottom_right, center_ref)
        boundary = [boundaries_lower, boundaries_upper, boundaries_left, boundaries_right]

        te_1.transfinite(boundary=boundary)
        self.blocks.append(te_1)



        airfoil_skeleton = BlockMesh()


        boundaries_lower = te_1_remaining_upper.copy()
        boundaries_lower.reverse()

        te_upper_line = te_upper_mesh.getLine(number=-1, direction="u").copy()
        te_upper_line.reverse()
        
        boundaries_lower.extend(te_upper_line[1:])
        #print(boundaries_lower)
        line_temp = mesh.getLine(number=-1, direction="u").copy()
        line_temp.reverse()
        boundaries_lower.extend(line_temp[1:])
        
        boundaries_lower.extend(te_lower_mesh.getLine(number=0, direction="u").copy()[1:]) 

        te_1_lower_circ = te_1_remaining_lower.copy()
        te_1_lower_circ.reverse()
        
        boundaries_lower.extend(te_1_lower_circ[1:])

        boundaries_lower.reverse()

        te_1_upper_line = te_1.getLine(number=0, direction="u").copy()
        L = 10
        x0, y0 = te_1_upper_line[0]
        boundaries_right = [p for p in te_1_upper_line if (p[0]-x0)**2 + (p[1]-y0)**2 <= L*L]
        
        
        te_1_lower_line = te_1.getLine(number=-1, direction="u").copy()
        boundaries_left = te_1_lower_line[:len(boundaries_right)]

        #centre_arc_te_skeleton = LineDistribution.line_intersection(boundaries_left[0], boundaries_left[-1], boundaries_right[0], boundaries_right[-1])
        #R_arc_te_skeleton = np.linalg.norm(np.asarray(boundaries_left[-1])-np.asarray(centre_arc_te_skeleton))
        #point3 = (R_arc_te_skeleton+centre_arc_te_skeleton[0], centre_arc_te_skeleton[1])
        #centre_arc_te_skeleton_corrected, R_arc_te_skeleton_corrected = LineDistribution.circle_center_radius_from_3pts(boundaries_left[-1], boundaries_right[-1], point3)
        p1 = np.array(boundaries_right[-1], float) 
        p2 = np.array(boundaries_left[-1], float) 
        T = np.array([1.0, 0.0])
        if np.allclose(p1, p2): raise ValueError("Points must be distinct.")
        M = 0.5*(p1+p2) 
        v = p2-p1
        n = np.array([-v[1], v[0]], float)
        n /= np.linalg.norm(n)
        s = n.dot(T - M)                      # closest point on the perpendicular bisector to T
        centre_arc_te_skeleton = M + s*n                           # center (h,k) near (1,0)
        R_arc_te_skeleton = np.linalg.norm(centre_arc_te_skeleton - p1)            # radius
        
        boundaries_upper = LineDistribution.dist_arc_with_uniform_spacing(p1=boundaries_right[-1], p2=boundaries_left[-1], 
                                                                            center=centre_arc_te_skeleton, radius=R_arc_te_skeleton,
                                                                            segments = len(boundaries_lower)-1, direction="ccw")
        


        boundaries_lower.reverse()
        #boundaries_upper.reverse()
        #boundaries_left.reverse()
        #boundaries_right.reverse()
        boundary = [boundaries_lower, boundaries_upper, boundaries_right, boundaries_left]
        airfoil_skeleton.transfinite(boundary=boundary)
        self.blocks.append(airfoil_skeleton)
        """
        #latest_but_one

        """
        airfoil_skeleton = BlockMesh()
        boundaries_lower = te_lower_mesh.getLine(number=0, direction="u").copy()
        boundaries_lower.reverse()
        boundaries_lower.extend(mesh.getLine(number=-1, direction="u").copy()[1:])
        boundaries_lower.extend(te_upper_mesh.getLine(number=-1, direction="u").copy()[1:]) 

        boundaries_lower.reverse()

        cell_thickness = np.linalg.norm(np.array(mesh.getLine(number= -1, direction="u").copy()[0]) - np.array(mesh.getLine(number= -2, direction="u").copy()[0]))
        skeleton_length = 10
        boundaries_right = LineDistribution.grow_to_min_length_line(boundaries_lower[0], h0=cell_thickness, 
                                                    r=1.04, 
                                                    L= skeleton_length, 
                                                    direction=(0, 1))

        skeleton_length_corrected = boundaries_right[-1][1] - boundaries_right[0][1]
        skeleton_top_right = boundaries_right[-1]

        boundaries_left = LineDistribution.grow_to_min_length_line(boundaries_lower[-1], h0=cell_thickness, 
                                                    r=1.04, 
                                                    L= skeleton_length, 
                                                    direction=(0, -1))

        skeleton_bottom_right = boundaries_left[-1]
        #boundaries_upper = LineDistribution.u_shape_with_ref_spacing(top_right= skeleton_top_right, bottom_right=skeleton_bottom_right, straight_len=1.3, ref_polyline=boundaries_lower)
        boundaries_upper = LineDistribution.u_shape_uniform(top_right= skeleton_top_right, bottom_right=skeleton_bottom_right, straight_len=1.3, n_segments=len(boundaries_lower)-1)
        boundaries_lower.reverse()
        boundaries_upper.reverse()
        #boundaries_left.reverse()
        #boundaries_right.reverse()
        boundary = [boundaries_lower, boundaries_upper, boundaries_left, boundaries_right]

        print(boundaries_upper)
        LineDistribution.plot_lines(boundary)

        #sys.exit()
        airfoil_skeleton.transfinite(boundary=boundary)
        self.blocks.append(airfoil_skeleton)
        """


        """

        te_2 = BlockMesh()


        angle_2 = 5

        length_te_2 = 20



        boundaries_left = te_1.getLine(number= -1, direction="v").copy()

        #boundaries_left = 

        x_max_te_1 = np.max(np.asarray(boundaries_left.copy(), dtype = float)[:,0])
        start_point = (x_max_te_1, 0.5*(max(yp_te) + min(yp_te)))
        
        cell_thickness = np.linalg.norm(np.array(te_1.getLine(number= -1, direction="v").copy()[0]) - np.array(te_1.getLine(number= -2, direction="v").copy()[0]))
        center_ref = LineDistribution.grow_to_min_length_line(start_point, h0=cell_thickness, 
                                                    r=1.04, 
                                                    L= length_te_2, 
                                                    direction=(1, 0))

        length_te_2_corrected = center_ref[-1][0] - center_ref[0][0]
        
        p_te_2_top_right = (p_te_1_top_right[0] + length_te_2_corrected, p_te_1_top_right[1] + length_te_2_corrected*np.tan(np.radians(angle_2)))

        p_te_2_bottom_right = (p_te_1_bottom_right[0] + length_te_2_corrected, p_te_1_bottom_right[1] - length_te_2_corrected*np.tan(np.radians(angle_2)))


        x_te_2_max = centre_arc_te[0] + np.linalg.norm(np.array(centre_arc_te) - np.array(p_te_2_top_right))
        point3 = (x_te_2_max, 0.5*(p_te_2_top_right[1] + p_te_2_bottom_right[1]))
        centre_arc_te_2, R_arc_te_2 = LineDistribution.circle_center_radius_from_3pts(p_te_2_top_right, p_te_2_bottom_right, point3)
        #sys.exit()
        boundaries_right = LineDistribution.dist_arc_with_uniform_spacing(p1=p_te_2_top_right, p2=p_te_2_bottom_right, 
                                                                            center=centre_arc_te_2, radius=R_arc_te_2,
                                                                            segments = len(boundaries_left)-1, direction="cw")
0
        boundaries_upper = LineDistribution.divide_line_by_reference(p_te_1_top_right, p_te_2_top_right, center_ref)
        boundaries_lower = LineDistribution.divide_line_by_reference(p_te_1_bottom_right, p_te_2_bottom_right, center_ref)
        boundary = [boundaries_lower, boundaries_upper, boundaries_left, boundaries_right]

        #LineDistribution.plot_lines(boundary)
        #sys.exit()
        te_2.transfinite(boundary=boundary)
        self.blocks.append(te_2)




        upper_circ_block = BlockMesh()
        boundaries_left = air_extrusion_up_remaining.copy()
        te_1_upper_line = te_1.getLine(number=0, direction='u').copy()
        boundaries_right = te_1_upper_line[:len(boundaries_left)]
        te_1_upper_line_remaining = te_1_upper_line[len(boundaries_left):]
        boundaries_lower = te_upper_mesh.getLine(number=-1, direction="u")
        centre_arc_te_upper = LineDistribution.line_intersection(boundaries_left[0], boundaries_left[-1], boundaries_right[0], boundaries_right[-1])
        R_arc_te_upper = np.linalg.norm(np.asarray(boundaries_left[-1])-np.asarray(centre_arc_te_upper))
        a1 = np.arctan2((boundaries_left[-1][1]-boundaries_left[0][1]), (boundaries_left[-1][0]-boundaries_left[0][0]))
        a2 = np.arctan2((boundaries_right[-1][1]-boundaries_right[0][1]),(boundaries_right[-1][0]-boundaries_right[0][0]))
        a_avg = 0.5*(a1+a2)
        point3 = (centre_arc_te_upper[0]*(1 + R_arc_te_upper*np.cos(a_avg)), centre_arc_te_upper[1]*(1 + R_arc_te_upper*np.sin(a_avg)))
        centre_arc_te_upper_corrected, R_arc_te_upper_corrected = LineDistribution.circle_center_radius_from_3pts(boundaries_left[-1], boundaries_right[-1], point3)

        boundaries_upper = LineDistribution.dist_arc_with_ref_spacing(p1=boundaries_left[-1], p2=boundaries_right[-1],  
                                                                            center=centre_arc_te_upper_corrected, radius=R_arc_te_upper_corrected,
                                                                            ref_polyline= boundaries_lower, direction="cw")
        boundary = [boundaries_lower, boundaries_upper, boundaries_left, boundaries_right]
        upper_circ_block.transfinite(boundary=boundary)
        #self.blocks.append(upper_circ_block)

        """
        
        
        """
        angle = 45
        te_right_line = te_mesh.getLine(number=-1, direction="u").copy()
        p_te_up = te_right_line[0]
        air_extrusion_up = mesh.getLine(number=-1, direction='v').copy()
        p_top_right = (p_te_up[0]+(air_extrusion_up[-1][1]-p_te_up[1])/np.tan(np.radians(angle)), air_extrusion_up[-1][1])


        p_te_down = te_right_line[-1]
        air_extrusion_down = mesh.getLine(number=0, direction='v').copy()
        p_bottom_right = (p_top_right[0], air_extrusion_down[-1][1])

        te_upper_mesh = BlockMesh()
        boundaries_lower = te_mesh.getLine(number=0, direction='v').copy()
        boundaries_left = mesh.getLine(number=-1, direction='v').copy()

        boundaries_right = LineDistribution.divide_line_by_reference(p_te_up, p_top_right, boundaries_left)
        boundaries_upper = LineDistribution.divide_line_by_reference(air_extrusion_up[-1], p_top_right, boundaries_lower)
        boundary = [boundaries_lower, boundaries_upper, boundaries_left, boundaries_right]
        te_upper_mesh.transfinite(boundary=boundary)

        self.blocks.append(te_upper_mesh)

        te_lower_mesh = BlockMesh()

        boundaries_upper = te_mesh.getLine(number=-1, direction='v').copy()
        boundaries_left = mesh.getLine(number=0, direction='v').copy()
        boundaries_right = LineDistribution.divide_line_by_reference(p_te_down, p_bottom_right, boundaries_left)
        boundaries_lower = LineDistribution.divide_line_by_reference(air_extrusion_down[-1], p_bottom_right, boundaries_upper)

        boundaries_left.reverse()
        boundaries_right.reverse()
        boundary = [boundaries_lower, boundaries_upper, boundaries_left, boundaries_right]
        #boundary = [boundaries_upper, boundaries_lower, boundaries_right, boundaries_left]
        #boundary = [boundaries_left, boundaries_right, boundaries_lower, boundaries_upper]
        #LineDistribution.plot_lines(boundary)
        te_lower_mesh.transfinite(boundary=boundary)

        self.blocks.append(te_lower_mesh)

        te_circular = BlockMesh()
        boundaries_left = te_mesh.getLine(number=-1, direction="u").copy()
        centre_arc_te = LineDistribution.line_intersection(p_te_up, p_top_right, p_te_down, p_bottom_right)
        print(centre_arc_te)

        R_arc_te = np.linalg.norm(np.asarray(p_top_right)-np.asarray(centre_arc_te))
        R_arc_te_1 = np.linalg.norm(np.asarray(p_bottom_right)-np.asarray(centre_arc_te))

        point3 = (centre_arc_te[0]+R_arc_te, centre_arc_te[1])
        print(f"{R_arc_te} {R_arc_te_1}")

        centre_arc_te, R_arc_te = LineDistribution.circle_center_radius_from_3pts(p_top_right, p_bottom_right, point3)
        
        print(centre_arc_te, R_arc_te)
        boundaries_right = LineDistribution.dist_arc_with_ref_spacing(p1=p_top_right, p2=p_bottom_right, 
                                                                      center=centre_arc_te, radius=R_arc_te,
                                                                      ref_polyline=boundaries_left, direction="cw")
        
        boundaries_upper = te_upper_mesh.getLine(number=-1, direction="v").copy()
        #print(boundaries_right)
        boundaries_lower = te_lower_mesh.getLine(number=-1, direction="v").copy()
        #boundaries_left.reverse()
        #boundaries_right.reverse()
        boundaries_lower.reverse()
        print(f"p_bottom = {p_bottom_right}")
        print(f"boundaries lower {boundaries_lower[0]} {boundaries_lower[-1]}")
        boundary = [boundaries_lower, boundaries_upper, boundaries_left, boundaries_right]
        

        LineDistribution.plot_lines(boundary)
        te_circular.transfinite(boundary=boundary)
        self.blocks.append(te_circular)
        """
        

        """
        V_block = BlockMesh()
        
        v_cfg = self.config.get("v_block", {})
        cg = v_cfg.get("centerline_growth", {})
        boundaries_left = te_lower_mesh.getLine(number=-1, direction="v")
        boundaries_left.extend(list(reversed(te_mesh.getLine(number=-1, direction="u")))[1:])

        boundaries_left.extend(te_upper_mesh.getLine(number=-1, direction="v")[1:])
        start_point = (xp_te[0], 0.5*(max(yp_te) + min(yp_te)))
        p0 = np.array(te_mesh.getLine(number=-1, direction="u")[0])
        p1 = np.array(te_mesh.getLine(number=-2, direction="u")[0])
        if cg.get("initial_cell_thickness") == "auto":
            cell_thickness = np.linalg.norm(p0-p1)
        else:
            cell_thickness = float(cg.get("initial_cell_thickness"))

        center_ref = LineDistribution.grow_to_min_length_line(start_point, h0=cell_thickness, 
                                                    r=cg.get("growth", 1.02), 
                                                    L=cg.get("min_length", 100.0), 
                                                    direction=(1, 0))
        slope = np.tan(np.radians(v_cfg.get("slope", 2.5)))
        x0, y0 = boundaries_left[-1][0], boundaries_left[-1][1]
        x_end1 = center_ref[-1][0]
        y_end1 = y0 + slope*(x_end1-x0)
        boundaries_upper = LineDistribution.divide_line_by_reference((x0,y0), (x_end1, y_end1), center_ref)

        slope = np.tan(np.radians(-1*v_cfg.get("slope", 2.5)))
        x0, y0 = boundaries_left[0][0], boundaries_left[0][1]
        x_end2 = center_ref[-1][0]
        y_end2 = y0 + slope*(x_end2-x0)
        boundaries_lower = LineDistribution.divide_line_by_reference((x0,y0), (x_end2, y_end2), center_ref)

        boundaries_right = LineDistribution.divide_line_by_reference((x_end2, y_end2), (x_end1, y_end1), boundaries_left)
        boundary = [boundaries_lower, boundaries_upper, boundaries_left, boundaries_right]

        V_block.transfinite(boundary=boundary)

        self.blocks.append(V_block)

        
        right_farfield_upper = BlockMesh()

        rf = self.config.get("right_farfield", {})
        c_radius = rf.get("c_radius", 50.0)
        te_upper_line = te_upper_mesh.getLine(number=-1, direction="u")
        p0 =  te_upper_mesh.getLine(number=0, direction="v")[0]
        p1 =  te_upper_mesh.getLine(number=0, direction="v")[-1]

        x_end = p0[0] + (p1[0]-p0[0])*(c_radius-p0[1])/(p1[1]-p0[1])
        #LineDistribution.plot_lines([te_upper_line])
        start = te_upper_line[0]
        #end = (start[0], c_radius)
        end = (x_end, c_radius)
        n_segments = rf.get("n_segments", 70)
        left_bound_cfg = rf.get("left_boundary")
        step_limit_left = left_bound_cfg.get("step_limit", 0.1)
        growth_left = left_bound_cfg.get("r", 1.08)
        right_bound_cfg = rf.get("right_boundary")
        step_limit_right = right_bound_cfg.get("step_limit", 0.5)
        growth_right = right_bound_cfg.get("r", 1.3)

        if left_bound_cfg.get("initial_cell_thickness") == "auto":
            right_farfield_upper_normal = np.array(te_upper_mesh.getLine(number=-1, direction="u")[0]) - np.array(te_upper_mesh.getLine(number=-2, direction="u")[0])
            right_farfield_upper_cell_thickness = np.linalg.norm(right_farfield_upper_normal)
        else:
            right_farfield_upper_cell_thickness = float(left_bound_cfg.get("initial_cell_thickness"))
        boundaries_left = LineDistribution.gp_to_ap_by_step_threshold_line(start, end, n_segments, right_farfield_upper_cell_thickness, growth_left, step_limit_left) 

        V_upper_line = V_block.getLine(number=-1, direction="u")
        start = V_upper_line[-1]
        end = (start[0], c_radius)

        if right_bound_cfg.get("initial_cell_thickness") == "auto":
            V_upper_normal = np.array(V_block.getLine(number=-1, direction="v")[-1]) - np.array(V_block.getLine(number=-1, direction="v")[-2])
            V_upper_cell_thickness = np.linalg.norm(V_upper_normal)
        else:
            V_upper_cell_thickness = float(right_bound_cfg.get("initial_cell_thickness"))


        boundaries_right = LineDistribution.gp_to_ap_by_step_threshold_line(start, end, n_segments, V_upper_cell_thickness, growth_right, step_limit_right)
        boundaries_lower = te_upper_mesh.getLine(number=-1, direction="u")
        boundaries_lower.extend(V_block.getLine(number=-1, direction="u")[1:])
        #boundaries_upper = [(x, c_radius) for (x, _) in boundaries_lower]
        boundaries_upper = LineDistribution.divide_line_by_reference(boundaries_left[-1], boundaries_right[-1], boundaries_lower)
        boundary = [boundaries_lower, boundaries_upper, boundaries_left, boundaries_right]
        right_farfield_upper.transfinite(boundary=boundary)

        self.blocks.append(right_farfield_upper)


        #####################################################done

        right_farfield_lower = BlockMesh()

        te_lower_line = te_lower_mesh.getLine(number=0, direction="u")
        start = te_lower_line[0]
        #end = (start[0], -1*c_radius)
        p0 =  te_lower_mesh.getLine(number=0, direction="v")[-1]
        p1 =  te_lower_mesh.getLine(number=0, direction="v")[0]
        
        x_end = p0[0] + (p1[0]-p0[0])*(-c_radius-p0[1])/(p1[1]-p0[1])
        end = (x_end, -1*c_radius)

        if left_bound_cfg.get("initial_cell_thickness") == "auto":
            right_farfield_lower_normal = np.array(te_lower_mesh.getLine(number=0, direction="u")[0]) - np.array(te_lower_mesh.getLine(number=1, direction="u")[0])
            right_farfield_lower_cell_thickness = np.linalg.norm(right_farfield_lower_normal)
        else:
            right_farfield_lower_cell_thickness = float(left_bound_cfg.get("initial_cell_thickness"))


        
        boundaries_left = LineDistribution.gp_to_ap_by_step_threshold_line(start, end, n_segments, right_farfield_lower_cell_thickness, growth_left, step_limit_left) 
        boundaries_left.reverse()

        
        V_lower_line = V_block.getLine(number=0, direction="u")
        start = V_lower_line[-1]
        end = (start[0], -1*c_radius)
        
        if right_bound_cfg.get("initial_cell_thickness") == "auto":
            V_lower_normal = np.array(V_block.getLine(number=-1, direction="v")[0]) - np.array(V_block.getLine(number=-1, direction="v")[1])
            V_lower_cell_thickness = np.linalg.norm(V_lower_normal)
        else:
            V_lower_cell_thickness = float(right_bound_cfg.get("initial_cell_thickness"))
        
        
        boundaries_right = LineDistribution.gp_to_ap_by_step_threshold_line(start, end, n_segments, V_lower_cell_thickness, growth_right, step_limit_right)
        boundaries_right.reverse()
        
        boundaries_upper = te_lower_mesh.getLine(number=0, direction="u").copy()
        boundaries_upper.extend(V_block.getLine(number=0, direction="u")[1:])
        
        #boundaries_lower = [(x, -1*c_radius) for (x, _) in boundaries_upper]
        boundaries_lower = LineDistribution.divide_line_by_reference(boundaries_left[0], boundaries_right[0], boundaries_upper)
        
        #boundaries_upper.reverse()
        #boundaries_lower.reverse()
        boundary = [boundaries_lower, boundaries_upper, boundaries_left, boundaries_right]

        right_farfield_lower.transfinite(boundary=boundary)

        self.blocks.append(right_farfield_lower)

        
        c_block = BlockMesh()
        p1 = right_farfield_upper.getLine(number = 0, direction = "v")[-1]
        p2 = right_farfield_lower.getLine(number = 0, direction = "v")[0]
        #sys.exit()
        boundaries_left = mesh.getLine(number=-1, direction="u")

        boundaries_right = LineDistribution.find_alphas(
            p2, p1,
            ref_polyline=boundaries_left,
            normals=surf_normals,
            alphaMin_max=0.7,
            alphaMax_max=0.9,
            gamma=0.01,
            tol_xi=1e-5,
            direction="cw",
            )


        boundaries_upper = right_farfield_upper.getLine(number=0, direction="v")
        boundaries_lower = right_farfield_lower.getLine(number=0, direction="v")
        boundaries_lower.reverse()

        #boundary = [boundaries_lower, boundaries_upper, boundaries_left, boundaries_right]
        boundary = [boundaries_left, boundaries_right, boundaries_lower, boundaries_upper]
        c_block.transfinite(boundary=boundary)
        
        self.blocks.append(c_block)
        """