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

        ##########################################################################################################
        te_circular_outer = BlockMesh()
        line_airfoil_bound = mesh.getLine(number=0, direction = 'u')
        p0 = np.array(line_airfoil_bound[0])
        p1 = np.array(line_airfoil_bound[1])
        p2 = np.array(line_airfoil_bound[-1])
        p3 = np.array(line_airfoil_bound[-2])

        n_po = 70
        p_te_up = p2
        p_te_down = p0

        len_te = np.linalg.norm(np.array(p_te_up)-np.array(p_te_down)) 

        y_upper = p_te_up[1] + 0.25*len_te

        y_lower = p_te_down[1] - 0.25*len_te

        p_top_right = p_te_up
        p_bottom_right = p_te_down

        te_upper_line = mesh.getLine(number=-1, direction='v').copy()
        boundaries_upper = [p for p in te_upper_line if p[1] <= y_upper]

        te_line_remaining_upper = [p for p in te_upper_line if p[1] > y_upper]
        te_line_remaining_upper.insert(0, boundaries_upper[-1])

        #te_line_remaining_upper.reverse()
        te_lower_line = mesh.getLine(number=0, direction='v').copy()
        boundaries_lower = te_lower_line[:len(boundaries_upper)]

        te_line_remaining_lower = te_lower_line[len(boundaries_upper):]
        te_line_remaining_lower.insert(0, boundaries_lower[-1])
        #te_line_remaining_lower.reverse()
        boundaries_left = LineDistribution.arc_like_spline_from_2pts_2tangents(p2, line_airfoil_bound[1], line_airfoil_bound[0],
                                                                     p0, line_airfoil_bound[-1], line_airfoil_bound[-2], n_points=n_po+1,  handle_scale=1.4)
        
        centre_te = 0.5*(p0+p2)
        #R_te = np.linalg.norm(centre_te-p0)
        #boundaries_upper = mesh.getLine(number=-1, direction='v').copy()
        #boundaries_lower = mesh.getLine(number=0, direction='v').copy()

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

        L_circ = np.pi*R_arc_te_skeleton
        line3 = LineDistribution.triple_cluster_line_smooth(p0, (p0[0], p0[1]+L_circ), n_points = n_po+1, r_center=1.5, r_end=10, sigma_center= 0.3, sigma_end=0.2)
        
        normal_ref = LineDistribution.dist_arc_with_normals_spline(p1=boundaries_upper[-1], p2=boundaries_lower[-1],
               ref_polyline=boundaries_left,
                center=centre_arc_te_skeleton,
                radius=R_arc_te_skeleton,
        )
        boundaries_right = LineDistribution.dist_arc_with_ref_spacing(p1=boundaries_upper[-1], p2=boundaries_lower[-1], 
                                                                    center=centre_arc_te_skeleton, radius=R_arc_te_skeleton,
                                                                    ref_polyline=normal_ref, direction="cw", alpha=0.9)

        boundaries_left.reverse()
        boundaries_right.reverse()
        #boundary = [boundaries_lower, boundaries_upper, boundaries_left, boundaries_right]
        boundary = [ boundaries_upper, boundaries_lower, boundaries_left, boundaries_right]

        te_circular_outer.transfinite(boundary=boundary)
        self.blocks.append(te_circular_outer)


        ##########################################################################################################


        """    
        ##########################################################################################################
        te_circular_middle = BlockMesh()

        boundaries_left = te_circular_outer.getLine(number=-1, direction = "v").copy()

        y_upper = boundaries_left[-1][1] + 5*len_te
        y_lower = boundaries_left[0][1] - 5*len_te

        p_top_right = p_te_up
        p_bottom_right = p_te_down

        te_upper_line_temp = te_line_remaining_upper.copy()
        
        boundaries_upper = [p for p in te_upper_line_temp if p[1] <= y_upper]

        #print(boundaries_upper)
        te_line_remaining_upper_middle = [p for p in te_upper_line_temp if p[1] > y_upper]
        te_line_remaining_upper_middle.insert(0, boundaries_upper[-1])

        #te_line_remaining_upper.reverse()


        te_lower_line_temp = te_line_remaining_lower.copy()
        boundaries_lower = te_lower_line_temp[:len(boundaries_upper)]

        #print(boundaries_lower)
        te_line_remaining_lower_middle = te_lower_line_temp[len(boundaries_upper):]
        te_line_remaining_lower_middle.insert(0, boundaries_lower[-1])
        #te_line_remaining_lower.reverse()

        centre_te = 0.5*(p0+p2)
        #R_te = np.linalg.norm(centre_te-p0)
        #boundaries_upper = mesh.getLine(number=-1, direction='v').copy()
        #boundaries_lower = mesh.getLine(number=0, direction='v').copy()

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

        L_circ = np.pi*R_arc_te_skeleton
        line3 = LineDistribution.triple_cluster_line_smooth(p0, (p0[0], p0[1]+L_circ), n_points = n_po+1, r_center=1.5, r_end=10, sigma_center= 0.3, sigma_end=0.2)
        

        #boundaries_right = LineDistribution.dist_arc_with_ref_spacing(p1=boundaries_upper[-1], p2=boundaries_lower[-1], 
        #                                                                    center=centre_arc_te_skeleton, radius=R_arc_te_skeleton,
        #                                                                    ref_polyline=boundaries_left, direction="cw", alpha=0.5)

        normal_ref = LineDistribution.dist_arc_with_normals_spline(p2=boundaries_upper[-1], p1=boundaries_lower[-1],
               ref_polyline=boundaries_left,
                center=centre_arc_te_skeleton,
                radius=R_arc_te_skeleton,
        )
        #print(boundaries_lower)
        boundaries_right = LineDistribution.dist_arc_with_ref_spacing(p1=boundaries_upper[-1], p2=boundaries_lower[-1], 
                                                                            center=centre_arc_te_skeleton, radius=R_arc_te_skeleton,
                                                                            ref_polyline=normal_ref, direction="cw", alpha=0.7)

        boundaries_left.reverse()
        #boundaries_right.reverse()
        #print(boundaries_left)
        #print(boundaries_right)
        #boundaries_upper.reverse()
        #boundaries_lower.reverse()
        #boundary = [boundaries_lower, boundaries_upper, boundaries_left, boundaries_right]
        boundary = [ boundaries_upper, boundaries_lower, boundaries_left, boundaries_right]
        #sys.exit()
 
        te_circular_middle.transfinite(boundary=boundary)
        self.blocks.append(te_circular_middle)


        ##########################################################################################################

        
        ##########################################################################################################
        te_circular_middle_1 = BlockMesh()

        boundaries_left = te_circular_middle.getLine(number=-1, direction = "v").copy()
        y_upper = boundaries_left[0][1] + 10*len_te
        y_lower = boundaries_left[-1][1] - 10*len_te

        p_top_right = p_te_up
        p_bottom_right = p_te_down

        te_upper_line_temp = te_line_remaining_upper_middle.copy()

        boundaries_upper = [p for p in te_upper_line_temp if p[1] <= y_upper]

        #print(boundaries_upper)
        te_line_remaining_upper_middle_1 = [p for p in te_upper_line_temp if p[1] > y_upper]
        te_line_remaining_upper_middle_1.insert(0, boundaries_upper[-1])

        #te_line_remaining_upper.reverse()


        te_lower_line_temp = te_line_remaining_lower_middle.copy()
        boundaries_lower = te_lower_line_temp[:len(boundaries_upper)]

        #print(boundaries_lower)
        te_line_remaining_lower_middle_1 = te_lower_line_temp[len(boundaries_upper):]
        te_line_remaining_lower_middle_1.insert(0, boundaries_lower[-1])
        #te_line_remaining_lower.reverse()

        centre_te = 0.5*(p0+p2)
        #R_te = np.linalg.norm(centre_te-p0)
        #boundaries_upper = mesh.getLine(number=-1, direction='v').copy()
        #boundaries_lower = mesh.getLine(number=0, direction='v').copy()

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

        L_circ = np.pi*R_arc_te_skeleton
        line3 = LineDistribution.triple_cluster_line_smooth(p0, (p0[0], p0[1]+L_circ), n_points = n_po+1, r_center=1.5, r_end=10, sigma_center= 0.3, sigma_end=0.2)
        

        #boundaries_right = LineDistribution.dist_arc_with_ref_spacing(p1=boundaries_upper[-1], p2=boundaries_lower[-1], 
        #                                                                    center=centre_arc_te_skeleton, radius=R_arc_te_skeleton,
        #                                                                    ref_polyline=boundaries_left, direction="cw", alpha=0.5)

        normal_ref = LineDistribution.dist_arc_with_normals_spline(p1=boundaries_upper[-1], p2=boundaries_lower[-1],
               ref_polyline=boundaries_left,
                center=centre_arc_te_skeleton,
                radius=R_arc_te_skeleton,
        )
        boundaries_right = LineDistribution.dist_arc_with_ref_spacing(p1=boundaries_upper[-1], p2=boundaries_lower[-1], 
                                                                            center=centre_arc_te_skeleton, radius=R_arc_te_skeleton,
                                                                            ref_polyline=normal_ref, direction="cw", alpha=0.5)

        boundaries_left.reverse()
        boundaries_right.reverse()
        boundary = [ boundaries_upper, boundaries_lower, boundaries_left, boundaries_right]
 
        te_circular_middle_1.transfinite(boundary=boundary)
        self.blocks.append(te_circular_middle_1)


        ##########################################################################################################


        ##########################################################################################################
        te_circular_middle_2 = BlockMesh()

        boundaries_left = te_circular_middle_1.getLine(number=-1, direction = "v").copy()
        y_upper = boundaries_left[-1][1] + 30*len_te
        y_lower = boundaries_left[0][1] - 30*len_te

        p_top_right = p_te_up
        p_bottom_right = p_te_down

        te_upper_line_temp = te_line_remaining_upper_middle_1.copy()

        boundaries_upper = [p for p in te_upper_line_temp if p[1] <= y_upper]

        #print(boundaries_upper)
        te_line_remaining_upper_middle_2 = [p for p in te_upper_line_temp if p[1] > y_upper]
        te_line_remaining_upper_middle_2.insert(0, boundaries_upper[-1])

        #te_line_remaining_upper.reverse()


        te_lower_line_temp = te_line_remaining_lower_middle_1.copy()
        boundaries_lower = te_lower_line_temp[:len(boundaries_upper)]

        #print(boundaries_lower)
        te_line_remaining_lower_middle_2 = te_lower_line_temp[len(boundaries_upper):]
        te_line_remaining_lower_middle_2.insert(0, boundaries_lower[-1])
        #te_line_remaining_lower.reverse()

        centre_te = 0.5*(p0+p2)
        #R_te = np.linalg.norm(centre_te-p0)
        #boundaries_upper = mesh.getLine(number=-1, direction='v').copy()
        #boundaries_lower = mesh.getLine(number=0, direction='v').copy()

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

        L_circ = np.pi*R_arc_te_skeleton
        line3 = LineDistribution.triple_cluster_line_smooth(p0, (p0[0], p0[1]+L_circ), n_points = n_po+1, r_center=1.5, r_end=10, sigma_center= 0.3, sigma_end=0.2)

        normal_ref = LineDistribution.dist_arc_with_normals_spline(p2=boundaries_upper[-1], p1=boundaries_lower[-1],
               ref_polyline=boundaries_left,
                center=centre_arc_te_skeleton,
                radius=R_arc_te_skeleton,
        )
        
        #LineDistribution.plot_lines([normal_ref])
        #print(boundaries_lower)
        boundaries_right = LineDistribution.dist_arc_with_ref_spacing(p1=boundaries_upper[-1], p2=boundaries_lower[-1], 
                                                                            center=centre_arc_te_skeleton, radius=R_arc_te_skeleton,
                                                                            ref_polyline=normal_ref, direction="cw", alpha=0.5)

        boundaries_left.reverse()
        #boundary = [boundaries_lower, boundaries_upper, boundaries_left, boundaries_right]
        boundary = [ boundaries_upper, boundaries_lower, boundaries_left, boundaries_right]
 
        te_circular_middle_2.transfinite(boundary=boundary)
        self.blocks.append(te_circular_middle_2)


        ##########################################################################################################

        ##########################################################################################################
        te_circular_outer_1 = BlockMesh()

        boundaries_upper = te_line_remaining_upper_middle_2.copy()
        boundaries_lower = te_line_remaining_lower_middle_2.copy()

        boundaries_left = te_circular_middle_2.getLine(number=-1, direction="v")
        #LineDistribution.plot_lines([p])
        #sys.exit()
        centre_te = 0.5*(p0+p2)
        #R_te = np.linalg.norm(centre_te-p0)
        #boundaries_upper = mesh.getLine(number=-1, direction='v').copy()
        #boundaries_lower = mesh.getLine(number=0, direction='v').copy()

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

        L_circ = np.pi*R_arc_te_skeleton
        line3 = LineDistribution.triple_cluster_line_smooth(p0, (p0[0], p0[1]+L_circ), n_points = n_po+1, r_center=4, r_end=25, sigma_center= 0.2, sigma_end=0.2)
        
  
        boundaries_right = LineDistribution.dist_arc_with_ref_spacing(p1=boundaries_upper[-1], p2=boundaries_lower[-1], 
                                                                            center=centre_arc_te_skeleton, radius=R_arc_te_skeleton,
                                                                            ref_polyline=line3, direction="cw", alpha=0.0)


        boundaries_left.reverse()
        boundaries_right.reverse()
        #boundary = [boundaries_lower, boundaries_upper, boundaries_left, boundaries_right]
        boundary = [ boundaries_upper, boundaries_lower, boundaries_left, boundaries_right]

        LineDistribution.plot_lines(boundary)
        te_circular_outer_1.transfinite(boundary=boundary)
        self.blocks.append(te_circular_outer_1)
        """
        ##########################################################################################################


        shock_box = BlockMesh()
        mesh_outer = mesh.getLine(number=-1, direction='u').copy()
        """
        boundaries_lower = [
            (x, y) for (x, y) in mesh_outer
            if 0.05 < x < 0.7 and y > 0
        ]
        """
        
        boundaries_lower = []
        surf_normals_lower = []

        for i, (x, y) in enumerate(mesh_outer):
            if 0.05 < x < 0.7 and y > 0:
                boundaries_lower.append((x, y))
                surf_normals_lower.append(surf_normals[i])
        
        boundaries_rest = [
            (x, y) for (x, y) in mesh_outer
            if not (0.05 < x < 0.7 and y > 0)
        ]

        boundaries_right_upper = [
            (x, y) for (x, y) in boundaries_rest
            if x > 0.7 and y > 0
        ]

        boundaries_rest_outer = [
            (x, y) for (x, y) in boundaries_rest
            if not (x > 0.7 and y > 0)
        ]
        
        p_right_down = max(boundaries_lower, key=lambda p: p[0])
        length = 0.6
        growth = 1.02
        cell_thick = np.linalg.norm(np.array(mesh.getLine(number=-1, direction='u').copy()[0])- np.array(mesh.getLine(number=-2, direction='u').copy()[0]))
        dy = []
        s = 0.0
        h = float(cell_thick)

        while s + h <= length + 1e-12:
            dy.append(h)
            s += h
            h *= growth

        dy = np.concatenate(([0.0], np.cumsum(dy)))

        # --- perpendicular polyline (+y direction) ---
        x0, y0 = p_right_down

        ex_kwargs = {
            "cell_thickness": cell_thick,
            "growth": growth,
            "extrusion_distance": length,
        }
        #airfoil_bd = [(x, y) for x, y in zip(xp, yp)]
        shock_box.extrudeLine_cell_thickness(boundaries_lower, surf_normals_lower, **ex_kwargs)
        self.blocks.append(shock_box)
        shock_right = shock_box.getLine(number= -1, direction="v")
        shock_left = shock_box.getLine(number= 0, direction="v")
        shock_upper = shock_box.getLine(number= -1, direction="u")
        """
        boundaries_right = [(x0, y0 + d) for d in dy]
        shock_right = boundaries_right.copy()
        p_left_down = min(boundaries_lower, key=lambda p: p[0])

        #boundaries_left = [(p_left_down[0], y) for (_, y) in boundaries_right]

        boundaries_left = LineDistribution.divide_line_by_reference(boundaries_lower[0], (boundaries_lower[0][0], boundaries_right[-1][1]), boundaries_right)
        shock_left = boundaries_left.copy()

        boundaries_upper = LineDistribution.divide_line_by_reference(boundaries_left[-1], boundaries_right[-1], boundaries_lower)
        shock_upper = boundaries_upper.copy()
        #print(boundaries_lower)
        #LineDistribution.plot_lines([boundaries_upper])
        #sys.exit()
        #boundaries_left.reverse()
        #boundaries_right.reverse()
        #boundaries_upper.reverse()
        #boundaries_lower.reverse()
        boundary = [ boundaries_lower, boundaries_upper, boundaries_left, boundaries_right]
        #lo = boundary.append(mesh_outer)
        #LineDistribution.plot_lines(boundary)
        #sys.exit()
        shock_box.transfinite(boundary=boundary)


        self.blocks.append(shock_box)
        """
        """
        ##########################################################################################################
        te_1 = BlockMesh()


        angle_1 = 3.5

        length_te_1 = 30


        y_upper = p_te_up[1] + 40*len_te

        y_lower = p_te_down[1] - 40*len_te

        p_top_right = p_te_up
        p_bottom_right = p_te_down

        te_circ_line = te_circular_outer_1.getLine(number= -1, direction="v").copy()
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

        cell_thickness = np.linalg.norm(np.array(te_circular_outer_1.getLine(number= -1, direction="v").copy()[0]) - np.array(te_circular_outer_1.getLine(number= -2, direction="v").copy()[0]))
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


        ##########################################################################################################
        
        #inner_airfoil_line = mesh.getLine(number=-1, direction="u").copy()
        inner_airfoil_line = boundaries_rest_outer.copy()
        inner_airfoil_line.extend(shock_left[1:])
       
        inner_airfoil_line.extend(shock_upper[1:])
        
        shock_right.reverse()

        inner_airfoil_line.extend(shock_right[1:])
        inner_airfoil_line.extend(boundaries_right_upper)
        #inner_airfoil_line.extend(te_circular_outer.getLine(number=-1, direction="v").copy()[1:-1])
        inner_airfoil_line.extend(te_1_remaining_upper[1:])
        
        inner_airfoil_line.extend(te_1.getLine(number=-1, direction="u").copy()[1:])
        inner_airfoil_line.extend(te_1.getLine(number=-1, direction="v").copy()[::-1][1:])
        
        temp_line = te_1.getLine(number=0, direction="u").copy()

        temp_line.reverse()
        inner_airfoil_line.extend(temp_line[1:])
        inner_airfoil_line.extend(te_1_remaining_lower[1:-1])

        LineDistribution.plot_lines([inner_airfoil_line])
        outer_airfoil_line = LineDistribution.closed_left_U(bottom_right=(100, -50), vert_len=100,horiz_len=100, n_segments=len(inner_airfoil_line)//6-1)[1:]
        """

        """
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
        """

        import gmsh
        airfoil_line = te_line_remaining_lower.copy()
        airfoil_line.reverse()
        airfoil_line1 = te_line_remaining_upper.copy()
        te_line = te_circular_outer.getLine(number=-1, direction="v").copy()
        airfoil_line.extend(te_line[1:])
        
        airfoil_line.extend(airfoil_line1[1:])






        angle_deg = 5.0     # magnitude of slope angle (deg)
        Lx        = 5.0      # horizontal length to the right (sets the vertical right boundary x)
        ds        = 0.01     # spacing
        # airfoil_line = [...]  # LEFT boundary polyline (list of (x,y))
        # ------------------------------------------------

        # Ensure left boundary goes bottom -> top
        if airfoil_line[0][1] > airfoil_line[-1][1]:
            airfoil_line = airfoil_line[::-1]

        # Left boundary endpoints
        xL_bot, yL_bot = airfoil_line[0]
        xL_top, yL_top = airfoil_line[-1]

        ang = np.deg2rad(angle_deg)

        # Common right-boundary x (vertical right side)
        xR = xL_top + Lx   # could also use xL_bot + Lx; same if Lx is "horizontal length"

        # =================================================
        # Upper side (+angle): top-left -> (xR, y_top + tan(ang)*Lx)
        # =================================================
        yR_top = yL_top + np.tan(+ang) * (xR - xL_top)

        LU = np.hypot(xR - xL_top, yR_top - yL_top)
        nU = max(1, int(np.ceil(LU / ds)))
        tU = np.linspace(0.0, 1.0, nU + 1)[1:]  # skip start point

        upper_line = [
            (xL_top + tt * (xR - xL_top),
            yL_top + tt * (yR_top - yL_top))
            for tt in tU
        ]

        # =================================================
        # Lower side (-angle): bottom-left -> (xR, y_bot - tan(ang)*Lx)
        # =================================================
        yR_bot = yL_bot + np.tan(-ang) * (xR - xL_bot)

        LL = np.hypot(xR - xL_bot, yR_bot - yL_bot)
        nL = max(1, int(np.ceil(LL / ds)))
        tL = np.linspace(0.0, 1.0, nL + 1)[1:]  # skip start point

        lower_line = [
            (xL_bot + tt * (xR - xL_bot),
            yL_bot + tt * (yR_bot - yL_bot))
            for tt in tL
        ]

        # =================================================
        # Right side: vertical line x = xR connecting (xR,yR_bot) -> (xR,yR_top)
        # =================================================
        LR = abs(yR_top - yR_bot)
        nR = max(1, int(np.ceil(LR / ds)))
        tR = np.linspace(0.0, 1.0, nR + 1)[1:-1]  # skip both ends (already in upper/lower)

        right_line = [
            (xR, yR_bot + tt * (yR_top - yR_bot))
            for tt in tR
        ]

        # =================================================
        # Final closed polygon (CCW)
        # left: bottom->top, upper: top->right, right: top->bottom, lower: right->left
        # =================================================
        poly_pts = airfoil_line + upper_line + right_line[::-1] + lower_line[::-1]



        inner_airfoil_line = boundaries_rest_outer.copy()
        inner_airfoil_line.extend(shock_left[:])
       
        inner_airfoil_line.extend(shock_upper[1:])
        
        shock_right.reverse()

        inner_airfoil_line.extend(shock_right[1:])
        inner_airfoil_line.extend(boundaries_right_upper)
        #inner_airfoil_line.extend(te_circular_outer.getLine(number=-1, direction="v").copy()[1:-1])
        inner_airfoil_line.extend(upper_line)
        inner_airfoil_line.extend(right_line[::-1])
        inner_airfoil_line.extend(lower_line[::-1])

        outer_airfoil_line = LineDistribution.closed_left_U(bottom_right=(100, -50), vert_len=100,horiz_len=100, n_segments=len(inner_airfoil_line)//6-1)[1:]



        """
        lc = 0.01
        algo = 6
        if len(poly_pts) < 3:
            raise ValueError("poly_pts must have at least 3 points")

        gmsh.initialize()
        gmsh.model.add("poly_tria_mesh")

        # ---- geometry (OCC is robust) ----
        # Create points
        p = [gmsh.model.occ.addPoint(x, y, 0.0) for (x, y) in poly_pts]

        # Create lines between consecutive points (and close)
        l = []
        for i in range(len(p)):
            l.append(gmsh.model.occ.addLine(p[i], p[(i + 1) % len(p)]))

        # Create a closed wire and a surface
        wire = gmsh.model.occ.addWire(l)
        surf = gmsh.model.occ.addPlaneSurface([wire])

        gmsh.model.occ.synchronize()

        # ---- mesh sizing: constant size in the region ----
        # (Simplest: set size at points + also a background Constant field)
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), lc)

        f_const = gmsh.model.mesh.field.add("Constant")
        gmsh.model.mesh.field.setNumber(f_const, "VIn", float(lc))

        f_res = gmsh.model.mesh.field.add("Restrict")
        gmsh.model.mesh.field.setNumber(f_res, "InField", f_const)
        gmsh.model.mesh.field.setNumbers(f_res, "SurfacesList", [surf])

        gmsh.model.mesh.field.setAsBackgroundMesh(f_res)

        # ---- generate triangles ----
        gmsh.option.setNumber("Mesh.Algorithm", algo)


        #######################################################################################


        inner = inner_airfoil_line
        outer = outer_airfoil_line
        lc_inner = 0.3   # small near inner boundary (airfoil layer)
        lc_outer = 3.2    # larger away from airfoil
        d1       = 0.3     # distance scale for growth
        p        = 1.5     # grading exponent

        #gmsh.initialize()
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
        surf1 = gmsh.model.geo.addPlaneSurface([outer_loop, inner_loop])  # hole = inner_loop
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









        gmsh.model.mesh.generate(2)


        # ---- extract nodes + TRI3 connectivity (0-based indices) ----
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        P3 = np.asarray(node_coords, dtype=float).reshape(-1, 3)

        types, elem_tags, node_tags_elem = gmsh.model.mesh.getElements(2)
        TRI3_TYPE = 2

        T_tags = None
        for etype, e_tags, conn in zip(types, elem_tags, node_tags_elem):
            if etype == TRI3_TYPE:
                n_elems = len(e_tags)
                if n_elems == 0:
                    continue
                nper = len(conn) // n_elems  # should be 3
                T_tags = np.asarray(conn, dtype=np.int64).reshape(-1, nper)
                break

        if T_tags is None:
            gmsh.finalize()
            raise RuntimeError("No TRI3 elements found.")

        tag_to_idx = {int(tag): i for i, tag in enumerate(node_tags.tolist())}
        T = np.vectorize(tag_to_idx.get, otypes=[np.int64])(T_tags)
        dict_tria = {"P": P3, "connectivity" : T}
        self.trias.append(dict_tria)

        gmsh.finalize()
        """


        lc_block1 = 0.01

        # Block 2 sizing (your old settings)
        lc_inner = 0.3
        lc_outer = 3.2
        d1       = 0.3

        algo = 6  # 2D meshing algorithm

        gmsh.initialize()
        gmsh.model.add("two_blocks_unstructured")
        gmsh.option.setNumber("Mesh.Algorithm", algo)

        # ---- Block 1: polygon surface ----
        p1 = [gmsh.model.occ.addPoint(x, y, 0.0) for (x, y) in poly_pts]
        l1 = [gmsh.model.occ.addLine(p1[i], p1[(i + 1) % len(p1)]) for i in range(len(p1))]
        w1 = gmsh.model.occ.addWire(l1)
        surf1 = gmsh.model.occ.addPlaneSurface([w1])

        # ---- Block 2: annulus surface (outer with inner hole) ----
        p_out = [gmsh.model.occ.addPoint(x, y, 0.0) for (x, y) in outer_airfoil_line]
        l_out = [gmsh.model.occ.addLine(p_out[i], p_out[(i + 1) % len(p_out)]) for i in range(len(p_out))]
        w_out = gmsh.model.occ.addWire(l_out)

        p_in = [gmsh.model.occ.addPoint(x, y, 0.0) for (x, y) in inner_airfoil_line]
        l_in = [gmsh.model.occ.addLine(p_in[i], p_in[(i + 1) % len(p_in)]) for i in range(len(p_in))]
        w_in = gmsh.model.occ.addWire(l_in)

        surf2 = gmsh.model.occ.addPlaneSurface([w_out, w_in])

        gmsh.model.occ.synchronize()

        # OPTIONAL (recommended if blocks touch/share boundaries):
        # makes conforming mesh if they overlap/touch
        gmsh.model.occ.fragment([(2, surf1), (2, surf2)], [])
        gmsh.model.occ.synchronize()


        # =======================
        # Mesh size fields (combine with Min)
        # =======================

        # --- Block 2 grading from inner boundary curves (Distance -> Threshold) ---
        f_dist = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(f_dist, "CurvesList", l_in)    # grade from inner curves
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

        # =======================
        # Extract global TRI3 mesh (same as you did)
        # =======================
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

        # gmsh.fltk.run()   # enable if you want to see it
        gmsh.finalize()