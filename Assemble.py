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
        #sys.exit()

        #sys.exit()

        te_upper_mesh = BlockMesh()
        boundaries_lower = te_mesh.getLine(number=0, direction='v')
        boundaries_left = mesh.getLine(number=-1, direction='v')
        top_left = boundaries_left[-1]
        bottom_right = boundaries_lower[-1]
        top_right = (bottom_right[0], top_left[1])
        boundaries_right = LineDistribution.divide_line_by_reference(bottom_right, top_right, boundaries_left)
        boundaries_upper = LineDistribution.divide_line_by_reference(top_left, top_right, boundaries_lower)
        boundary = [boundaries_lower, boundaries_upper, boundaries_left, boundaries_right]
        te_upper_mesh.transfinite(boundary=boundary)


        self.blocks.append(te_upper_mesh)

        te_lower_mesh = BlockMesh()

        boundaries_upper = te_mesh.getLine(number=-1, direction='v')
        boundaries_left = mesh.getLine(number=0, direction='v')
        bottom_left = boundaries_left[-1]
        top_right = boundaries_upper[-1]
        bottom_right = (top_right[0], bottom_left[1])
        boundaries_right = LineDistribution.divide_line_by_reference(top_right, bottom_right, boundaries_left)
        boundaries_lower = LineDistribution.divide_line_by_reference(bottom_left, bottom_right, boundaries_upper)

        boundaries_left.reverse()
        boundaries_right.reverse()       
        boundary = [boundaries_lower, boundaries_upper, boundaries_left, boundaries_right]
        #boundary = [boundaries_upper, boundaries_lower, boundaries_right, boundaries_left]
        #boundary = [boundaries_left, boundaries_right, boundaries_lower, boundaries_upper]
        #LineDistribution.plot_lines(boundary)
        te_lower_mesh.transfinite(boundary=boundary)

        self.blocks.append(te_lower_mesh)

        

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
