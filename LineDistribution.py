import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline  # needs SciPy
class LineDistribution:
    @staticmethod    
    def plot_lines(lines, colors=None, markers=None, labels=None):
        """
        Plot multiple lines, each defined by a list of (x, y) points.

        Parameters:
            lines (list of list of tuple): List of lines, where each line is a list of (x, y) points.
            colors (list of str): List of colors for each line.
            markers (list of str): List of marker styles for each line.
            labels (list of str): List of labels for each line.
        """
        if not lines:
            raise ValueError("No lines to plot.")

        num_lines = len(lines)

        # Default styling
        if colors is None:
            colors = ['blue', 'red', 'green', 'orange', 'purple', 'black'] * (num_lines // 6 + 1)
        if markers is None:
            markers = ['o', 's', '^', 'd', 'x', '*'] * (num_lines // 6 + 1)
        if labels is None:
            labels = [f'Line {i+1}' for i in range(num_lines)]

        plt.figure()
        for i, line in enumerate(lines):
            if not line:
                continue
            x, y = zip(*line)
            plt.plot(x, y, color=colors[i], marker=markers[i], label=labels[i])

        #plt.gca().set_aspect('equal')
        plt.grid(True)
        plt.legend()
        plt.show()
    
    @staticmethod
    def divide_line_by_reference(start_point, end_point, line_ref):
        """
        Create a new line from start_point to end_point,
        divided in the same proportional segments as line_ref.

        Parameters:
            start_point (tuple): (x, y) starting point of the new line.
            end_point (tuple): (x, y) ending point of the new line.
            line_ref (list of tuple): reference line to copy proportions from.

        Returns:
            list of tuple: New line points in the same format as line_ref.
        """

        # Convert reference line to numpy array
        ref = np.array(line_ref)
        segment_lengths = np.linalg.norm(np.diff(ref, axis=0), axis=1)
        total_segments = len(segment_lengths)

        if np.sum(segment_lengths) == 0:
            raise ValueError("Reference line has zero length.")

        # Cumulative proportions (0 to 1, inclusive)
        proportions = np.cumsum(np.insert(segment_lengths, 0, 0)) / np.sum(segment_lengths)

        # Direction vector from start to end
        start = np.array(start_point)
        end = np.array(end_point)
        direction = end - start
        total_length = np.linalg.norm(direction)

        if total_length == 0:
            raise ValueError("Start and end points are the same.")

        unit_direction = direction / total_length

        # Generate new points along the direction
        new_points = [start + prop * total_length * unit_direction for prop in proportions]

        return [tuple(pt) for pt in new_points]


    @staticmethod
    def divide_line(start, end, n):
        """
        Divide the line between `start` and `end` into `n` equal segments (n+1 points total).

        Parameters:
            start: tuple (x0, y0)
            end: tuple (x1, y1)
            n: int, number of segments

        Returns:
            List of (x, y) tuples including both start and end points.
        """
        x_vals = np.linspace(start[0], end[0], n + 1)
        y_vals = np.linspace(start[1], end[1], n + 1)
        return list(zip(x_vals, y_vals))
    
    @staticmethod
    def parallel_polyline_through_point(line, through):
        """
        Parallel to a straight (colinear) polyline that passes through `through`.
        Returns a list of shifted points.
        """
        P = np.asarray(line, float)
        p0, p1 = P[0], P[-1]
        q = np.array(through, float)

        t = p1 - p0
        t = t / np.linalg.norm(t)
        n = np.array([-t[1], t[0]])

        d = np.dot(q - p0, n)
        shift = d * n

        return [tuple(pt + shift) for pt in P]
    
    
    @staticmethod
    def arc_with_ref_spacing(p1, p2, ref_polyline, radius, direction="ccw", long_arc=False):
        """
        Return points on a circular arc of given `radius` passing through p1 -> p2,
        with spacing proportions matching `ref_polyline`. Includes both endpoints.

        Parameters
        ----------
        p1, p2 : (x, y)
            Endpoints of the chord.
        ref_polyline : array-like of shape (N, 2)
            Reference polyline whose segment-length fractions define the spacing.
        radius : float
            Circle radius (must satisfy radius >= ||p2 - p1|| / 2).
        direction : {"ccw", "cw"}
            Orientation along the arc from p1 to p2.
        long_arc : bool
            False -> minor arc (<= π), True -> major arc (> π).

        Returns
        -------
        list[(x, y)] of length N
        """

        ref = np.asarray(ref_polyline, float)
        if ref.ndim != 2 or ref.shape[1] != 2:
            raise ValueError("ref_polyline must be an array/list of (x, y) points")
        N = len(ref)
        if N < 2:
            return [tuple(p1), tuple(p2)]

        p1 = np.asarray(p1, float)
        p2 = np.asarray(p2, float)
        v = p2 - p1
        d = float(np.linalg.norm(v))
        if d == 0:
            raise ValueError("p1 and p2 must be distinct")

        R = float(radius)
        if R < d * 0.5 - 1e-14:
            raise ValueError(f"radius too small: need R >= d/2 = {0.5*d:g}")

        # Midpoint and unit chord/normal
        M = 0.5 * (p1 + p2)
        u = v / d
        n = np.array([-u[1], u[0]])  # rotate 90° CCW

        # Centers on perpendicular bisector: M ± h n
        # with h = sqrt(R^2 - (d/2)^2)
        h_sq = R*R - (0.5*d)*(0.5*d)
        h = 0.0 if h_sq < 0 else float(np.sqrt(h_sq))

        centers = [M + h*n]
        if h > 1e-15:  # two distinct centers unless R == d/2
            centers.append(M - h*n)

        def wrap_to_pi(a):
            """Wrap angle to (-pi, pi]."""
            return (a + np.pi) % (2*np.pi) - np.pi

        def delta_for_center(C, dir_ccw: bool):
            # angles of p1, p2 around center C
            a1 = np.arctan2(p1[1] - C[1], p1[0] - C[0])
            a2 = np.arctan2(p2[1] - C[1], p2[0] - C[0])
            d_raw = wrap_to_pi(a2 - a1)

            if dir_ccw:
                # want CCW: positive delta; add 2π if needed
                if d_raw <= 0:
                    d_raw += 2*np.pi
            else:
                # want CW: negative delta; subtract 2π if needed
                if d_raw >= 0:
                    d_raw -= 2*np.pi

            # minor arc has |delta| <= π (within tolerance)
            is_long = (abs(d_raw) > np.pi + 1e-12)
            return d_raw, a1, is_long

        want_ccw = (str(direction).lower() == "ccw")

        # Evaluate both candidate centers; pick one that matches long_arc flag
        best = None
        for C in centers:
            delta, a1, is_long = delta_for_center(C, want_ccw)
            score = (is_long == bool(long_arc))  # exact match preferred
            if best is None or score:            # prefer matching long/minor
                best = (C, delta, a1, is_long)
                if score:
                    break

        C, delta, a1, _ = best

        # Cumulative-length fractions from reference (0..1)
        seg = np.diff(ref, axis=0)
        seg_len = np.linalg.norm(seg, axis=1)
        total = float(seg_len.sum())
        if not np.isfinite(total) or total <= 0:
            t = np.linspace(0.0, 1.0, N)  # fallback: uniform
        else:
            s = np.insert(np.cumsum(seg_len), 0, 0.0)
            t = s / total

        # Map fractions to arc angles and build points
        angles = a1 + delta * t
        xs = C[0] + R * np.cos(angles)
        ys = C[1] + R * np.sin(angles)
        return [tuple(pt) for pt in np.column_stack([xs, ys])]

 
    @staticmethod
    def line_intersection(
        p1, p2,   # line 1 through p1->p2
        p3, p4,   # line 2 through p3->p4
    ) :
        
        """
        Returns the intersection point of the two lines (or segments if as_segment=True).
        If lines are parallel (no unique intersection) returns None.
        If segments are collinear and overlapping, returns None (not a single point).

        Coordinates are floats; use `eps` to control numeric tolerance.
        """

        def _cross(ax: float, ay: float, bx: float, by: float) -> float:
            return ax*by - ay*bx

        x1,y1 = p1; x2,y2 = p2
        x3,y3 = p3; x4,y4 = p4

        r = (x2 - x1, y2 - y1)
        s = (x4 - x3, y4 - y3)

        rxs = _cross(r[0], r[1], s[0], s[1])
        q_p = (x3 - x1, y3 - y1)


        # Unique intersection exists for infinite lines
        t = _cross(q_p[0], q_p[1], s[0], s[1]) / rxs

        # If treating as segments, require the intersection to lie within both [0,1]

        ix = x1 + t * r[0]
        iy = y1 + t * r[1]
        return (ix, iy)



    @staticmethod
    def dist_arc_with_ref_spacing(
        p1, p2,
        center, radius,
        ref_polyline,
        direction="ccw",
        alpha=1.0,          # 0 = uniform, 1 = ref-based
        eps: float = 1e-6,
    ):
        """
        Return points on the circular arc from p1 to p2 with spacing given by a
        blend between uniform spacing and the cumulative-length fractions of
        `ref_polyline`. Includes both endpoints.

        - `center` and `radius` define the circle.
        - `direction` is "ccw" or "cw" for the sweep from p1 to p2.
        - `alpha` in [0, 1]: 0 → purely uniform spacing, 1 → purely ref-based spacing.
        - If ref_polyline has <2 points or zero total length, fallback to uniform spacing.
        """
        # --- validate / coerce inputs ---
        C = np.asarray(center, dtype=float)
        p1 = np.asarray(p1, dtype=float)
        p2 = np.asarray(p2, dtype=float)
        R = float(radius)

        if R <= 0:
            raise ValueError("radius must be positive")

        # Clamp alpha to [0, 1] just in case
        alpha = float(alpha)
        if alpha < 0.0:
            alpha = 0.0
        elif alpha > 1.0:
            alpha = 1.0

        # Reference polyline: array of (x, y)
        ref = np.asarray(ref_polyline, dtype=float)
        if ref.ndim != 2 or ref.shape[1] != 2:
            raise ValueError("ref_polyline must be an array/list of (x, y) points")
        N = len(ref)
        if N < 2:
            # trivial: just endpoints
            return [tuple(p1), tuple(p2)]

        # --- start/end angles ---
        a1 = np.arctan2(p1[1] - C[1], p1[0] - C[0])
        a2 = np.arctan2(p2[1] - C[1], p2[0] - C[0])

        two_pi = 2.0 * np.pi
        # CCW delta in [0, 2π)
        delta_ccw = (a2 - a1) % two_pi
        # CW delta in (-2π, 0]
        delta_cw = -((a1 - a2) % two_pi)

        if direction.lower() == "ccw":
            delta = delta_ccw
        elif direction.lower() == "cw":
            delta = delta_cw
        else:
            raise ValueError('direction must be "ccw" or "cw"')

        # --- uniform fractions (N points) ---
        t_uniform = np.linspace(0.0, 1.0, N)

        # --- cumulative-length fractions from the reference polyline ---
        seg = np.diff(ref, axis=0)
        seg_len = np.linalg.norm(seg, axis=1)
        total = float(seg_len.sum())

        if total <= eps:
            # Fallback: no meaningful ref spacing → use uniform only
            t = t_uniform
        else:
            s = np.insert(np.cumsum(seg_len), 0, 0.0)  # 0, l1, l1+l2, ...
            t_ref = s / total

            # --- blend between uniform and ref-based fractions ---
            if alpha == 0.0:
                t = t_uniform
            elif alpha == 1.0:
                t = t_ref
            else:
                t = (1.0 - alpha) * t_uniform + alpha * t_ref

        # --- map fractions to arc angles and build points ---
        angles = a1 + delta * t
        xs = C[0] + R * np.cos(angles)
        ys = C[1] + R * np.sin(angles)

        polyLine = [tuple(pt) for pt in np.column_stack([xs, ys])]
        polyLine[0] = tuple(p1)
        polyLine[-1] = tuple(p2)
        return polyLine

    @staticmethod
    def dist_arc_with_normals_spline(
        p1, p2,
        center, radius,
        ref_polyline,
        eps: float = 1e-9,
    ):
        """
        Build a circular polyline on the circle (center, radius) such that:

        - First point is exactly p1
        - Last point is exactly p2
        - For interior points, we fit a spline through ref_polyline, compute
          smooth tangents, turn them into normals, and intersect those normals
          with the circle.

        The number of output points == len(ref_polyline).
        """
        C = np.asarray(center, dtype=float)
        R = float(radius)
        if R <= 0.0:
            raise ValueError("radius must be positive")

        p1 = np.asarray(p1, dtype=float)
        p2 = np.asarray(p2, dtype=float)

        ref = np.asarray(ref_polyline, dtype=float)
        if ref.ndim != 2 or ref.shape[1] != 2:
            raise ValueError("ref_polyline must be an array/list of (x, y) points")

        N = len(ref)
        if N == 0:
            return []
        if N == 1:
            # degenerate: just return p1 (or p2, but they should coincide anyway)
            return [tuple(p1)]

        # --- 1) arc-length parameter s in [0, 1] for ref_polyline ---
        seg = np.diff(ref, axis=0)
        seg_len = np.linalg.norm(seg, axis=1)
        s = np.concatenate(([0.0], np.cumsum(seg_len)))
        total_len = float(s[-1])

        if total_len < eps:
            # all ref points essentially identical → just copy p1..p2 linearly
            out = []
            for t in np.linspace(0.0, 1.0, N):
                q = (1.0 - t) * p1 + t * p2
                out.append((float(q[0]), float(q[1])))
            return out

        s /= total_len

        x = ref[:, 0]
        y = ref[:, 1]

        # --- 2) parametric cubic splines x(s), y(s) ---
        cs_x = CubicSpline(s, x, bc_type="natural")
        cs_y = CubicSpline(s, y, bc_type="natural")

        # --- 3) tangents from derivatives at the same s-values ---
        dx = cs_x(s, 1)
        dy = cs_y(s, 1)
        tangents = np.column_stack([dx, dy])

        # normalize tangents; fallback to radial if degenerate
        tangents_norm = np.linalg.norm(tangents, axis=1)
        for i in range(N):
            if tangents_norm[i] < eps:
                v = ref[i] - C
                nv = np.linalg.norm(v)
                if nv < eps:
                    tangents[i] = np.array([1.0, 0.0])
                else:
                    tangents[i] = v / nv
            else:
                tangents[i] /= tangents_norm[i]

        # --- 4) normals by +90° rotation ---
        normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])

        # --- 5) build circular polyline ---
        result = [None] * N
        result[0] = (float(p1[0]), float(p1[1]))
        result[-1] = (float(p2[0]), float(p2[1]))

        for i in range(1, N - 1):
            P = ref[i]
            n = normals[i]

            # Check if ref point is already on circle
            vPC = P - C
            dPC = np.linalg.norm(vPC)
            if abs(dPC - R) <= eps:
                Q = P
                result[i] = (float(Q[0]), float(Q[1]))
                continue

            # Line-circle intersection: |P + s*n - C|^2 = R^2, |n|=1
            b = 2.0 * np.dot(n, vPC)
            c = np.dot(vPC, vPC) - R * R
            disc = b * b - 4.0 * c

            if disc < 0.0:
                # fallback: radial projection from center
                if dPC < eps:
                    Q = C + np.array([R, 0.0])
                else:
                    Q = C + R * vPC / dPC
                result[i] = (float(Q[0]), float(Q[1]))
                continue

            sqrt_disc = np.sqrt(disc)
            s1 = (-b + sqrt_disc) * 0.5
            s2 = (-b - sqrt_disc) * 0.5
            s_best = s1 if abs(s1) < abs(s2) else s2

            Q = P + s_best * n
            result[i] = (float(Q[0]), float(Q[1]))

        return result
    
    @staticmethod    
    def circle_center_radius_from_3pts(p1, p2, p3, eps: float = 1e-12):
        """
        Return (center_x, center_y), radius of the unique circle through three non-collinear points.
        Raises ValueError if points are (near-)collinear.
        """
        x1, y1 = map(float, p1)
        x2, y2 = map(float, p2)
        x3, y3 = map(float, p3)

        # Linear system for center (perpendicular bisectors)
        A = np.array([[2*(x2 - x1), 2*(y2 - y1)],
                    [2*(x3 - x1), 2*(y3 - y1)]], dtype=float)
        b = np.array([(x2**2 + y2**2) - (x1**2 + y1**2),
                    (x3**2 + y3**2) - (x1**2 + y1**2)], dtype=float)

        det = np.linalg.det(A)
        if abs(det) < eps:
            raise ValueError("Points are collinear or nearly collinear; no unique circle.")

        cx, cy = np.linalg.solve(A, b)
        R = float(np.hypot(x1 - cx, y1 - cy))
        return (float(cx), float(cy)), R
        
    @staticmethod
    def dist_arc_with_uniform_spacing(
        p1, p2,
        center, radius,
        segments,          # mandatory: number of equal-length segments (>=1)
        direction="ccw",
        eps: float = 1e-6
    ):
        """
        Return points on the circular arc from p1 to p2 with uniform arc-length spacing.
        Includes both endpoints. Uses exactly `segments + 1` points.

        Args:
            p1, p2: endpoints on the circle (x, y)
            center: circle center (x, y)
            radius: circle radius (>0)
            segments: number of equal-length segments along the arc (>=1)
            direction: "ccw" or "cw" sweep from p1 to p2
            eps: tolerance
        """
        # --- validate / coerce inputs ---
        if segments is None:
            raise ValueError("segments is required and must be >= 1")
        if int(segments) < 1:
            raise ValueError("segments must be an integer >= 1")
        segments = int(segments)

        C = np.asarray(center, dtype=float)
        p1 = np.asarray(p1, dtype=float)
        p2 = np.asarray(p2, dtype=float)
        R = float(radius)
        if R <= 0:
            raise ValueError("radius must be positive")

        # --- angles ---
        a1 = np.arctan2(p1[1] - C[1], p1[0] - C[0])
        a2 = np.arctan2(p2[1] - C[1], p2[0] - C[0])

        two_pi = 2.0 * np.pi
        delta_ccw = (a2 - a1) % two_pi      # [0, 2π)
        delta_cw  = -((a1 - a2) % two_pi)   # (-2π, 0]
        if direction.lower() == "ccw":
            delta = delta_ccw
        elif direction.lower() == "cw":
            delta = delta_cw
        else:
            raise ValueError('direction must be "ccw" or "cw"')

        # Degenerate sweep → just endpoints
        if abs(delta) <= eps:
            return [tuple(p1), tuple(p2)]

        # --- uniform fractions and points ---
        n_pts = segments + 1
        t = np.linspace(0.0, 1.0, n_pts)
        angles = a1 + delta * t
        xs = C[0] + R * np.cos(angles)
        ys = C[1] + R * np.sin(angles)

        polyLine = [tuple(pt) for pt in np.column_stack([xs, ys])]
        polyLine[0]  = tuple(p1)  # pin exact endpoints
        polyLine[-1] = tuple(p2)
        return polyLine

      
    @staticmethod
    def closed_left_U(bottom_right, vert_len, horiz_len, ref_polyline=None, n_segments=100):
        """
        Build a CLOSED left-sided U with a right closing edge, starting at bottom_right:
        1) go left by horiz_len (bottom straight),
        2) semicircle of diameter vert_len (bottom->top),
        3) go right by horiz_len (top straight to top_right),
        4) go straight down to bottom_right (right edge).

        Spacing:
        - If ref_polyline is provided: follow its arc-length fractions.
        - Else: uniform fractions with n_segments segments (returns n_segments+1 points).

        Returns
        -------
        list[(x, y)]  # closed: first point == last point
        """
        BR = np.asarray(bottom_right, float)
        H  = float(vert_len)
        L  = float(horiz_len)
        if H <= 0 or L < 0:
            raise ValueError("vert_len must be > 0 and horiz_len >= 0")

        # Geometry
        TR = np.array([BR[0], BR[1] + H])   # top-right
        Cx = BR[0] - L                      # semicircle center x (left of right edge)
        Cy = BR[1] + 0.5 * H
        R  = 0.5 * H

        # Segment lengths
        L_bot, L_arc, L_top, L_right = L, np.pi * R, L, H
        L_tot = L_bot + L_arc + L_top + L_right

        # Fractions t in [0,1]
        if ref_polyline is not None:
            ref = np.asarray(ref_polyline, float)
            if ref.ndim != 2 or ref.shape[1] != 2:
                raise ValueError("ref_polyline must be an (N,2) array-like")
            seg_len = np.linalg.norm(np.diff(ref, axis=0), axis=1) if len(ref) >= 2 else np.array([1.0])
            if seg_len.sum() <= 0:
                t = np.linspace(0.0, 1.0, len(ref) or 2)
            else:
                s = np.insert(np.cumsum(seg_len), 0, 0.0)
                t = s / s[-1]
        else:
            if n_segments < 1:
                raise ValueError("n_segments must be >= 1 when ref_polyline is None")
            t = np.linspace(0.0, 1.0, n_segments + 1)

        # Walk along the path by arclength
        out = []
        for ind, frac in enumerate(t):
            s_along = frac * L_tot
            if s_along <= L_bot:
                # bottom straight: move LEFT from BR
                p = np.array([BR[0] - s_along, BR[1]])
            elif s_along <= L_bot + L_arc:
                # left semicircle: θ: -π/2 -> +π/2 (bottom -> top)
                s_arc = s_along - L_bot
                tau   = s_arc / L_arc
                theta = -np.pi/2 + np.pi * tau
                x = Cx - R * np.cos(theta)
                y = Cy + R * np.sin(theta)
                p = np.array([x, y])
            elif s_along <= L_bot + L_arc + L_top:
                # top straight: move RIGHT to TR
                s_top = s_along - (L_bot + L_arc)
                p = np.array([Cx + s_top, TR[1]])
                s_along_next = t[ind+1]*L_tot
                if s_along_next > L_bot + L_arc + L_top:
                    p = np.array([TR[0], TR[1]])
            else:
                # right closing edge: move DOWN to BR
                s_right = s_along - (L_bot + L_arc + L_top)
                p = np.array([TR[0], TR[1] - s_right])

            out.append((float(p[0]), float(p[1])))

        # ensure closure
        if out[0] != out[-1]:
            out.append(out[0])
        return out

    @staticmethod
    def arc_like_spline_from_2pts_2tangents(
        p1, t1_pA, t1_pB,
        p2, t2_pA, t2_pB,
        n_points: int = 50,
        handle_scale: float = 1.0,
        eps: float = 1e-12,
    ):
        """
        Construct an arc-like cubic Bézier curve between two points with prescribed
        tangent directions at the endpoints, sampled ~uniformly in arc length.

        Returns a polyline: list of (x, y) tuples of length n_points.
        """

        p1 = np.asarray(p1, dtype=float)
        p2 = np.asarray(p2, dtype=float)
        t1_pA = np.asarray(t1_pA, dtype=float)
        t1_pB = np.asarray(t1_pB, dtype=float)
        t2_pA = np.asarray(t2_pA, dtype=float)
        t2_pB = np.asarray(t2_pB, dtype=float)

        # Tangent directions
        t1 = t1_pB - t1_pA
        t2 = t2_pB - t2_pA

        n1 = np.linalg.norm(t1)
        n2 = np.linalg.norm(t2)
        if n1 < eps or n2 < eps:
            raise ValueError("Tangent direction is degenerate (zero length).")

        t1 /= n1
        t2 /= n2

        # Chord
        chord = p2 - p1
        chord_len = np.linalg.norm(chord)
        if chord_len < eps:
            raise ValueError("Endpoints are too close or identical.")

        base_handle_len = chord_len / 3.0
        L1 = handle_scale * base_handle_len
        L2 = handle_scale * base_handle_len

        # Bézier control points
        P0 = p1
        P3 = p2
        P1 = P0 + L1 * t1
        P2 = P3 - L2 * t2

        def bezier_eval(t):
            """Evaluate cubic Bézier at scalar or array t."""
            t = np.asarray(t, dtype=float)
            one_minus_t = 1.0 - t
            B0 = one_minus_t**3
            B1 = 3.0 * one_minus_t**2 * t
            B2 = 3.0 * one_minus_t * t**2
            B3 = t**3
            return (
                B0[:, None] * P0[None, :] +
                B1[:, None] * P1[None, :] +
                B2[:, None] * P2[None, :] +
                B3[:, None] * P3[None, :]
            )

        # ---- Reparameterize to ~uniform arc length ----
        # Oversample in t, compute cumulative chord length, then invert.
        n_sample = max(10 * n_points, 200)  # dense sampling
        t_sample = np.linspace(0.0, 1.0, n_sample)
        pts_sample = bezier_eval(t_sample)

        # chord-length along the sampled curve
        diffs = np.diff(pts_sample, axis=0)
        seg_lens = np.linalg.norm(diffs, axis=1)
        s = np.concatenate(([0.0], np.cumsum(seg_lens)))
        total_len = s[-1]

        if total_len < eps:
            # curve is effectively a point/very short line
            return [tuple(p1)] * n_points

        # normalize arc-length to [0,1]
        s /= total_len

        # target equally spaced arc-lengths
        s_target = np.linspace(0.0, 1.0, n_points)

        # for each s_target, find corresponding position by linear interpolation
        pts_uniform = []
        j = 0
        for st in s_target:
            # advance j until s[j] <= st <= s[j+1]
            while j < len(s) - 2 and s[j+1] < st:
                j += 1
            s0, s1 = s[j], s[j+1]
            p0, p1_ = pts_sample[j], pts_sample[j+1]

            if s1 - s0 < eps:
                alpha = 0.0
            else:
                alpha = (st - s0) / (s1 - s0)

            p = (1.0 - alpha) * p0 + alpha * p1_
            pts_uniform.append((float(p[0]), float(p[1])))

        # ensure exact endpoints (avoid tiny numerical drift)
        pts_uniform[0] = (float(P0[0]), float(P0[1]))
        pts_uniform[-1] = (float(P3[0]), float(P3[1]))

        return pts_uniform


    @staticmethod
    def arc_like_spline_from_2pts_2tangents_ref_spacing(
        p1, t1_pA, t1_pB,
        p2, t2_pA, t2_pB,
        ref_polyline,
        handle_scale: float = 1.0,
        eps: float = 1e-12,
    ):
        """
        Construct an arc-like cubic Bézier curve between two points with prescribed
        tangent directions at the endpoints, with point spacing proportional to a
        reference polyline.

        Inputs
        ------
        p1 : (x, y) start point on the curve
        t1_pA, t1_pB : two points defining the tangent line at p1
        p2 : (x, y) end point on the curve
        t2_pA, t2_pB : two points defining the tangent line at p2
        ref_polyline : list/array of (x, y) points; its chord-length fractions
                    determine the spacing along the Bézier curve
        handle_scale : scales handle length; >1 more curved, <1 flatter
        eps : small tolerance

        Returns
        -------
        pts : list[(x, y)] of length len(ref_polyline),
            distributed along the Bézier with spacing proportional
            to the reference polyline's segment lengths.
        """

        # --- basic checks + array conversion ---
        p1 = np.asarray(p1, dtype=float)
        p2 = np.asarray(p2, dtype=float)
        t1_pA = np.asarray(t1_pA, dtype=float)
        t1_pB = np.asarray(t1_pB, dtype=float)
        t2_pA = np.asarray(t2_pA, dtype=float)
        t2_pB = np.asarray(t2_pB, dtype=float)

        ref_polyline = np.asarray(ref_polyline, dtype=float)
        if ref_polyline.ndim != 2 or ref_polyline.shape[1] != 2:
            raise ValueError("ref_polyline must be an array/list of (x, y) points.")
        if ref_polyline.shape[0] < 2:
            raise ValueError("ref_polyline must contain at least 2 points.")

        n_points = ref_polyline.shape[0]

        # --- tangent directions ---
        t1 = t1_pB - t1_pA
        t2 = t2_pB - t2_pA

        n1 = np.linalg.norm(t1)
        n2 = np.linalg.norm(t2)
        if n1 < eps or n2 < eps:
            raise ValueError("Tangent direction is degenerate (zero length).")

        t1 /= n1
        t2 /= n2

        # --- chord + handle lengths ---
        chord = p2 - p1
        chord_len = np.linalg.norm(chord)
        if chord_len < eps:
            raise ValueError("Endpoints are too close or identical.")

        base_handle_len = chord_len / 3.0
        L1 = handle_scale * base_handle_len
        L2 = handle_scale * base_handle_len

        # --- Bézier control points ---
        P0 = p1
        P3 = p2
        P1 = P0 + L1 * t1
        P2 = P3 - L2 * t2

        def bezier_eval(t):
            """Evaluate cubic Bézier at array of t in [0,1]."""
            t = np.asarray(t, dtype=float)
            one_minus_t = 1.0 - t
            B0 = one_minus_t**3
            B1 = 3.0 * one_minus_t**2 * t
            B2 = 3.0 * one_minus_t * t**2
            B3 = t**3
            return (
                B0[:, None] * P0[None, :] +
                B1[:, None] * P1[None, :] +
                B2[:, None] * P2[None, :] +
                B3[:, None] * P3[None, :]
            )

        # --- 1) build normalized arc-length s_ref from reference polyline ---
        diffs_ref = np.diff(ref_polyline, axis=0)
        seg_lens_ref = np.linalg.norm(diffs_ref, axis=1)
        s_ref = np.concatenate(([0.0], np.cumsum(seg_lens_ref)))
        total_len_ref = s_ref[-1]

        if total_len_ref < eps:
            # reference is effectively a point / degenerate line
            return [tuple(p1)] * n_points

        s_ref /= total_len_ref  # normalize to [0,1]
        s_target = s_ref        # these are the fractions we want to mimic

        # --- 2) build s(t) for Bézier via oversampling ---
        n_sample = max(10 * n_points, 400)
        t_sample = np.linspace(0.0, 1.0, n_sample)
        pts_sample = bezier_eval(t_sample)

        diffs = np.diff(pts_sample, axis=0)
        seg_lens = np.linalg.norm(diffs, axis=1)
        s = np.concatenate(([0.0], np.cumsum(seg_lens)))
        total_len = s[-1]

        if total_len < eps:
            return [tuple(p1)] * n_points

        s /= total_len

        # --- 3) map each s_target to a parameter tau along the Bézier ---
        pts_out = []
        j = 0
        for st in s_target:
            # clamp to [0,1] in case of tiny numerical drift
            st = min(max(st, 0.0), 1.0)

            # find segment such that s[j] <= st <= s[j+1]
            while j < len(s) - 2 and s[j+1] < st:
                j += 1

            s0, s1 = s[j], s[j+1]
            t0, t1_ = t_sample[j], t_sample[j+1]

            if s1 - s0 < eps:
                tau = t0
            else:
                alpha = (st - s0) / (s1 - s0)
                tau = t0 + alpha * (t1_ - t0)

            pt = bezier_eval(np.array([tau]))[0]
            pts_out.append((float(pt[0]), float(pt[1])))

        # ensure exact endpoints
        pts_out[0] = (float(P0[0]), float(P0[1]))
        pts_out[-1] = (float(P3[0]), float(P3[1]))

        return pts_out
