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
    def blended_proportional_uniform(start_point, end_point, line_ref, alpha=0.8):
        """
        Create a new line from start_point to end_point where the segment
        lengths are a blend of:
            - proportional-to-reference (line_ref)
            - uniform spacing
        using weight alpha in [0,1].

        The number of points equals len(line_ref).

        Args:
            start_point (tuple): (x, y)
            end_point (tuple): (x, y)
            line_ref (list[tuple]): reference line defining segment proportions
            alpha (float): blend factor (0 → uniform, 1 → proportional)

        Returns:
            list[tuple]: new blended line points (same count as line_ref)
        """
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be in [0, 1].")

        # Vector direction and total length
        p1 = np.asarray(start_point, dtype=float)
        p2 = np.asarray(end_point, dtype=float)
        direction = p2 - p1
        total_length = np.linalg.norm(direction)
        if total_length == 0:
            raise ValueError("Start and end points are identical.")

        # Reference-based segment lengths
        ref = np.asarray(line_ref, dtype=float)
        if ref.ndim != 2 or ref.shape[1] != 2 or len(ref) < 2:
            raise ValueError("line_ref must have at least two (x, y) points.")

        seg_ref = np.linalg.norm(np.diff(ref, axis=0), axis=1)
        seg_ref = seg_ref / np.sum(seg_ref) * total_length

        # Uniform segment lengths (equal division)
        n_seg = len(seg_ref)
        seg_uni = np.full(n_seg, total_length / n_seg)

        # Blended segment lengths
        seg_blend = alpha * seg_ref + (1 - alpha) * seg_uni

        # Build cumulative positions along p1→p2
        cum_dist = np.insert(np.cumsum(seg_blend), 0, 0.0)
        unit_dir = direction / total_length
        pts = p1 + cum_dist[:, None] * unit_dir

        return [tuple(pt) for pt in pts]
    
    @staticmethod

    def create_distributed_line(start_point, end_point, num_segments, first_segment_length):
        """
        Creates a line with points distributed to match a target first segment length.
        This function calculates the required 'power' for a polynomial distribution.

        Args:
            start_point (tuple): The (x0, y0) coordinates of the line start.
            end_point (tuple): The (x1, y1) coordinates of the line end.
            num_segments (int): The number of segments to divide the line into.
            first_segment_length (float): The desired length of the first segment.

        Returns:
            A tuple containing:
            - list: A list of (x, y) points defining the new segmented line.
            - list: A list of floats representing the length of each segment.
            Returns (None, None) on failure.
        """
        x0, y0 = start_point
        x1, y1 = end_point
        
        # --- 1. Calculate Line Length and Validate Input ---
        line_length = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
        
        if not (0 < first_segment_length < line_length):
            print(f"Error: First segment length ({first_segment_length:.2f}) must be > 0 and < total length ({line_length:.2f}).")
            return None, None
            
        # --- 2. Calculate the required 'power' using logarithms ---
        try:
            power = np.log(first_segment_length / line_length) / np.log(1 / num_segments)
        except ValueError:
            print("Error during log calculation. Check inputs.")
            return None, None
            
        # --- 3. Generate points using the calculated power ---
        new_points = []
        total_dx = x1 - x0
        total_dy = y1 - y0
        
        for i in range(num_segments + 1):
            t = i / num_segments
            warped_t = t ** power
            
            new_x = x0 + warped_t * total_dx
            new_y = y0 + warped_t * total_dy
            new_points.append((new_x, new_y))

        # --- 4. Calculate the length of each individual segment ---
        segment_lengths = []
        for i in range(len(new_points) - 1):
            p1 = new_points[i]
            p2 = new_points[i+1]
            length = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            segment_lengths.append(length)
            
        return new_points, segment_lengths

    @staticmethod
    def symmetric_grow_decay_line(start_point, end_point, h0, r=1.05, *, even=False):
        """
        Return a polyline [(x0,y0), ... , (xN,yN)] from start->end where
        segment lengths grow geometrically from h0 by ratio r up to the middle,
        then decay symmetrically back to h0. Only the points are returned.
        """
        if r <= 1.0:
            raise ValueError("r must be > 1.")
        x0, y0 = start_point
        x1, y1 = end_point
        vec = np.array([x1 - x0, y1 - y0], dtype=float)
        L = float(np.linalg.norm(vec))
        if L <= 0:
            raise ValueError("Start and end points must differ.")
        if L < 2*h0:
            raise ValueError("Total length must be at least 2*h0.")
        d = vec / L  # unit direction

        def sum_wing(k_):
            return 0.0 if k_ <= 0 else h0 * (r**k_ - 1.0) / (r - 1.0)

        # choose k (growth steps per side) if not provided
        k = int(np.floor(np.log(1.0 + L*(r - 1.0)/(2.0*h0)) / np.log(r)))
        while 2.0*sum_wing(k) > L and k > 0:
            k -= 1

        wing = [h0 * (r**m) for m in range(k)]
        S_w = sum(wing)
        remainder = L - 2.0*S_w

        if k == 0:
            seg_lengths = [L/2.0, L/2.0]  # trivial two segments
        else:
            if not even:
                seg_lengths = wing + [remainder] + wing[::-1]
            else:
                peak = wing[-1]
                left = wing.copy(); right = wing[::-1]
                left[-1]  = peak + 0.5*remainder
                right[0]  = peak + 0.5*remainder
                seg_lengths = left + right

        s = np.insert(np.cumsum(seg_lengths), 0, 0.0)  # cumulative distances
        line = [(x0 + si*d[0], y0 + si*d[1]) for si in s]
        line[-1] = (x1, y1)  # snap exact end

        return line
    
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
    def grow_to_min_length_line(start_point, h0, r, L, direction, max_steps=1_000_000):
        """
        Return a polyline [(x0,y0), ...] starting at start_point.
        Segment sizes grow geometrically: h0, h0*r, h0*r^2, ...
        Keep adding FULL segments until the accumulated length >= L (may exceed L).
        
        Args:
            start_point (tuple): (x0, y0)
            h0 (float): initial segment thickness (>0)
            r (float): growth ratio (>1)
            L (float): minimum total length to reach (>=0)
            direction (tuple): (dx, dy) non-zero direction vector
            max_steps (int): safety cap on number of segments

        Returns:
            list[(x,y)]: points from start to final (inclusive)
        """
        if h0 <= 0:  raise ValueError("h0 must be > 0.")
        if r <= 1:   raise ValueError("r must be > 1.")
        if L < 0:    raise ValueError("L must be >= 0.")

        dx, dy = map(float, direction)
        norm = np.hypot(dx, dy)
        if norm == 0:
            raise ValueError("direction must be a non-zero vector.")
        ux, uy = dx / norm, dy / norm

        x, y = map(float, start_point)
        line = [(x, y)]

        if L == 0:
            return line

        step = float(h0)
        total = 0.0
        steps = 0

        while total < L:
            if steps >= max_steps:
                raise RuntimeError("Exceeded max_steps; check r/L/h0.")
            x += step * ux
            y += step * uy
            line.append((x, y))
            total += step
            step *= r
            steps += 1

        return line

    @staticmethod
    def gp_to_ap_by_step_threshold_line(start_point, end_point, N, h0, r, step_limit, *, eps=1e-12):
        """
        Polyline with N segments (N+1 points) from start_point to end_point.
        - Use GP lengths: h0, h0*r, ... until the CURRENT GP step >= step_limit.
        - Then switch to AP starting at that last GP size; choose AP increment so
        the total sum equals the start–end distance and lengths stay non-decreasing.

        Returns: list[(x, y)] of length N+1.
        """
        if N < 1:               raise ValueError("N must be >= 1.")
        if h0 <= 0:             raise ValueError("h0 must be > 0.")
        if r <= 1.0:            raise ValueError("r must be > 1.")
        if step_limit <= 0:     raise ValueError("step_limit must be > 0.")

        x0, y0 = map(float, start_point)
        x1, y1 = map(float, end_point)
        vec = np.array([x1 - x0, y1 - y0], dtype=float)
        L = float(np.linalg.norm(vec))
        if L == 0.0:
            return [(x0, y0)]
        ux, uy = vec / L

        # Minimal feasibility: can't make N non-decreasing segments starting below h0 if total < N*h0
        if L < N * h0 - eps:
            raise ValueError(f"Line too short for N non-decreasing segments starting at h0; need ≥ {N*h0:g}, got {L:g}.")

        # --- GP count from step threshold ---
        # minimal n such that h0 * r^(n-1) >= step_limit
        n_gp_goal = int(np.ceil(1.0 + np.log(step_limit / h0) / np.log(r))) if step_limit > h0 else 1
        # ensure at least one AP segment remains
        n_gp = min(max(1, n_gp_goal), N - 1)

        def gp_sum(k):
            # sum of first k GP terms starting at h0
            return 0.0 if k == 0 else h0 * (r**k - 1.0) / (r - 1.0)

        # Back off n_gp if needed so AP increment ≥ 0 and GP sum ≤ L
        while True:
            M = N - n_gp                      # AP terms
            gp_lengths = h0 * (r ** np.arange(n_gp, dtype=float))
            Sg = gp_lengths.sum()
            a0_ap = gp_lengths[-1]            # first AP term (continuity)
            remaining = L - Sg
            # Conditions: remaining ≥ M*a0_ap (so AP inc ≥ 0) and remaining ≥ 0
            if remaining + eps >= M * a0_ap and remaining >= -eps:
                break
            n_gp -= 1
            if n_gp < 1:
                raise ValueError("Inputs inconsistent: cannot allocate GP→AP with non-decreasing lengths.")

        # --- Build final lengths ---
        lengths = list(gp_lengths)
        M = N - n_gp
        remaining = L - sum(lengths)

        if M == 1:
            # Single AP length: it equals the remainder (guaranteed ≥ a0_ap)
            lengths.append(remaining)
        else:
            # remaining = M*a0 + inc * M*(M-1)/2  -> solve inc
            a0_ap = gp_lengths[-1]
            inc = 2.0 * (remaining - M * a0_ap) / (M * (M - 1))
            # numerical guard
            if inc < 0 and inc > -1e-14:
                inc = 0.0
            ap_lengths = a0_ap + inc * np.arange(M, dtype=float)
            lengths.extend(ap_lengths.tolist())
        # tiny drift fix
        drift = L - sum(lengths)
        if abs(drift) > 1e-10:
            lengths[-1] += drift

        # --- Convert to points ---
        s = np.insert(np.cumsum(lengths), 0, 0.0)  # 0..L, N+1 points
        pts = [(x0 + si * ux, y0 + si * uy) for si in s]
        return pts
    
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
    def semicircle_arc_with_ref_spacing(p1, p2, ref_polyline, direction="ccw"):
        """
        Return points on the semicircle (minimal radius: diameter = p1–p2)
        from p1 to p2 with spacing proportions matching ref_polyline.
        Includes both endpoints. direction: "ccw" or "cw".
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
        d = np.linalg.norm(v)
        if d == 0:
            raise ValueError("p1 and p2 must be distinct")

        # Minimal circle: center at midpoint, radius = d/2
        C = 0.5 * (p1 + p2)
        R = 0.5 * d

        # Start angle at p1; semicircle delta
        a1 = np.arctan2(p1[1] - C[1], p1[0] - C[0])
        delta = np.pi if direction == "ccw" else -np.pi

        # Cumulative-length fractions from reference (0..1)
        seg = np.diff(ref, axis=0)
        seg_len = np.linalg.norm(seg, axis=1)
        total = seg_len.sum()
        if total <= 0:
            t = np.linspace(0.0, 1.0, N)          # fallback: uniform
        else:
            s = np.insert(np.cumsum(seg_len), 0, 0.0)
            t = s / total

        # Map fractions to arc angles and build points
        angles = a1 + delta * t
        xs = C[0] + R * np.cos(angles)
        ys = C[1] + R * np.sin(angles)
        return [tuple(pt) for pt in np.column_stack([xs, ys])]

    @staticmethod
    def semicircle_intersections_along_normals(p1, p2, ref_polyline, normals, direction="ccw"):
        """
        For each reference point r_i with normal n_i, return the intersection point
        of the ray r_i + t n_i (t > 0) with the outer semicircle defined by the
        minimal circle (centered at midpoint of p1–p2, radius |p2–p1|/2).

        Special handling (per request):
        - For the FIRST ref point: return p1 directly.
        - For the LAST  ref point: return p2 directly.

        Parameters
        ----------
        p1, p2 : array-like of shape (2,)
        ref_polyline : array-like of shape (N, 2)
        normals : array-like of shape (N, 2)
        direction : {"ccw","cw"}

        Returns
        -------
        pts : list[tuple[float,float]]
            Intersection points on the chosen semicircle, aligned with ref_polyline.
        """
        ref = np.asarray(ref_polyline, dtype=float)
        nrm = np.asarray(normals, dtype=float)

        if ref.ndim != 2 or ref.shape[1] != 2:
            raise ValueError("ref_polyline must be an array/list of (x, y) points")
        if nrm.shape != ref.shape:
            raise ValueError("normals must have shape (N, 2) matching ref_polyline")
        if direction not in ("ccw", "cw"):
            raise ValueError("direction must be 'ccw' or 'cw'")

        p1 = np.asarray(p1, dtype=float)
        p2 = np.asarray(p2, dtype=float)
        v = p2 - p1
        d = np.linalg.norm(v)
        if d == 0:
            raise ValueError("p1 and p2 must be distinct")

        # Circle center and radius
        C = 0.5 * (p1 + p2)
        R = 0.5 * d

        # Start angle at p1; signed semicircle sweep
        a1 = np.arctan2(p1[1] - C[1], p1[0] - C[0])
        delta = np.pi if direction == "ccw" else -np.pi

        def wrap_to_pi(a):
            return (a + np.pi) % (2*np.pi) - np.pi

        def angle_on_arc(theta):
            """Check if an angle lies on the chosen semicircle."""
            if direction == "ccw":
                off = wrap_to_pi(theta - a1)
                return (off >= -1e-12) and (off <= np.pi + 1e-12)
            else:
                off = wrap_to_pi(theta - a1)
                return (off <= 1e-12) and (off >= -np.pi - 1e-12)

        N = len(ref)
        out = [None] * N                             # <<< CHANGED: preallocate to set first/last easily

        if N == 0:
            return []
        if N == 1:
            return [tuple(p1)]                       # <<< CHANGED: degenerate case—map single point to p1

        out[0]  = (p1[0], p1[1])                     # <<< CHANGED: first point -> p1
        out[-1] = (p2[0], p2[1])                     # <<< CHANGED: last  point -> p2

        # Process only interior points: indices 1..N-2
        for idx in range(1, N-1):                    # <<< CHANGED: iterate interior indices only
            r = ref[idx]
            n = nrm[idx]

            o = r - C
            a = np.dot(n, n)
            b = 2.0 * np.dot(o, n)
            c = np.dot(o, o) - R*R

            if a == 0.0:
                out[idx] = (np.nan, np.nan)
                continue

            disc = b*b - 4*a*c
            if disc < -1e-14:
                out[idx] = (np.nan, np.nan)
                continue
            disc = max(disc, 0.0)

            sqrt_disc = np.sqrt(disc)
            t1 = (-b - sqrt_disc) / (2*a)
            t2 = (-b + sqrt_disc) / (2*a)

            # keep only forward-ray hits (positive normal direction) that lie on chosen semicircle
            candidates = []
            t_candidates = []
            for t in (t1, t2):
                if t <= 0:
                    continue
                P = r + t*n
                ang = np.arctan2(P[1] - C[1], P[0] - C[0])
                candidates.append((P[0], P[1]))
                t_candidates.append(t)


            if len(candidates) == 0:
                out[idx] = (np.nan, np.nan)
                print("found1")
            elif len(candidates) == 1:
                out[idx] = candidates[0]
                print("found")
            else:
                # Choose the one strictly in the positive normal direction with the smallest positive t
                # (closest along outward normal).
                j = int(np.argmin(t_candidates))
                out[idx] = candidates[j]

        return out
    @staticmethod
    def semicircle_intersections_along_normals_cosine(p1, p2, ref_polyline, normals):
        """
        Compute intersection points of outward normals with the minimal semicircle
        defined by diameter (p1–p2), using the law of cosines.

        For each reference point X with normal n:
            - Center O = midpoint(p1, p2)
            - Radius R = |p2 - p1| / 2
            - Angle θ = angle between XO and n
            - Distance XY = OX*cosθ + sqrt(R² - OX²*sin²θ)
            - Intersection point Y = X + XY * n̂

        Assumes each normal intersects the correct semicircle in the forward direction.
        The first and last points are directly connected to p1 and p2.
        """
        ref = np.asarray(ref_polyline, float)
        nrm = np.asarray(normals, float)

        p1 = np.asarray(p1, float)
        p2 = np.asarray(p2, float)
        O = 0.5 * (p1 + p2)
        R = 0.5 * np.linalg.norm(p2 - p1)

        N = len(ref)
        if N == 0:
            return []
        if N == 1:
            return [tuple(p1)]

        out = [None] * N
        out[0]  = (p1[0], p1[1])     # first point → p1
        out[-1] = (p2[0], p2[1])     # last  point → p2

        for i in range(1, N-1):
            X = ref[i]
            n = nrm[i]
            n_len = np.linalg.norm(n)
            if n_len == 0:
                out[i] = (np.nan, np.nan)
                continue
            n_hat = n / n_len

            XO = O - X
            OX = np.linalg.norm(XO)

            if OX == 0.0:
                # X at center: go straight out by R
                Y = X + R * n_hat
            else:
                cos_th = np.dot(XO, n_hat) / OX
                cos_th = np.clip(cos_th, -1.0, 1.0)
                sin2 = 1.0 - cos_th**2
                under = R*R - (OX*OX)*sin2
                if under < 0:
                    under = 0.0
                # forward intersection only
                XY = OX * cos_th + np.sqrt(under)
                Y = X + XY * n_hat

            out[i] = (Y[0], Y[1])

        return out
    
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
    def map_airfoil_to_semicircle_blended(p1, p2, ref_polyline, normals,
                                          direction="ccw", alphaMin=0.6, alphaMax=1.0, use_slerp=False):
        """
        Map airfoil points to a semicircle while blending spacing and normal direction.

        Combines:
          1. Arc-based spacing (preserves surface point distribution)
          2. Normal projection (improves near-wall orthogonality)
          3. Circle intersection to ensure all points lie exactly on semicircle

        Parameters
        ----------
        p1, p2 : (2,) array-like
            Endpoints of the semicircle diameter.
        ref_polyline : (N, 2) array-like
            Airfoil surface coordinates from p1 to p2.
        normals : (N, 2) array-like
            Outward surface normals at each airfoil point.
        direction : {'ccw', 'cw'}, optional
            Direction to construct the semicircle.
        alpha : float, optional
            Blend weight between 'connect-to-arc' direction and normal.
            0 → purely arc connection, 1 → purely normal.
        use_slerp : bool, optional
            If True, use spherical interpolation (smoother angular blend).

        Returns
        -------
        mapped_points : list of (x, y)
            Mapped coordinates on the semicircle.
        """

        # ---------- helpers defined inside ----------
        def _unit(v):
            n = np.linalg.norm(v)
            return v / n if n > 0 else v

        def _slerp_2d(u, v, alpha):
            dot = np.clip(np.dot(u, v), -1.0, 1.0)
            phi = np.arccos(dot)
            if phi < 1e-6:
                return _unit((1 - alpha) * u + alpha * v)
            s = np.sin(phi)
            return _unit(
                np.sin((1 - alpha) * phi) / s * u + np.sin(alpha * phi) / s * v
            )
        # ---------------------------------------------

        ref = np.asarray(ref_polyline, float)
        nrm = np.asarray(normals, float)
        if ref.ndim != 2 or ref.shape[1] != 2:
            raise ValueError("ref_polyline must be (N,2)")
        if nrm.shape != ref.shape:
            raise ValueError("normals must be same shape as ref_polyline")

        p1 = np.asarray(p1, float)
        p2 = np.asarray(p2, float)
        v = p2 - p1
        d = np.linalg.norm(v)
        if d == 0:
            raise ValueError("p1 and p2 must be distinct")

        O = 0.5 * (p1 + p2)
        R = 0.5 * d

        # Step 1: distribute target points on semicircle by reference spacing
        a1 = np.arctan2(p1[1] - O[1], p1[0] - O[0])
        delta = np.pi if direction == "ccw" else -np.pi
        seg = np.diff(ref, axis=0)
        seg_len = np.linalg.norm(seg, axis=1)
        total = seg_len.sum()
        if total <= 0:
            t = np.linspace(0.0, 1.0, len(ref))
        else:
            s = np.insert(np.cumsum(seg_len), 0, 0.0)
            t = s / total
        angles = a1 + delta * t
        Sx = O[0] + R * np.cos(angles)
        Sy = O[1] + R * np.sin(angles)
        S = np.column_stack([Sx, Sy])  # spacing targets on semicircle

        # Step 2: blend direction and find forward intersection with circle
        Y = np.empty_like(ref)
        Y[0] = p1
        Y[-1] = p2
        def variable_alpha(N, alpha_min=0.3, alpha_max=1.0):
            i = np.arange(N)
            s = i / (N - 1)
            return alpha_min + (alpha_max - alpha_min) * np.cos(np.pi * (s - 0.5))**2        
        alphas = variable_alpha(len(ref), alpha_min=alphaMin, alpha_max=alphaMax)
        for i in range(1, len(ref) - 1):
            X = ref[i]
            u = _unit(S[i] - X)
            n = _unit(nrm[i])
            alpha_use = alphas[i]
            d_hat = _slerp_2d(u, n, alpha_use) if use_slerp else _unit((1 - alpha_use) * u + alpha_use * n)

            b = np.dot(O - X, d_hat)
            c = np.dot(O - X, O - X) - R * R
            disc = b * b - c
            if disc < 0:
                disc = 0.0
            t_forw = b + np.sqrt(disc)

            # fallback if intersection goes backwards
            if t_forw < 0:
                d_hat = u
                b = np.dot(O - X, d_hat)
                disc = b * b - c
                t_forw = b + np.sqrt(max(0.0, disc))

            Y[i] = X + t_forw * d_hat

        return [tuple(pt) for pt in Y]




    @staticmethod
    def find_alphas(
        p1, p2, ref_polyline, normals,
        direction="ccw", use_slerp=False,
        alphaMin_max=0.7, alphaMax_max=0.9,
        tol_xi=1e-6,
        gamma = 0.01
    ):
        def _unwrap_arc_params(Y, O, start_angle, direction):
            """
            Compute arc parameter xi for points Y on the circle centered at O.
            xi is measured from start_angle, increasing CCW if direction='ccw',
            decreasing if 'cw' (we flip sign so we always check increasing).
            """
            theta = np.arctan2(Y[:,1] - O[1], Y[:,0] - O[0])   # [-pi, pi]
            # shift by start_angle so the sequence should run roughly in [0, pi]
            theta_rel = np.unwrap(theta - start_angle)
            if direction == "cw":
                theta_rel = -theta_rel
            # normalize to [0, pi] length (monotonic check is scale-free, but this helps intuition)
            # Note: map_airfoil_to_semicircle_blended already constrains to semicircle
            return theta_rel

        def _xi_monotone_ok(xi, tol=1e-6):
            diffs = np.diff(xi)
            return np.all(diffs >= tol), diffs

        # Helper to evaluate monotonicity for given alphas
        def evaluate(alpha_min, alpha_max):
            Y = np.array(LineDistribution.map_airfoil_to_semicircle_blended(
                p1, p2,
                ref_polyline=ref_polyline,
                normals=normals,
                direction=direction,
                alphaMin=alpha_min,
                alphaMax=alpha_max,
                use_slerp=use_slerp
            ), float)
            O = 0.5*(np.array(p1)+np.array(p2))
            a_start = np.arctan2(p1[1]-O[1], p1[0]-O[0]) if direction=="ccw" else np.arctan2(p2[1]-O[1], p2[0]-O[0])
            xi = _unwrap_arc_params(Y, O, a_start, direction)
            ok, diffs = _xi_monotone_ok(xi, tol=tol_xi)
            return ok, Y, xi, diffs

 
        # alphaMin from high to low (inclusive), by gamma
        a_min = alphaMin_max
        selected = []
        # Use while loops to avoid floating-point inclusivity issues
        while a_min >= 0.0:

            # alphaMax from current alphaMin up to alphaMax_max (inclusive), by gamma
    
            a_max = alphaMax_max

            while a_max >= a_min + 1e-12:

                ok, Y, *_ = evaluate(
                    a_min, a_max
                )

                if ok:
                    # FIRST FEASIBLE: return immediately, per your spec/order
                    #print(f"using alphaMin = {a_min} and alphaMax = {a_max}")
                    selected.append((a_min, a_max))
                    break
                    #return [tuple(pt) for pt in Y]

                a_max -= gamma

            a_min -= gamma

        selected = np.asarray(selected)          # shape (N, 2)
        scores = 0.2*selected[:,0] + 0.8*selected[:,1]
        idx_best = int(np.argmax(scores))
        print(f"using alphaMin = {selected[idx_best][0]} and alphaMax = {selected[idx_best][1]}")
        ok, Y, *_ = evaluate(
            selected[idx_best][0], selected[idx_best][1]
        )
        return [tuple(pt) for pt in Y]


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
    """
    @staticmethod
    def dist_arc_with_ref_spacing(
        p1, p2,
        center, radius,
        ref_polyline,      # reference polyline for spacing
        direction="ccw",
        eps: float = 1e-6
    ):
        C = np.asarray(center, dtype=float)
        p1 = np.asarray(p1, dtype=float)
        p2 = np.asarray(p2, dtype=float)
        R = float(radius)
        if R <= 0:
            raise ValueError("radius must be positive")

        ref_polyline = np.asarray(ref_polyline, dtype=float)
        if ref_polyline.ndim != 2 or ref_polyline.shape[1] != 2:
            raise ValueError("ref_polyline must be an array/list of (x, y) points.")
        if ref_polyline.shape[0] < 2:
            raise ValueError("ref_polyline must contain at least 2 points.")

        n_pts = ref_polyline.shape[0]

        # --- angles for the arc ---
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
            # mimic length of ref_polyline: endpoints with repeats
            pts = [tuple(p1)] * (n_pts - 1) + [tuple(p2)]
            return pts

        # --- spacing fractions from reference polyline ---
        diffs_ref = np.diff(ref_polyline, axis=0)
        seg_lens_ref = np.linalg.norm(diffs_ref, axis=1)
        s_ref = np.concatenate(([0.0], np.cumsum(seg_lens_ref)))
        total_len_ref = s_ref[-1]
        if total_len_ref <= eps:
            # reference is degenerate, fall back to uniform spacing
            t = np.linspace(0.0, 1.0, n_pts)
        else:
            s_ref /= total_len_ref         # normalize to [0,1]
            t = s_ref                      # use these as arc-length fractions

        # --- map fractions to arc angles & points ---
        angles = a1 + delta * t
        xs = C[0] + R * np.cos(angles)
        ys = C[1] + R * np.sin(angles)

        polyLine = [tuple(pt) for pt in np.column_stack([xs, ys])]
        polyLine[0]  = tuple(p1)  # pin exact endpoints
        polyLine[-1] = tuple(p2)
        return polyLine
    """
    @staticmethod
    def u_shape_with_ref_spacing(top_right, bottom_right, straight_len, ref_polyline):
        """
        U on the LEFT:
        top straight (left from top_right by straight_len)
        -> semicircle with center/radius per user's formula
        -> bottom straight (from arc end to bottom_right).
        Spacing follows ref_polyline segment-length fractions.
        """
        TR = np.asarray(top_right, float)
        BR = np.asarray(bottom_right, float)

        # Center and radius from your formula
        Cx = TR[0] - float(straight_len)
        Cy = 0.5 * (TR[1] + BR[1])
        R  = 0.5 * abs(TR[1] - BR[1])

        if R <= 0:
            raise ValueError("top_right and bottom_right must have different y to form a U.")

        # Key points
        A = np.array([Cx, TR[1]])  # start of semicircle (top tangent)
        B = np.array([Cx, BR[1]])  # end of semicircle (bottom tangent)

        # Segment lengths
        L_top = abs(TR[0] - Cx)            # should equal straight_len
        L_arc = np.pi * R                   # semicircle (left side)
        L_bot = abs(BR[0] - Cx)            # straight to bottom_right
        L_tot = L_top + L_arc + L_bot

        # Build cumulative-length fractions from reference
        ref = np.asarray(ref_polyline, float)
        seg_len = np.linalg.norm(np.diff(ref, axis=0), axis=1)
        if seg_len.sum() <= 0:
            t = np.linspace(0.0, 1.0, len(ref))
        else:
            s = np.insert(np.cumsum(seg_len), 0, 0.0)
            t = s / s[-1]

        out = []
        for frac in t:
            s_along = frac * L_tot
            if s_along <= L_top:  # top straight, move left
                p = np.array([TR[0] - s_along, TR[1]])
            elif s_along <= L_top + L_arc:  # left semicircle: θ: +π/2 → −π/2
                s_arc = s_along - L_top
                tau   = s_arc / L_arc
                theta = np.pi/2 - np.pi * tau
                x = Cx - R * np.cos(theta)
                y = Cy + R * np.sin(theta)
                p = np.array([x, y])
            else:  # bottom straight, go to bottom_right
                s_bot = s_along - (L_top + L_arc)
                # allow either direction on x just in case
                dir_sign = np.sign(BR[0] - Cx) if BR[0] != Cx else 1.0
                p = np.array([Cx + dir_sign * s_bot, BR[1]])
            out.append((float(p[0]), float(p[1])))

        return out

    def u_shape_uniform(top_right, bottom_right, straight_len, n_segments):
        """
        Uniformly distribute points along a U on the LEFT:
        top straight (left from top_right by straight_len)
        -> semicircle with center/radius per the same formula
        -> bottom straight (from arc end to bottom_right).

        Parameters
        ----------
        top_right, bottom_right : (x, y)
        straight_len : float
        n_segments : int   # total segments along the whole U (returns n_segments+1 points)

        Returns
        -------
        list[(x, y)] of length n_segments+1
        """
        if n_segments < 1:
            raise ValueError("n_segments must be >= 1")

        TR = np.asarray(top_right, float)
        BR = np.asarray(bottom_right, float)

        # Center & radius (same as your spacing-by-ref version)
        Cx = TR[0] - float(straight_len)
        Cy = 0.5 * (TR[1] + BR[1])
        R  = 0.5 * abs(TR[1] - BR[1])
        if R <= 0:
            raise ValueError("top_right and bottom_right must have different y to form a U.")

        # Segment lengths
        L_top = abs(TR[0] - Cx)
        L_arc = np.pi * R
        L_bot = abs(BR[0] - Cx)
        L_tot = L_top + L_arc + L_bot

        # Uniform fractions
        t = np.linspace(0.0, 1.0, n_segments + 1)

        out = []
        for frac in t:
            s_along = frac * L_tot
            if s_along <= L_top:  # top straight, move left
                p = np.array([TR[0] - s_along, TR[1]])
            elif s_along <= L_top + L_arc:  # semicircle (left): θ: +π/2 → −π/2
                s_arc = s_along - L_top
                tau   = s_arc / L_arc
                theta = np.pi/2 - np.pi * tau
                x = Cx - R * np.cos(theta)
                y = Cy + R * np.sin(theta)
                p = np.array([x, y])
            else:  # bottom straight, move to BR
                s_bot = s_along - (L_top + L_arc)
                dir_sign = np.sign(BR[0] - Cx) if BR[0] != Cx else 1.0
                p = np.array([Cx + dir_sign * s_bot, BR[1]])
            out.append((float(p[0]), float(p[1])))

        return out
        
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
        
    @staticmethod
    def symmetric_decay_grow_line(start_point, end_point, h0, r=1.05, *, even=False):
        """
        Return a polyline [(x0,y0), ... , (xN,yN)] from start->end where
        segment lengths decay geometrically from h0 by ratio r toward the middle,
        then grow symmetrically back to h0. Only the points are returned.

        This is roughly the "inverse" of symmetric_grow_decay_line:
        - here: large segments near the endpoints, small near the middle
        - symmetric_grow_decay_line: small near endpoints, large near middle

        Parameters
        ----------
        start_point : (x, y)
        end_point   : (x, y)
        h0          : initial segment length at each end (largest)
        r           : >1, geometric ratio for decay (h0, h0/r, h0/r^2, ...)
        even        : if False → single central segment
                    if True  → split center into two equal segments

        Raises
        ------
        ValueError if geometry/parameters are inconsistent.
        """
        if r <= 1.0:
            raise ValueError("r must be > 1.")
        x0, y0 = start_point
        x1, y1 = end_point
        vec = np.array([x1 - x0, y1 - y0], dtype=float)
        L = float(np.linalg.norm(vec))
        if L <= 0:
            raise ValueError("Start and end points must differ.")
        if L < 2*h0:
            raise ValueError("Total length must be at least 2*h0.")
        d = vec / L  # unit direction

        def sum_wing(k_):
            """Total length of one 'wing' with k segments: h0, h0/r, ..., h0/r^(k-1)."""
            if k_ <= 0:
                return 0.0
            # geometric series: h0 * (1 - r^{-k}) / (1 - 1/r)
            return h0 * (1.0 - r**(-k_)) / (1.0 - 1.0/r)

        # choose k (decay steps per side)
        # grow k until adding one more step would overshoot L/2 per side
        k = 1
        while True:
            if 2.0 * sum_wing(k) > L:
                k -= 1
                break
            k += 1
            # safety guard
            if k > 10_000:
                raise RuntimeError("symmetric_decay_grow_line: k search did not converge.")

        if k < 0:
            k = 0

        # build one wing: h0, h0/r, ..., h0/r^(k-1)
        if k == 0:
            # trivial: just two equal segments
            seg_lengths = [L/2.0, L/2.0]
        else:
            wing = [h0 / (r**m) for m in range(k)]
            S_w = sum(wing)
            remainder = L - 2.0 * S_w

            if remainder < 0:
                # should not happen due to k search, but guard anyway
                raise ValueError("Negative remainder: parameters inconsistent.")

            if not even:
                # one center segment
                seg_lengths = wing + [remainder] + wing[::-1]
            else:
                # two central segments: adjust the smallest (middle) ones
                left = wing.copy()
                right = wing[::-1]
                # smallest segments are at the inner end: wing[-1], right[0]
                left[-1]  = left[-1]  + 0.5 * remainder
                right[0]  = right[0]  + 0.5 * remainder
                seg_lengths = left + right

        # cumulative distances and points
        s = np.insert(np.cumsum(seg_lengths), 0, 0.0)
        line = [(x0 + si * d[0], y0 + si * d[1]) for si in s]
        line[-1] = (x1, y1)  # snap exact end

        return line

    @staticmethod
    def symmetric_decay_grow_line_fixed_N(start_point, end_point, h0, r, N):
        """
        Polyline from start_point to end_point with N segments (N+1 points).
        Segment lengths follow a decay-then-grow geometric pattern:

        h_0 ~ h0, h_1 ~ h0/r, ..., then mirrored back, last segment ~ h0.

        The pattern is scaled uniformly so that the sum of segment lengths is
        exactly the distance between start_point and end_point.

        Parameters
        ----------
        start_point : (x, y)
        end_point   : (x, y)
        h0          : base thickness/segment size at both ends (before scaling)
        r           : >= 1, geometric ratio for decay (r = 1 → uniform spacing)
        N           : number of segments (must be even → N/2 decay, N/2 grow)

        Returns
        -------
        line : list of (x, y) points, length N+1
        """
        if N % 2 != 0:
            raise ValueError("N must be even (N/2 decay, N/2 grow).")
        if r < 1.0:
            raise ValueError("r must be >= 1. (r = 1 → uniform spacing)")

        x0, y0 = start_point
        x1, y1 = end_point
        vec = np.array([x1 - x0, y1 - y0], dtype=float)
        L = float(np.linalg.norm(vec))
        if L <= 0:
            raise ValueError("Start and end points must differ.")

        # raw pattern (before scaling)
        if r == 1.0:
            # uniform raw lengths → uniform spacing after scaling
            seg_raw = np.full(N, h0, dtype=float)
        else:
            idx = np.arange(N)
            m = np.minimum(idx, N - 1 - idx)       # distance from nearest end
            seg_raw = h0 * (r ** (-m))             # decay then grow

        # scale so sum(seg_lengths) = L
        S = float(np.sum(seg_raw))
        scale = L / S
        seg_lengths = seg_raw * scale

        # cumulative positions
        d = vec / L
        s = np.insert(np.cumsum(seg_lengths), 0, 0.0)  # length N+1
        line = [(x0 + si * d[0], y0 + si * d[1]) for si in s]
        # snap exact end to avoid drift
        line[-1] = (x1, y1)

        return line
    @staticmethod
    def triple_cluster_line(start_point, end_point, n_points, r=1.2, eps=1e-12):
        """
        Return a polyline [(x0,y0), ... , (xN-1,yN-1)] from start->end where
        segment lengths are smallest near both ends and near the center,
        and larger in between (three clusters).

        Parameters
        ----------
        start_point : (x, y)
        end_point   : (x, y)
        n_points    : total number of points (>= 3)
        r           : > 1, geometric factor controlling how strong the clustering is
        eps         : small tolerance

        Returns
        -------
        line : list of (x, y) of length n_points
            Points are exactly on the straight line, with nonuniform spacing.
        """
        if n_points < 3:
            raise ValueError("n_points must be at least 3 for end+center clustering.")
        if r <= 1.0:
            raise ValueError("r must be > 1.0 to get clustering.")
        
        x0, y0 = start_point
        x1, y1 = end_point
        vec = np.array([x1 - x0, y1 - y0], dtype=float)
        L = float(np.linalg.norm(vec))
        if L <= eps:
            raise ValueError("Start and end points must differ.")
        
        d = vec / L  # unit direction
        
        # Number of segments
        M = n_points - 1
        
        # Index of each segment: 0 ... M-1
        idx = np.arange(M)
        
        # "center" segment index (for odd M it's exact middle, for even M it's left-middle)
        mid = M // 2
        
        # distance in index space to each "cluster center":
        # - left end (0)
        # - right end (M-1)
        # - center (mid)
        dist_left  = idx
        dist_right = (M - 1) - idx
        dist_mid   = np.abs(idx - mid)
        
        # for each segment, its "cluster distance" is distance to the nearest of the three
        m = np.minimum(np.minimum(dist_left, dist_right), dist_mid)
        
        # raw segment sizes: smallest when m=0 (ends & center), largest in-between
        # you can tweak this shape; r>1 makes neighbors larger.
        seg_raw = r ** m   # relative sizes
        
        # scale so that sum of seg_lengths = L
        S = float(np.sum(seg_raw))
        seg_lengths = (L / S) * seg_raw
        
        # cumulative distances
        s = np.insert(np.cumsum(seg_lengths), 0, 0.0)  # length n_points
        
        # points on the line
        line = [(x0 + si * d[0], y0 + si * d[1]) for si in s]
        line[-1] = (x1, y1)  # snap exact end
        
        return line
    @staticmethod
    def triple_cluster_line_two_ratios(
        start_point, end_point, n_points,
        r_end=1.3, r_center=1.3, eps=1e-12
    ):
        """
        Return a polyline [(x0,y0), ... , (xN-1,yN-1)] from start->end where
        segment lengths are smallest near both ends and near the center, with
        *different* geometric clustering ratios at the ends and at the center.

        Parameters
        ----------
        start_point : (x, y)
        end_point   : (x, y)
        n_points    : total number of points (>= 3)
        r_end       : > 1, geometric factor controlling clustering at the ends
        r_center    : > 1, geometric factor controlling clustering at the center
        eps         : small tolerance

        Returns
        -------
        line : list of (x, y) of length n_points
            Points are exactly on the straight line, with nonuniform spacing.
        """
        if n_points < 3:
            raise ValueError("n_points must be at least 3 for end+center clustering.")
        if r_end <= 1.0 or r_center <= 1.0:
            raise ValueError("r_end and r_center must both be > 1.0.")

        x0, y0 = start_point
        x1, y1 = end_point
        vec = np.array([x1 - x0, y1 - y0], dtype=float)
        L = float(np.linalg.norm(vec))
        if L <= eps:
            raise ValueError("Start and end points must differ.")

        d = vec / L  # unit direction

        # Number of segments
        M = n_points - 1
        idx = np.arange(M)

        # center segment index (for even M, this is the left of the two middles)
        mid = M // 2

        # distances in index space
        dist_left  = idx
        dist_right = (M - 1) - idx
        dist_end   = np.minimum(dist_left, dist_right)   # distance to nearest end
        dist_mid   = np.abs(idx - mid)                   # distance to center

        # Decide which cluster each segment is "closest" to:
        # if closer to center → use r_center, else use r_end
        seg_raw = np.empty(M, dtype=float)
        use_center = dist_mid <= dist_end
        use_end    = ~use_center

        seg_raw[use_center] = r_center ** dist_mid[use_center]
        seg_raw[use_end]    = r_end    ** dist_end[use_end]

        # Now seg_raw encodes relative segment lengths: smallest near cluster centers
        # (where dist = 0 → factor ~1), larger between clusters.

        # Scale so that sum of seg_lengths = L
        S = float(np.sum(seg_raw))
        seg_lengths = (L / S) * seg_raw

        # cumulative distances
        s = np.insert(np.cumsum(seg_lengths), 0, 0.0)  # length n_points

        # points on the line
        line = [(x0 + si * d[0], y0 + si * d[1]) for si in s]
        line[-1] = (x1, y1)  # snap exact end

        return line

    @staticmethod
    def triple_cluster_line_smooth(
        start_point, end_point, n_points,
        r_end=2.0, r_center=2.0,
        sigma_end=0.15, sigma_center=0.10,
        eps=1e-12,
    ):
        """
        Distribute n_points along the straight line from start_point to end_point
        with three smooth clusters:
        - refined near both ends
        - refined near the center
        and gradual changes in segment length (no abrupt jumps).

        Parameters
        ----------
        start_point : (x, y)
        end_point   : (x, y)
        n_points    : int >= 3, total number of points
        r_end       : > 1, strength of clustering at the ends
        r_center    : > 1, strength of clustering at the center
                    (bigger => stronger clustering, i.e. smaller segments)
        sigma_end   : width of the end clusters in param space [0,1]
        sigma_center: width of the center cluster in param space [0,1]
        eps         : small tolerance

        Returns
        -------
        line : list[(x, y)] of length n_points
            Points lie on the straight line, spacing smoothly varying.
        """
        if n_points < 3:
            raise ValueError("n_points must be at least 3 for end+center clustering.")
        if r_end <= 1.0 or r_center <= 1.0:
            raise ValueError("r_end and r_center must both be > 1.0.")

        x0, y0 = start_point
        x1, y1 = end_point
        vec = np.array([x1 - x0, y1 - y0], dtype=float)
        L = float(np.linalg.norm(vec))
        if L <= eps:
            raise ValueError("Start and end points must differ.")

        d = vec / L  # unit direction

        # number of segments
        M = n_points - 1
        idx = np.arange(M, dtype=float)

        # param for segment centers in [0,1]
        u = (idx + 0.5) / M

        def gauss(u, mu, sigma):
            return np.exp(-0.5 * ((u - mu) / max(sigma, eps))**2)

        # base density (1 everywhere) plus three bumps
        # r_end, r_center control *extra* density at the clusters
        density = (
            1.0
            + (r_end - 1.0)   * (gauss(u, 0.0, sigma_end) + gauss(u, 1.0, sigma_end))
            + (r_center - 1.0) * gauss(u, 0.5, sigma_center)
        )

        # segment lengths ∝ 1 / density (more density => smaller segments)
        seg_raw = 1.0 / density

        # scale to match total length
        S = float(np.sum(seg_raw))
        seg_lengths = (L / S) * seg_raw

        # cumulative distances → points
        s = np.insert(np.cumsum(seg_lengths), 0, 0.0)  # length n_points
        line = [(x0 + si * d[0], y0 + si * d[1]) for si in s]
        line[-1] = (x1, y1)  # snap exact end

        return line
    
    @staticmethod
    def ellipse_like_conic_from_2pts_2tangents(
        p1, t1_pA, t1_pB,
        p2, t2_pA, t2_pB,
        n_points: int = 50,
        weight: float = 0.6,      # 0<w<1 => ellipse-like
        eps: float = 1e-12,
    ):
        """
        Construct an ellipse-like conic arc between two points with prescribed tangent
        directions at the endpoints, using a *rational quadratic Bézier* (conic).

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

        # Basic endpoint sanity
        chord = p2 - p1
        if np.linalg.norm(chord) < eps:
            return [tuple(p1)] * n_points

        # --- Find intersection of the two tangent lines ---
        # Line1: p1 + a * t1
        # Line2: p2 + b * (-t2)   (because tangent at p2 points "outward" along t2)
        d1 = t1
        d2 = -t2

        # Solve p1 + a d1 = p2 + b d2  =>  a d1 - b d2 = (p2 - p1)
        A = np.column_stack([d1, -d2])  # 2x2
        rhs = (p2 - p1)

        detA = np.linalg.det(A)
        if abs(detA) < eps:
            # tangents nearly parallel: conic control point goes to infinity
            # -> fall back to something reasonable (quadratic with midpoint control)
            P0 = p1
            P2 = p2
            P1 = 0.5 * (p1 + p2)
        else:
            a_b = np.linalg.solve(A, rhs)
            a = a_b[0]
            P0 = p1
            P2 = p2
            P1 = p1 + a * d1  # intersection point

        # Weight sanity
        w = float(weight)
        if w <= eps:
            raise ValueError("weight must be > 0. For ellipse-like arcs, use 0 < weight < 1.")

        def rational_quadratic_eval(t):
            """
            Rational quadratic Bézier:
              C(t) = ( (1-t)^2 P0 + 2 w t(1-t) P1 + t^2 P2 ) / ( (1-t)^2 + 2 w t(1-t) + t^2 )
            """
            t = np.asarray(t, dtype=float)
            omt = 1.0 - t

            b0 = omt * omt
            b1 = 2.0 * w * t * omt
            b2 = t * t
            denom = (b0 + b1 + b2)

            # shape (N,2)
            num = (b0[:, None] * P0[None, :] +
                   b1[:, None] * P1[None, :] +
                   b2[:, None] * P2[None, :])
            return num / denom[:, None]

        # ---- Reparameterize to ~uniform arc length ----
        n_sample = max(10 * n_points, 300)
        t_sample = np.linspace(0.0, 1.0, n_sample)
        pts_sample = rational_quadratic_eval(t_sample)

        diffs = np.diff(pts_sample, axis=0)
        seg_lens = np.linalg.norm(diffs, axis=1)
        s = np.concatenate(([0.0], np.cumsum(seg_lens)))
        total_len = s[-1]

        if total_len < eps:
            return [tuple(p1)] * n_points

        s /= total_len
        s_target = np.linspace(0.0, 1.0, n_points)

        pts_uniform = []
        j = 0
        for st in s_target:
            while j < len(s) - 2 and s[j + 1] < st:
                j += 1
            s0, s1 = s[j], s[j + 1]
            p0, p1_ = pts_sample[j], pts_sample[j + 1]

            if s1 - s0 < eps:
                alpha = 0.0
            else:
                alpha = (st - s0) / (s1 - s0)

            p = (1.0 - alpha) * p0 + alpha * p1_
            pts_uniform.append((float(p[0]), float(p[1])))

        pts_uniform[0] = (float(P0[0]), float(P0[1]))
        pts_uniform[-1] = (float(P2[0]), float(P2[1]))
        return pts_uniform