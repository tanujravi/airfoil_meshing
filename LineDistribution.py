import numpy as np
import matplotlib.pyplot as plt

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