import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline  # needs SciPy


class LineDistribution:
    """Utility class for distributing and generating polylines.

    Provides static helper methods for:
    - plotting polylines,
    - redistributing points based on a reference line,
    - generating circular arcs with reference-based spacing,
    - constructing arc-like curves using splines or Bézier representations.
    """

    @staticmethod
    def plot_lines(lines, colors=None, markers=None, labels=None):
        """Plot multiple polylines defined by point coordinates.

        :param lines: list of lines, where each line is a list of ``(x, y)`` points.
        :type lines: list[list[tuple[float, float]]]
        :param colors: list of colors for each line.
        :type colors: list[str], optional
        :param markers: list of marker styles for each line.
        :type markers: list[str], optional
        :param labels: list of labels for each line.
        :type labels: list[str], optional
        :raises ValueError: if ``lines`` is empty.
        :return: None
        :rtype: None
        """
        if not lines:
            raise ValueError("No lines to plot.")

        num_lines = len(lines)

        # Default styling
        if colors is None:
            colors = ["blue", "red", "green", "orange", "purple", "black"] * (
                num_lines // 6 + 1
            )
        if markers is None:
            markers = ["o", "s", "^", "d", "x", "*"] * (num_lines // 6 + 1)
        if labels is None:
            labels = [f"Line {i+1}" for i in range(num_lines)]

        plt.figure()
        for i, line in enumerate(lines):
            if not line:
                continue
            x, y = zip(*line)
            plt.plot(x, y, color=colors[i], marker=markers[i], label=labels[i])

        # plt.gca().set_aspect('equal')
        plt.grid(True)
        plt.legend()
        plt.show()

    @staticmethod
    def divide_line_by_reference(start_point, end_point, line_ref):
        """Create a line divided according to a reference line's spacing.

        A new polyline from ``start_point`` to ``end_point`` is generated such
        that the relative segment lengths match those of ``line_ref``.

        :param start_point: starting point of the new line.
        :type start_point: tuple[float, float]
        :param end_point: ending point of the new line.
        :type end_point: tuple[float, float]
        :param line_ref: reference polyline defining spacing proportions.
        :type line_ref: list[tuple[float, float]]
        :return: new polyline with the same number of points as ``line_ref``.
        :rtype: list[tuple[float, float]]
        :raises ValueError: if the reference line has zero length or
            if ``start_point`` and ``end_point`` coincide.
        """

        # Convert reference line to numpy array
        ref = np.array(line_ref)
        segment_lengths = np.linalg.norm(np.diff(ref, axis=0), axis=1)
        total_segments = len(segment_lengths)

        if np.sum(segment_lengths) == 0:
            raise ValueError("Reference line has zero length.")

        # Cumulative proportions (0 to 1, inclusive)
        proportions = np.cumsum(np.insert(segment_lengths, 0, 0)) / np.sum(
            segment_lengths
        )

        # Direction vector from start to end
        start = np.array(start_point)
        end = np.array(end_point)
        direction = end - start
        total_length = np.linalg.norm(direction)

        if total_length == 0:
            raise ValueError("Start and end points are the same.")

        unit_direction = direction / total_length

        # Generate new points along the direction
        new_points = [
            start + prop * total_length * unit_direction for prop in proportions
        ]

        return [tuple(pt) for pt in new_points]

    @staticmethod
    def arc_with_ref_spacing(
        p1, p2, ref_polyline, radius, direction="ccw", long_arc=False
    ):
        """Generate a circular arc with spacing taken from a reference polyline.

        The arc connects ``p1`` to ``p2`` on a circle of given ``radius``.
        The number of points and their relative spacing are taken from
        ``ref_polyline``.

        :param p1: starting point of the arc.
        :type p1: tuple[float, float]
        :param p2: ending point of the arc.
        :type p2: tuple[float, float]
        :param ref_polyline: reference polyline defining spacing proportions.
        :type ref_polyline: list[tuple[float, float]]
        :param radius: circle radius (must satisfy ``radius >= |p2-p1|/2``).
        :type radius: float
        :param direction: arc direction, ``"ccw"`` or ``"cw"``.
        :type direction: str, optional
        :param long_arc: if ``True``, use the major arc; otherwise the minor arc.
        :type long_arc: bool, optional
        :return: polyline points on the circular arc.
        :rtype: list[tuple[float, float]]
        :raises ValueError: if inputs are invalid or geometry is infeasible.
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
        h_sq = R * R - (0.5 * d) * (0.5 * d)
        h = 0.0 if h_sq < 0 else float(np.sqrt(h_sq))

        centers = [M + h * n]
        if h > 1e-15:  # two distinct centers unless R == d/2
            centers.append(M - h * n)

        def wrap_to_pi(a):
            """Wrap angle to (-pi, pi]."""
            return (a + np.pi) % (2 * np.pi) - np.pi

        def delta_for_center(C, dir_ccw: bool):
            # angles of p1, p2 around center C
            a1 = np.arctan2(p1[1] - C[1], p1[0] - C[0])
            a2 = np.arctan2(p2[1] - C[1], p2[0] - C[0])
            d_raw = wrap_to_pi(a2 - a1)

            if dir_ccw:
                # want CCW: positive delta; add 2π if needed
                if d_raw <= 0:
                    d_raw += 2 * np.pi
            else:
                # want CW: negative delta; subtract 2π if needed
                if d_raw >= 0:
                    d_raw -= 2 * np.pi

            # minor arc has |delta| <= π (within tolerance)
            is_long = abs(d_raw) > np.pi + 1e-12
            return d_raw, a1, is_long

        want_ccw = str(direction).lower() == "ccw"

        # Evaluate both candidate centers; pick one that matches long_arc flag
        best = None
        for C in centers:
            delta, a1, is_long = delta_for_center(C, want_ccw)
            score = is_long == bool(long_arc)  # exact match preferred
            if best is None or score:  # prefer matching long/minor
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
    def dist_arc_with_ref_spacing(
        p1,
        p2,
        center,
        radius,
        ref_polyline,
        direction="ccw",
        alpha=1.0,  # 0 = uniform, 1 = ref-based
        eps: float = 1e-6,
    ):
        """Generate a circular arc with blended spacing control.

        Spacing is computed as a blend between uniform spacing and spacing
        derived from the cumulative arc-length of ``ref_polyline``.

        :param p1: starting point of the arc.
        :type p1: tuple[float, float]
        :param p2: ending point of the arc.
        :type p2: tuple[float, float]
        :param center: center of the circle.
        :type center: tuple[float, float]
        :param radius: circle radius.
        :type radius: float
        :param ref_polyline: reference polyline defining spacing proportions.
        :type ref_polyline: list[tuple[float, float]]
        :param direction: arc direction, ``"ccw"`` or ``"cw"``.
        :type direction: str, optional
        :param alpha: blending factor (0 = uniform, 1 = reference-based).
        :type alpha: float, optional
        :param eps: numerical tolerance.
        :type eps: float, optional
        :return: polyline points on the circular arc.
        :rtype: list[tuple[float, float]]
        :raises ValueError: if inputs are invalid.
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
        p1,
        p2,
        center,
        radius,
        ref_polyline,
        eps: float = 1e-9,
    ):
        """Generate a circular arc using spline-based normals.

        A cubic spline is fitted through ``ref_polyline`` to compute smooth
        tangents and normals. Interior points are obtained by intersecting
        these normals with the target circle.

        :param p1: starting point of the arc.
        :type p1: tuple[float, float]
        :param p2: ending point of the arc.
        :type p2: tuple[float, float]
        :param center: center of the circle.
        :type center: tuple[float, float]
        :param radius: circle radius.
        :type radius: float
        :param ref_polyline: reference polyline defining number of points.
        :type ref_polyline: list[tuple[float, float]]
        :param eps: numerical tolerance.
        :type eps: float, optional
        :return: polyline points on the circular arc.
        :rtype: list[tuple[float, float]]
        :raises ValueError: if geometry or inputs are invalid.
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
    def closed_left_U(
        bottom_right, vert_len, horiz_len, ref_polyline=None, n_segments=100
    ):
        """Construct a closed left-sided U-shaped polyline.

        The shape consists of:
        - a bottom horizontal segment,
        - a left semicircle,
        - a top horizontal segment,
        - a vertical closing edge on the right.

        :param bottom_right: bottom-right corner of the shape.
        :type bottom_right: tuple[float, float]
        :param vert_len: vertical height of the U.
        :type vert_len: float
        :param horiz_len: horizontal length of the top and bottom segments.
        :type horiz_len: float
        :param ref_polyline: optional reference polyline for spacing control.
        :type ref_polyline: list[tuple[float, float]], optional
        :param n_segments: number of segments if ``ref_polyline`` is not provided.
        :type n_segments: int, optional
        :return: closed polyline where first and last points coincide.
        :rtype: list[tuple[float, float]]
        :raises ValueError: if geometric parameters are invalid.
        """
        BR = np.asarray(bottom_right, float)
        H = float(vert_len)
        L = float(horiz_len)
        if H <= 0 or L < 0:
            raise ValueError("vert_len must be > 0 and horiz_len >= 0")

        # Geometry
        TR = np.array([BR[0], BR[1] + H])  # top-right
        Cx = BR[0] - L  # semicircle center x (left of right edge)
        Cy = BR[1] + 0.5 * H
        R = 0.5 * H

        # Segment lengths
        L_bot, L_arc, L_top, L_right = L, np.pi * R, L, H
        L_tot = L_bot + L_arc + L_top + L_right

        # Fractions t in [0,1]
        if ref_polyline is not None:
            ref = np.asarray(ref_polyline, float)
            if ref.ndim != 2 or ref.shape[1] != 2:
                raise ValueError("ref_polyline must be an (N,2) array-like")
            seg_len = (
                np.linalg.norm(np.diff(ref, axis=0), axis=1)
                if len(ref) >= 2
                else np.array([1.0])
            )
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
                tau = s_arc / L_arc
                theta = -np.pi / 2 + np.pi * tau
                x = Cx - R * np.cos(theta)
                y = Cy + R * np.sin(theta)
                p = np.array([x, y])
            elif s_along <= L_bot + L_arc + L_top:
                # top straight: move RIGHT to TR
                s_top = s_along - (L_bot + L_arc)
                p = np.array([Cx + s_top, TR[1]])
                s_along_next = t[ind + 1] * L_tot
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
        p1,
        t1_pA,
        t1_pB,
        p2,
        t2_pA,
        t2_pB,
        n_points: int = 50,
        handle_scale: float = 1.0,
        eps: float = 1e-12,
    ):
        """Construct an arc-like cubic Bézier curve with prescribed tangents.

        A cubic Bézier curve is defined between ``p1`` and ``p2`` using tangent
        directions inferred from the point pairs ``(t1_pA, t1_pB)`` and
        ``(t2_pA, t2_pB)``. The curve is reparameterized to obtain approximately
        uniform arc-length spacing.

        :param p1: starting point of the curve.
        :type p1: tuple[float, float]
        :param t1_pA: first point defining tangent direction at ``p1``.
        :type t1_pA: tuple[float, float]
        :param t1_pB: second point defining tangent direction at ``p1``.
        :type t1_pB: tuple[float, float]
        :param p2: ending point of the curve.
        :type p2: tuple[float, float]
        :param t2_pA: first point defining tangent direction at ``p2``.
        :type t2_pA: tuple[float, float]
        :param t2_pB: second point defining tangent direction at ``p2``.
        :type t2_pB: tuple[float, float]
        :param n_points: number of output points.
        :type n_points: int, optional
        :param handle_scale: scaling factor for Bézier control handles.
        :type handle_scale: float, optional
        :param eps: numerical tolerance.
        :type eps: float, optional
        :return: arc-length reparameterized Bézier polyline.
        :rtype: list[tuple[float, float]]
        :raises ValueError: if tangents or geometry are degenerate.
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
                B0[:, None] * P0[None, :]
                + B1[:, None] * P1[None, :]
                + B2[:, None] * P2[None, :]
                + B3[:, None] * P3[None, :]
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

        # ensure exact endpoints (avoid tiny numerical drift)
        pts_uniform[0] = (float(P0[0]), float(P0[1]))
        pts_uniform[-1] = (float(P3[0]), float(P3[1]))

        return pts_uniform

    @staticmethod
    def size_x_linear(x, x_left, x_right, lc_left, lc_right):
        xi = (x - x_left) / (x_right - x_left)
        xi = np.clip(xi, 0.0, 1.0)
        return lc_left + (lc_right - lc_left) * xi

    @staticmethod
    def graded_straight_segment(p_start, p_end, size_fun, include_start=False, include_end=True):
        """
        Create points on a straight segment such that local spacing follows size_fun(x).

        p_start, p_end : tuples
        size_fun(x)    : returns desired local edge length
        """
        p_start = np.asarray(p_start, dtype=float)
        p_end = np.asarray(p_end, dtype=float)

        d = p_end - p_start
        L = np.linalg.norm(d)

        if L < 1e-14:
            pts = []
            if include_start:
                pts.append((float(p_start[0]), float(p_start[1])))
            if include_end and not include_start:
                pts.append((float(p_end[0]), float(p_end[1])))
            return pts

        e = d / L
        pts = []

        if include_start:
            pts.append((float(p_start[0]), float(p_start[1])))

        s = 0.0
        max_iter = 100000

        for _ in range(max_iter):
            x_here = p_start[0] + s * e[0]
            h = max(1e-10, float(size_fun(x_here)))

            if s + h >= L:
                break

            s += h
            p = p_start + s * e
            pts.append((float(p[0]), float(p[1])))

        if include_end:
            pts.append((float(p_end[0]), float(p_end[1])))

        return pts

    @staticmethod
    def geometric_fractions(length, first_cell, growth):
        """Normalized node positions of a geometric (GP) cell distribution.

        The cell sizes form a geometric series with the given ratio, starting
        from ``first_cell``. The number of cells is the one whose series comes
        closest to ``length``; the cells are then scaled uniformly so that the
        series closes exactly, which leaves the first cell within half a cell of
        ``first_cell``.

        The returned fractions are dimensionless and can be mapped onto any
        straight segment, which keeps the streamwise distribution consistent
        across segments of different lengths.

        :param length: total length spanned by the distribution.
        :type length: float
        :param first_cell: target size of the first cell.
        :type first_cell: float
        :param growth: ratio between consecutive cells. Should be >= 1.
        :type growth: float
        :return: increasing fractions from ``0.0`` to ``1.0`` (``n + 1`` values).
        :rtype: list[float]
        """
        L = float(length)
        a = float(first_cell)
        r = float(growth)

        if L <= 0.0:
            raise ValueError("'length' must be > 0.")
        if a <= 0.0:
            raise ValueError("'first_cell' must be > 0.")
        if r < 1.0:
            raise ValueError("'growth' must be >= 1.")
        if a >= L:
            raise ValueError("'first_cell' must be smaller than 'length'.")

        if abs(r - 1.0) < 1e-12:
            n = max(1, int(round(L / a)))
            return [i / n for i in range(n + 1)]

        # Number of cells of the series a, a*r, a*r**2, ... summing to 'length'.
        n = max(1, int(round(np.log1p(L * (r - 1.0) / a) / np.log(r))))

        if n == 1:
            return [0.0, 1.0]

        sizes = [a * r**i for i in range(n)]
        total = sum(sizes)

        fractions = [0.0]
        acc = 0.0
        for h in sizes:
            acc += h
            fractions.append(acc / total)
        fractions[-1] = 1.0

        return fractions

    @staticmethod
    def segment_from_fractions(p_start, p_end, fractions):
        """Place points on a straight segment at the given normalized positions.

        :param p_start: start point ``(x, y)``.
        :type p_start: tuple[float, float]
        :param p_end: end point ``(x, y)``.
        :type p_end: tuple[float, float]
        :param fractions: increasing values from ``0.0`` to ``1.0``.
        :type fractions: list[float]
        :return: points along the segment.
        :rtype: list[tuple[float, float]]
        """
        p_start = np.asarray(p_start, dtype=float)
        p_end = np.asarray(p_end, dtype=float)
        d = p_end - p_start

        return [
            (float(p_start[0] + t * d[0]), float(p_start[1] + t * d[1]))
            for t in fractions
        ]

    @staticmethod
    def uniform_segment(p_start, p_end, n_points):
        """Place ``n_points`` equally spaced points on a straight segment.

        :param p_start: start point ``(x, y)``.
        :type p_start: tuple[float, float]
        :param p_end: end point ``(x, y)``.
        :type p_end: tuple[float, float]
        :param n_points: number of points, including both end points.
        :type n_points: int
        :return: points along the segment.
        :rtype: list[tuple[float, float]]
        """
        n_points = int(n_points)
        if n_points < 2:
            raise ValueError("'n_points' must be >= 2.")

        return LineDistribution.segment_from_fractions(
            p_start, p_end, np.linspace(0.0, 1.0, n_points)
        )