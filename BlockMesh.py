import numpy as np
from scipy.interpolate import splprep

"""
This code is taken and adapted according to requirement from PyAero repository.
GitHub: https://github.com/chiefenne/PyAero 
"""


class BlockMesh:
    """Container class for block-structured 2D meshes.

    The mesh is represented as a list of *U-lines* (lines in u-direction), where
    each U-line is a list of `(x, y)` tuples. V-lines (v-direction) are derived
    by collecting points at constant i-index across all U-lines.
    """

    def __init__(self):
        """Initialize an empty mesh container.

        Creates an empty list of U-lines. Each U-line is expected to be a list of
        `(x, y)` tuples of equal length across the mesh once fully defined.
        """
        self.ULines = list()

    def addLine(self, line):
        """Append a U-line to the mesh.

        :param line: list of `(x, y)` tuples describing a polyline.
        :type line: list[tuple[float, float]]
        :return: None
        :rtype: None
        """
        # line is a list of (x, y) tuples
        self.ULines.append(line)

    def setUlines(self, ulines):
        """Replace the current U-lines by a new list.

        :param ulines: list of U-lines; each U-line is a list of `(x, y)` tuples.
        :type ulines: list[list[tuple[float, float]]]
        :return: None
        :rtype: None
        """
        self.ULines = ulines

    def extrudeLine_cell_thickness(
        self, line, normals, cell_thickness=0.04, growth=1.05, extrusion_distance=0.4
    ):
        """Extrude a polyline along supplied normals using geometric layer growth.

        :param line: base polyline as list of `(x, y)` tuples.
        :type line: list[tuple[float, float]]
        :param normals: normal vectors for each point in `line`.
            Must have shape `(n, 2)` and correspond pointwise to `line`.
        :type normals: array-like
        :param cell_thickness: first-layer thickness (t).
        :type cell_thickness: float, optional
        :param growth: geometric growth ratio between successive layers (r).
        :type growth: float, optional
        :param extrusion_distance: total extrusion distance / boundary-layer height (L).
        :type extrusion_distance: float, optional
        :return: None
        :rtype: None
        """
        # ensure float64 during all math
        pts = np.asarray(line, dtype=np.float64)  # (n,2)
        normals = np.asarray(normals, dtype=np.float64)  # (n,2)
        x, y = pts[:, 0], pts[:, 1]

        spacing, _ = self.spacing_cell_thickness(
            cell_thickness=cell_thickness,
            growth=growth,
            extrusion_distance=extrusion_distance,
        )  # spacing is np.float64 array

        for s in spacing:
            xo = x + s * normals[:, 0]  # float64
            yo = y + s * normals[:, 1]  # float64

            # Option 1: tuples of np.float64 scalars
            poly = list(zip(map(np.float64, xo), map(np.float64, yo)))

            # Option 2 (also fine): tuples of Python float (also 64-bit)
            # poly = [(float(a), float(b)) for a, b in zip(xo, yo)]

            self.addLine(poly)  # addLine stays unchanged (expects tuples)

    def getNodeCoo(self, node):
        """Return the coordinate of a node addressed by structured indices.

        The node is addressed by `(I, J)` where:
        - `I` is the index along a U-line (u-direction),
        - `J` is the index of the U-line (v-direction).

        :param node: node indices as `(I, J)`.
        :type node: tuple[int, int] | list[int]
        :return: node coordinate as NumPy array `[x, y]`.
        :rtype: np.ndarray
        """
        I, J = node[0], node[1]
        uline = self.getULines()[J]
        point = uline[I]
        return np.array(point)

    def setNodeCoo(self, node, new_pos):
        """Set the coordinate of a node addressed by structured indices.

        :param node: node indices as `(I, J)`.
        :type node: tuple[int, int] | list[int]
        :param new_pos: new coordinate as `(x, y)` or array-like of length 2.
        :type new_pos: tuple[float, float] | np.ndarray | list[float]
        :return: None
        :rtype: None
        """
        I, J = node[0], node[1]
        uline = self.getULines()[J]
        uline[I] = new_pos
        return

    def makeSubBlock(self, ij=[]):
        """Create a sub-block defined by index bounds and return it as a new mesh.

        The sub-block is defined by `ij = [istart, iend, jstart, jend]` and is
        extracted from the current block using its boundary curves:
            - lower boundary: U-line at `jstart`, columns `istart..iend`
            - upper boundary: U-line at `jend`,   columns `istart..iend`
            - left boundary:  V-line at `istart`, rows    `jstart..jend`
            - right boundary: V-line at `iend`,   rows    `jstart..jend`

        The interior of the returned sub-block is generated via transfinite
        interpolation using these four boundaries (see :meth:`transfinite`).
        The resulting mesh has exactly:
            `(iend - istart + 1)` points in u-direction and
            `(jend - jstart + 1)` points in v-direction.

        :param ij: index bounds `[istart, iend, jstart, jend]`.
        :type ij: list[int]
        :return: new :class:`BlockMesh` instance corresponding to the sub-block.
        :rtype: BlockMesh
        """

        istart, iend, jstart, jend = ij

        U, V = self.getDivUV()
        lower_u = self.getLine(number=jstart, direction="u")  # U-line at jstart
        upper_u = self.getLine(number=jend, direction="u")  # U-line at jend
        left_v = self.getLine(number=istart, direction="v")  # V-line at istart
        right_v = self.getLine(number=iend, direction="v")  # V-line at iend

        # Slice to the requested subrange
        lower = lower_u[istart : iend + 1]
        upper = upper_u[istart : iend + 1]
        left = left_v[jstart : jend + 1]
        right = right_v[jstart : jend + 1]

        # Build boundary curves from the parent mesh
        sub = BlockMesh()
        sub.transfinite(boundary=[lower, upper, left, right])

        return sub

    def extrudeLine_spacing(self, line, lengths, direction):
        """Extrude a polyline in a specified direction using explicit spacing.

        :param line: base polyline as list of `(x, y)` tuples.
        :type line: list[tuple[float, float]]
        :param lengths: list/array of layer thicknesses (not cumulative).
        :type lengths: array-like
        :param direction: extrusion direction vector `(nx, ny)`.
        :type direction: tuple[float, float] | list[float] | np.ndarray
        :return: None
        :rtype: None
        """
        x, y = list(zip(*line))
        x = np.array(x)
        y = np.array(y)

        # Normalize the custom direction
        nx, ny = direction
        norm = np.sqrt(nx**2 + ny**2)
        nx /= norm
        ny /= norm

        spacing = np.insert(np.cumsum(lengths), 0, 0.0)
        for i in range(0, len(spacing)):
            xo = x + spacing[i] * nx
            yo = y + spacing[i] * ny
            new_line = list(zip(xo.tolist(), yo.tolist()))
            self.addLine(new_line)

    @staticmethod
    def spacing_cell_thickness(cell_thickness=0.04, growth=1.1, extrusion_distance=0.4):
        """Compute cumulative layer offsets using geometric growth.

        Generates a list of cumulative offsets starting at 0, with the first layer
        thickness equal to `cell_thickness` and subsequent thicknesses growing by
        the constant ratio `growth` until reaching (or slightly exceeding)
        `extrusion_distance`.

        :param cell_thickness: first-layer thickness (t).
        :type cell_thickness: float, optional
        :param growth: geometric growth ratio (r).
        :type growth: float, optional
        :param extrusion_distance: total extrusion distance / boundary-layer height (L).
        :type extrusion_distance: float, optional
        :return: tuple `(spacing, length)` where
            - `spacing` is a list of cumulative offsets including 0.0,
            - `length` is the sum of the generated cumulative offsets.
        :rtype: tuple[list[float], float]
        """
        # add cell thickness of first layer
        if cell_thickness <= 0:
            raise ValueError("cell_thickness must be > 0")
        if growth <= 0:
            raise ValueError("growth must be > 0")
        if extrusion_distance <= 0:
            return [0.0], 0.0

        # Uniform spacing case: r = 1
        if np.isclose(growth, 1.0):
            divisions = int(np.ceil(extrusion_distance / cell_thickness))
            spacing = [i * cell_thickness for i in range(divisions + 1)]

            length = spacing[-1]
            return spacing, length
        
        spacing = [cell_thickness]
        N = np.log(1 + (growth - 1) * extrusion_distance / cell_thickness) / np.log(
            growth
        )
        divisions = int(np.ceil(N))

        for i in range(divisions - 1):
            spacing.append(spacing[0] + spacing[-1] * growth)

        spacing.insert(0, 0.0)
        length = np.sum(spacing)

        return spacing, length

    @staticmethod
    def unit_vector(vector):
        """Return the unit vector of the given vector.

        :param vector: input vector.
        :type vector: np.ndarray
        :return: normalized vector with unit Euclidean norm.
        :rtype: np.ndarray
        """
        return vector / np.linalg.norm(vector)

    def getVLines(self):
        """Return all V-lines derived from the stored U-lines.

        A V-line is constructed by collecting points with the same `I` index
        across all U-lines.

        :return: list of V-lines, each a list of `(x, y)` tuples.
        :rtype: list[list[tuple[float, float]]]
        """
        vlines = list()
        U, V = self.getDivUV()

        # loop over all u-lines
        for i in range(U + 1):
            # prepare new v-line
            vline = list()
            # collect i-th point on each u-line
            for uline in self.getULines():
                vline.append(uline[i])
            vlines.append(vline)

        return vlines

    def getDivUV(self):
        """Return the number of divisions in u and v directions.

        For a structured grid, if there are `U+1` points along each U-line and
        `V+1` U-lines total, then the number of divisions is `(U, V)`.

        :return: `(U, V)` where U is divisions in u-direction and V in v-direction.
        :rtype: tuple[int, int]
        """
        u = len(self.getULines()[0]) - 1
        v = len(self.getULines()) - 1
        return u, v

    def getULines(self):
        """Return the stored U-lines.

        :return: list of U-lines.
        :rtype: list[list[tuple[float, float]]]
        """
        return self.ULines

    def getLine(self, number=0, direction="u"):
        """Return a specific line in u- or v-direction.

        :param number: index of the requested line.
        :type number: int, optional
        :param direction: `'u'` for U-lines or `'v'` for V-lines.
        :type direction: str, optional
        :return: requested line as list of `(x, y)` tuples.
        :rtype: list[tuple[float, float]]
        """
        if direction.lower() == "u":
            lines = self.getULines()
        if direction.lower() == "v":
            lines = self.getVLines()
        return lines[number]

    def transfinite(self, boundary=[], ij=[]):
        """Generate interior nodes by transfinite interpolation (TFI).

        This method fills a structured block from four boundary curves (lower,
        upper, left, right). Boundaries can be provided explicitly via `boundary`
        or extracted from the current mesh via `ij`.

        Reference:
        http://en.wikipedia.org/wiki/Transfinite_interpolation

                       upper
                --------------------
                |                  |
                |                  |
           left |                  | right
                |                  |
                |                  |
                --------------------
                       lower

        :param boundary: optional list `[lower, upper, left, right]`, where each
            entry is a list of `(x, y)` tuples.
        :type boundary: list[list[tuple[float, float]]], optional
        :param ij: optional subrange indices `[istart, iend, jstart, jend]` used to
            extract boundaries from the current mesh.
        :type ij: list[int], optional
        :return: None
        :rtype: None
        """

        if boundary:
            lower = boundary[0]
            upper = boundary[1]
            left = boundary[2]
            right = boundary[3]
        elif ij:
            lower = self.getULines()[ij[2]][ij[0] : ij[1] + 1]
            upper = self.getULines()[ij[3]][ij[0] : ij[1] + 1]
            left = self.getVLines()[ij[0]][ij[2] : ij[3] + 1]
            right = self.getVLines()[ij[1]][ij[2] : ij[3] + 1]
        else:
            lower = self.getULines()[0]
            upper = self.getULines()[-1]
            left = self.getVLines()[0]
            right = self.getVLines()[-1]

        # FIXME
        # FIXME left and right need to swapped from input
        # FIXME
        # FIXME like: left, right = right, left
        # FIXME

        lower = np.array(lower)
        upper = np.array(upper)
        left = np.array(left)
        right = np.array(right)

        # convert the block boundary curves into parametric form
        # as curves need to be between 0 and 1
        # interpolate B-spline through data points
        # here, a linear interpolant is derived "k=1"
        # splprep returns:
        # tck ... tuple (t,c,k) containing the vector of knots,
        #         the B-spline coefficients, and the degree of the spline.
        #   u ... array of the parameters for each given point (knot)
        tck_lower, u_lower = splprep(lower.T, s=0, k=1)
        tck_upper, u_upper = splprep(upper.T, s=0, k=1)
        tck_left, u_left = splprep(left.T, s=0, k=1)
        tck_right, u_right = splprep(right.T, s=0, k=1)

        nodes = np.zeros((len(left) * len(lower), 2))

        # corner points
        c1 = lower[0]
        c2 = upper[0]
        c3 = lower[-1]
        c4 = upper[-1]

        for i, xi in enumerate(u_lower):
            for j, eta in enumerate(u_left):

                node = i * len(u_left) + j

                point = (
                    (1.0 - xi) * left[j]
                    + xi * right[j]
                    + (1.0 - eta) * lower[i]
                    + eta * upper[i]
                    - (
                        (1.0 - xi) * (1.0 - eta) * c1
                        + (1.0 - xi) * eta * c2
                        + xi * (1.0 - eta) * c3
                        + xi * eta * c4
                    )
                )

                nodes[node, 0] = point[0]
                nodes[node, 1] = point[1]

        vlines = list()
        vline = list()
        i = 0
        for node in nodes:
            i += 1
            vline.append(node)
            if i % len(left) == 0:
                vlines.append(vline)
                vline = list()

        vlines.reverse()

        if ij:
            ulines = self.makeUfromV(vlines)
            n = -1
            for k in range(ij[2], ij[3] + 1):
                n += 1
                self.ULines[k][ij[0] : ij[1] + 1] = ulines[n]
        else:
            self.ULines = self.makeUfromV(vlines)

        return

    @staticmethod
    def makeUfromV(vlines):
        """Convert a list of V-lines into a list of U-lines.

        This is a helper to switch the internal representation after generating
        nodes in V-line order.

        :param vlines: list of V-lines.
        :type vlines: list[list[tuple[float, float]]]
        :return: list of U-lines.
        :rtype: list[list[tuple[float, float]]]
        """
        ulines = list()
        uline = list()
        for i in range(len(vlines[0])):
            for vline in vlines:
                x, y = vline[i][0], vline[i][1]
                uline.append((x, y))
            ulines.append(uline[::-1])
            uline = list()
        return ulines
