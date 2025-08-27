import numpy as np
from scipy.interpolate import splprep

class BlockMesh:
    def __init__(self):
        self.ULines = list()

    def addLine(self, line):
        # line is a list of (x, y) tuples
        self.ULines.append(line)    
    def setUlines(self, ulines):
        self.ULines = ulines
    def extrudeLine_cell_thickness(self, line, normals, cell_thickness=0.04,
                                   growth=1.05,
                                   extrusion_distance= 0.4):
        x, y = list(zip(*line))
        #x = np.array(x)
        #y = np.array(y)
        spacing, _ = self.spacing_cell_thickness(
            cell_thickness=cell_thickness,
            growth=growth,
            extrusion_distance=extrusion_distance)
        for i in range(0, len(spacing)):
            xo = x + spacing[i] * normals[:, 0]
            yo = y + spacing[i] * normals[:, 1]
            line = list(zip(xo.tolist(), yo.tolist()))
            self.addLine(line)

    def extrudeLine_spacing(self, line, lengths, direction):
        """
        Extrude a line in a fixed custom normal direction using specified spacing.

        Args:
            line: List of (x, y) tuples representing the original line.
            spacing: List of offsets (floats) from the original line.
            normal_direction: Tuple (nx, ny) specifying the direction to extrude in.
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

        # add cell thickness of first layer
        spacing = [cell_thickness]
        N = np.log(1 + (growth - 1) * extrusion_distance / cell_thickness) / np.log(growth)
        divisions = int(np.ceil(N))

        for i in range(divisions - 1):
            spacing.append(spacing[0] + spacing[-1] * growth)

        spacing.insert(0, 0.0)
        length = np.sum(spacing)

        return spacing, length
    @staticmethod
    def unit_vector(vector):
        """ Returns the unit vector of the vector.  """
        return vector / np.linalg.norm(vector)

    def getVLines(self):
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
        u = len(self.getULines()[0]) - 1
        v = len(self.getULines()) - 1
        return u, v

    def getULines(self):
        return self.ULines

    def getLine(self, number=0, direction='u'):
        if direction.lower() == 'u':
            lines = self.getULines()
        if direction.lower() == 'v':
            lines = self.getVLines()
        return lines[number]

    def transfinite(self, boundary=[], ij=[]):
        """Make a transfinite interpolation.

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

        Example input for the lower boundary:
            lower = [(0.0, 0.0), (0.1, 0.3),  (0.5, 0.4)]
        """

        if boundary:
            lower = boundary[0]
            upper = boundary[1]
            left = boundary[2]
            right = boundary[3]
        elif ij:
            lower = self.getULines()[ij[2]][ij[0]:ij[1] + 1]
            upper = self.getULines()[ij[3]][ij[0]:ij[1] + 1]
            left = self.getVLines()[ij[0]][ij[2]:ij[3] + 1]
            right = self.getVLines()[ij[1]][ij[2]:ij[3] + 1]
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

                point = (1.0 - xi) * left[j] + xi * right[j] + \
                    (1.0 - eta) * lower[i] + eta * upper[i] - \
                    ((1.0 - xi) * (1.0 - eta) * c1 + (1.0 - xi) * eta * c2 +
                     xi * (1.0 - eta) * c3 + xi * eta * c4)

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
                self.ULines[k][ij[0]:ij[1] + 1] = ulines[n]
        else:
            self.ULines = self.makeUfromV(vlines)

        return

    @staticmethod
    def makeUfromV(vlines):
        ulines = list()
        uline = list()
        for i in range(len(vlines[0])):
            for vline in vlines:
                x, y = vline[i][0], vline[i][1]
                uline.append((x, y))
            ulines.append(uline[::-1])
            uline = list()
        return ulines
