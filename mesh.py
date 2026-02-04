import numpy as np
import os
import logging
logger = logging.getLogger(__name__)
"""
This code is taken in part and adapted according to requirement from PyAero repository.
GitHub: https://github.com/chiefenne/PyAero 
"""


class Mesh:
    def __init__(self, vertices, connectivity):
        # add mesh to Wind-tunnel instance
        self.mesh = vertices, connectivity

        # generate cell to vertex connectivity from mesh
        self.makeLCV()

        # generate cell to edge connectivity from mesh
        self.makeLCE()

        # generate boundaries from mesh connectivity
        self.makeBoundaries()
    
    def makeLCV(self):
        """Make cell to vertex connectivity for the mesh
           LCV is identical to connectivity
        """
        _, connectivity = self.mesh
        self.LCV = connectivity


    def makeLCE(self):
        """Make cell to edge connectivity for the mesh"""
        _, connectivity = self.mesh
        self.LCE = dict()
        self.edges = list()

        for i, cell in enumerate(connectivity):
            # example for quadrilateral:
            # cell: [0, 1, 5, 4]
            # edges: [(0,1), (1,5), (5,4), (4,0)]
            edges = [(cell[j], cell[(j + 1) % len(cell)])
                           for j in range(len(cell))]

            # all edges for cell i
            self.LCE[i] = edges

            # all edges in one list
            self.edges += [tuple(sorted(edge)) for edge in edges]

    def makeBoundaries(self):
        """A boundary edge is an edge that belongs only to one cell"""

        vertices, _ = self.mesh
        vertices = np.array(vertices)

        edges = self.edges

        seen = set()
        unique = list()
        doubles = set()
        for edge in edges:
            if edge not in seen:
                seen.add(edge)
                unique.append(edge)
            else:
                doubles.add(edge)

        self.boundary_edges = [edge for edge in unique if edge not in doubles]

        # tag edges for boundary definitions
        # FIXME
        # FIXME here it's done the dirty way
        # FIXME at least try to make it faster later
        # FIXME
        self.boundary_tags = {'airfoil': [],
                              'inlet': [],
                              'outlet': [],
                              'top': [],
                              'bottom': []}
        
        ### FIXME
        ### FIXME too dirty below (do not work with toplerances!!!)
        ### FIXME

        xmax = np.max(vertices[:,0])
        y_vals = vertices[:, 1]

        unique_y, counts = np.unique(y_vals, return_counts=True)

        sorted_idx = np.argsort(counts)[::-1]

        top_two_y = unique_y[sorted_idx[:2]]

        ymax = np.max(top_two_y)
        ymin = np.min(top_two_y)

        for edge in self.boundary_edges:
            x1 = vertices[edge[0]][0]
            y1 = vertices[edge[0]][1]
            x2 = vertices[edge[1]][0]
            y2 = vertices[edge[1]][1]
            tol = 1e-6  # tolerance for coordinate comparison
            if x1 > -0.1 and x1 < 1.1 and y1 < 0.5 and y1 > -0.5:
                self.boundary_tags['airfoil'].append(edge)
            elif abs(x1 - xmax) < tol and abs(x2 - xmax) < tol:
                self.boundary_tags['outlet'].append(edge)
            elif abs(y1 - ymax) < tol and abs(y2 - ymax) < tol:
                self.boundary_tags['top'].append(edge)
            elif abs(y1 - ymin) < tol and abs(y2 - ymin) < tol:
                self.boundary_tags['bottom'].append(edge)
            else:
                self.boundary_tags['inlet'].append(edge)

        return
    
    def write_obj(self, filename):
        with open(filename, 'w') as f:
            # Write vertices
            vertices, _ = self.mesh
            for v in vertices:
                if len(v) == 2:
                    f.write(f"v {v[0]} {v[1]} 0.0\n")  # Add z=0 for 2D vertices
                else:
                    f.write(f"v {v[0]} {v[1]} {v[2]}\n")

            # Write faces (convert 0-based to 1-based indices)
            for face in self.LCV:
                indices_str = ' '.join(str(i + 1) for i in face)
                f.write(f"f {indices_str}\n")


