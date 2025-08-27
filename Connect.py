from scipy import spatial
import numpy as np

class Connect:
    def __init__(self):
        pass

    def getConnectivity(self, block):

        connectivity = list()

        U, V = block.getDivUV()
        up = U + 1
        for u in range(U):
            for v in range(V):
                p1 = v * up + u
                p2 = p1 + up
                p3 = p2 + 1
                p4 = p1 + 1
                connectivity.append((p1, p2, p3, p4))

        return connectivity

    def getVertices(self, block):
        """Make a list of point tuples from a BlockMesh object

        Args:
            block (BlockMesh): BlockMesh object

        Returns:
            list: list of point tuples
                  # [(x1, y1), (x2, y2), (x3, y3), ... , (xn, yn)]
        """
        vertices = list()
        for uline in block.getULines():
            vertices += uline
        return vertices

    def getNearestNeighboursPairs(self, vertices, radius=1.e-6):
        tree = spatial.cKDTree(vertices)
        pairs = tree.query_pairs(radius, p=2., eps=0)
        return pairs

    def getNearestNeighbours(self, vertices, neighbours, radius=1.e-6):
        """Get the nearest neighbours to each vertex in a list of vertices
        uses Scipy kd-tree for quick nearest-neighbor lookup

        Args:
            vertices (list of tuples): Vertices for which nearest neighbours
                                       should be searched
            neighbours (list of tuples): These are the neighbours which
                                         are being searched
            radius (float, optional): Search neighbours within this radius

        Returns:
            vertex_and_neighbours(dictionary): Contains vertices searched
                                               as key and a list of nearest
                                               neighbours as values
        """

        # setup k-dimensional tree
        tree = spatial.cKDTree(neighbours)

        vertex_and_neighbours = dict()
        for vertex_id, vertex in enumerate(vertices):
            vertex_and_neighbours[vertex_id] = \
                tree.query_ball_point(vertex, radius)

        return vertex_and_neighbours
    
    def shiftConnectivity(self, connectivity, shift):

        if shift == 0:
            return connectivity

        connectivity_shifted = list()
        for cell in connectivity:
            new_cell = [vertex + shift for vertex in cell]
            connectivity_shifted.append(new_cell)

        return connectivity_shifted
    
    def connectAllBlocks(self, blocks):

        # compile global vertex list and cell connectivity from all blocks
        vertices = list()
        connectivity = list()

        for i, block in enumerate(blocks):

            # accumulated number of vertices
            # for i = 0 shift is automatically 0
            # so the connectivity of the first block doesn't get shifted
            # thus, this variable must be set before 'vertices += ...'
            shift = len(vertices)

            # concatenate vertices of all blocks
            # vertices += [vertex for vertex in self.getVertices(block)]
            vertices += self.getVertices(block)

            # shift the block connectivity by accumulated number of vertices
            # from all blocks before this one
            connectivity_block = \
                self.shiftConnectivity(self.getConnectivity(block), shift)
            connectivity += [tuple(cell) for cell in connectivity_block]

    
        vertices = [(vertex[0], vertex[1]) for vertex in vertices]

        # search vertices of all blocks against themselves
        # finds itself AND multiple connections (i.e. vertices from neighbour blocks)
        # uses Scipy kd-tree for quick nearest-neighbor lookup
        # the distance tolerance is specified via the radius variable
        vertex_and_neighbours = self.getNearestNeighbours(vertices,
                                                          vertices,
                                                          radius=1.e-6)

        # substitute vertex ids in connectivity at block connections
        connectivity_connected = list()
        for cell in connectivity:
            cell_new = list()
            for node in cell:
                # if there is only one vertex in vertex_and_neighbours,
                # then it is taken as it is
                # if there is more than one vertex,
                # then the minimum vertex index is used
                # so a few vertices remain unused and need to be removed later
                node_new = min(vertex_and_neighbours[node])
                cell_new.append(node_new)
            connectivity_connected.append(cell_new)

        # use numpy arrays
        unconnected = np.array(connectivity)
        connected = np.array(connectivity_connected)

        # deleted nodes
        deleted_nodes = np.unique(unconnected[np.where(connected != unconnected)])

        # delete unused vertices
        vertices_clean = [v for i,v in enumerate(vertices)
                        if i not in sorted(deleted_nodes.tolist())]

        # find remaining node ids
        remaining_nodes = np.setdiff1d(np.unique(connected), deleted_nodes)

        # replace node ids so that a contiguous numbering is established
        # divakar, method 3 (https://stackoverflow.com/a/55950051/2264936)
        mapping = {rn:i for i, rn in enumerate(remaining_nodes)}
        k = np.array(list(mapping.keys()))
        v = np.array(list(mapping.values()))
        mapping_ar = np.zeros(k.max()+1,dtype=v.dtype)
        mapping_ar[k] = v
        connectivity_clean = mapping_ar[connected]

        return (vertices_clean, connectivity_clean)

    def write_obj(self, vertices, connectivity, filename):
        with open(filename, 'w') as f:
            # Write vertices
            for v in vertices:
                if len(v) == 2:
                    f.write(f"v {v[0]} {v[1]} 0.0\n")  # Add z=0 for 2D vertices
                else:
                    f.write(f"v {v[0]} {v[1]} {v[2]}\n")

            # Write faces (convert 0-based to 1-based indices)
            for face in connectivity:
                indices_str = ' '.join(str(i + 1) for i in face)
                f.write(f"f {indices_str}\n")