"""
This code is taken and adapted according to requirement from PyAero repository.
GitHub: https://github.com/chiefenne/PyAero 
"""

from scipy import spatial
import numpy as np
class Connect:
    """Connectivity and vertex utilities for block-structured meshes.

    This helper class provides routines to:
    - build cell connectivity (quad elements) for a :class:`BlockMesh`,
    - extract a flattened vertex list from one or more blocks,
    - detect and merge near-duplicate vertices using a KD-tree, and
    - assemble multiple blocks (and optional TRI3 patches) into a single,
      globally connected mesh with compact node numbering.

    """
    def __init__(self):
        pass

    def getConnectivity(self, block):
        """Generate quad-cell connectivity for a structured :class:`BlockMesh`.

        :param block: block from which to generate connectivity.
        :type block: BlockMesh
        :return: list of quad cells as tuples of four vertex indices.
        :rtype: list[tuple[int, int, int, int]]
        """
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
        """Flatten a :class:`BlockMesh` into a global vertex list for that block.

        Vertices are returned in the same order used by :meth:`getConnectivity`,
        i.e., concatenating all U-lines in sequence.

        :param block: block to extract vertices from.
        :type block: BlockMesh
        :return: list of `(x, y)` vertex tuples:
            `[(x1, y1), (x2, y2), ..., (xn, yn)]`.
        :rtype: list[tuple[float, float]]
        """
        vertices = list()
        for uline in block.getULines():
            vertices += uline
        return vertices

    def getNearestNeighboursPairs(self, vertices, radius=1.e-8):
        """Find pairs of vertices within a given distance tolerance.

        :param vertices: vertex coordinates.
        :type vertices: array-like of shape (N, 2)
        :param radius: distance tolerance used to detect duplicates.
        :type radius: float, optional
        :return: set of index pairs within the tolerance.
        :rtype: set[tuple[int, int]]
        """
        tree = spatial.cKDTree(vertices)
        pairs = tree.query_pairs(radius, p=2., eps=0)
        return pairs

    def getNearestNeighbours(self, vertices, neighbours, radius=1.e-8):
        """Map each vertex to indices of neighbour points within a radius.

        Builds a KD-tree on `neighbours` and for each vertex in `vertices`
        finds all neighbour indices within `radius` using `query_ball_point`.

        This is typically used to identify near-duplicate vertices across
        different blocks when assembling a global mesh.

        :param vertices: query points (vertices to be searched).
        :type vertices: array-like of shape (N, 2)
        :param neighbours: candidate neighbour points (searched set).
        :type neighbours: array-like of shape (M, 2)
        :param radius: search tolerance (Euclidean distance).
        :type radius: float, optional
        :return: dictionary mapping `vertex_id -> [neighbour_ids]`, where
            the neighbour ids are indices into the `neighbours` array.
        :rtype: dict[int, list[int]]
        """
        # setup k-dimensional tree
        tree = spatial.cKDTree(neighbours)

        vertex_and_neighbours = dict()
        for vertex_id, vertex in enumerate(vertices):
            vertex_and_neighbours[vertex_id] = \
                tree.query_ball_point(vertex, radius)

        return vertex_and_neighbours
    
    def shiftConnectivity(self, connectivity, shift):
        """Shift connectivity indices by a constant offset.

        Useful when concatenating multiple blocks into a single global
        connectivity array.

        :param connectivity: list of cells; each cell is an iterable of indices.
        :type connectivity: list[tuple[int, ...]] | list[list[int]]
        :param shift: integer offset added to every index.
        :type shift: int
        :return: shifted connectivity in the same cell topology.
        :rtype: list[list[int]] | list[tuple[int, ...]]
        """
        if shift == 0:
            return connectivity

        connectivity_shifted = list()
        for cell in connectivity:
            new_cell = [vertex + shift for vertex in cell]
            connectivity_shifted.append(new_cell)

        return connectivity_shifted
    
    def connectAllBlocks(self, blocks, trias):
        """Assemble multiple blocks into a single connected mesh.

        All vertices and cell connectivities from the provided blocks are
        concatenated into a global mesh. Near-duplicate vertices are detected
        using a KD-tree search and merged within a fixed tolerance. The final
        vertex list and connectivity are compacted to remove unused nodes.

        Optional triangulated patches (TRI3 elements) can be appended to the
        mesh and are included in the merging and renumbering process.

        :param blocks: list of structured block meshes.
        :type blocks: list[BlockMesh]
        :param trias: list of triangulated patches. Each entry must contain
            the keys ``"P"`` (point coordinates of shape ``(N,2)`` or ``(N,3)``)
            and ``"connectivity"`` (integer array of shape ``(M,3)``).
        :type trias: list[dict]
        :return: tuple ``(vertices, connectivity)`` where
            ``vertices`` is the list of unique vertex coordinates and
            ``connectivity`` is the global cell connectivity.
        :rtype: tuple[list[tuple[float, float]], list[list[int]]]
        :raises ValueError: if triangulated patch data has invalid shape.
        """
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


        for d in trias:
            P3 = np.asarray(d["P"], dtype=float)
            T  = np.asarray(d["connectivity"], dtype=int)
            if P3.ndim != 2 or P3.shape[1] < 2:
                raise ValueError("tria P must be (N,2) or (N,3)")
            if T.ndim != 2 or T.shape[1] != 3:
                raise ValueError("tria connectivity must be (M,3) for TRI3")
            shift = len(vertices)
            verts_t = [(float(x), float(y)) for x, y, *rest in P3]
            vertices += verts_t
            con_t = [[int(i) + shift, int(j) + shift, int(k) + shift] for i, j, k in T]
            connectivity += con_t
    
        vertices = [(vertex[0], vertex[1]) for vertex in vertices]

        # search vertices of all blocks against themselves
        # finds itself AND multiple connections (i.e. vertices from neighbour blocks)
        # uses Scipy kd-tree for quick nearest-neighbor lookup
        # the distance tolerance is specified via the radius variable
        vertex_and_neighbours = self.getNearestNeighbours(vertices,
                                                          vertices,
                                                          radius=1.e-8)

        connectivity_connected = []
        for cell in connectivity:
            # replace each node by the smallest index among its near-duplicates
            connectivity_connected.append([min(vertex_and_neighbours[n]) for n in cell])

        # 4) Build compact numbering: keep only vertices that are actually referenced
        used_old_ids = sorted({n for cell in connectivity_connected for n in cell})
        old_to_new = {old: new for new, old in enumerate(used_old_ids)}

        vertices_clean = [vertices[old] for old in used_old_ids]
        connectivity_clean = [[old_to_new[n] for n in cell] for cell in connectivity_connected]

        return vertices_clean, connectivity_clean


