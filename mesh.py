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

    def write_obj(self, filename):
        with open(filename, 'w') as f:
            # Write vertices
            vertices, connectivity = self.mesh
            for v in vertices:
                if len(v) == 2:
                    f.write(f"v {v[0]} {v[1]} 0.0\n")  # Add z=0 for 2D vertices
                else:
                    f.write(f"v {v[0]} {v[1]} {v[2]}\n")

            # Write faces (convert 0-based to 1-based indices)
            for face in connectivity:
                indices_str = ' '.join(str(i + 1) for i in face)
                f.write(f"f {indices_str}\n")


