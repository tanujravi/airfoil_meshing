"""Airfoil contour tools and geometrical properties.
"""

from typing import Self, Tuple
import logging
import numpy as np
from pandas import read_csv
from scipy.interpolate import BSpline, make_interp_spline
import scipy
import Elliptic
logging.basicConfig(
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
    level=logging.INFO
)


class Airfoil(object):
    """Container class for airfoil contour tools.
    """
    def __init__(self, x: np.ndarray, y: np.ndarray, k: int=2):
        """Construct airfoil contour from points.

        The following assumptions about the airfoil are made:
        - the x-coordinate is scaled to [0, 1]
        - the airfoil has a blunt trailing edge (TE)
        - the points are sorted from the lower to the upper TE

        :param x: x-coordinate of contour points
        :type x: np.ndarray
        :param y: y-coordinate of contour points
        :type y: np.ndarray
        :param k: polynomial degree of spline contour approximation,
            defaults to 2
        :type k: int, optional
        """
        self._x = x
        self._y = y
        self._sp_x, self._sp_y = self._fit_contour(k)

    @classmethod
    def from_contour_file(cls, filename: str, k: int=2) -> Self:
        """Create airfoil from contour data in textfile.

        :param filename: contour data filename or path to file
        :type filename: str
        :param k: polynomial degree of spline contour approximation,
            defaults to 2
        :type k: int, optional
        :return: instance of `Airfoil`
        :rtype: Self
        """
        df = read_csv(filename, comment="#", sep=r"\s+", header=None, names=("x", "y"))
        return cls(df.x, df.y, k)
    
    def _fit_contour(self, k: int) -> Tuple[BSpline, BSpline]:
        """Create contour spline approximation.

        Both coordinates are parametrized using an arc length normalized
        to [0, 1] to create a smooth fit. Note that the trailing edge is
        not included.

        :param k: polynomial degree of spline contour approximation
        :type k: int
        :return: `BSpline` instances for both coordinates
        :rtype: Tuple[BSpline, BSpline]
        """
        s = np.cumsum(np.linalg.norm(np.vstack((np.diff(self._x), np.diff(self._y))), axis=0))
        s = np.concatenate(([0], s))
        s /= s[-1]
        return make_interp_spline(s, self._x, k), make_interp_spline(s, self._y, k)
    
    def _evaluate_contour(self, n_points: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate contour splines for a given number of points.

        Note that the trailing edge is not included in the evaluation.

        :param n_points: number of points to distribute on the contour
        :type n_points: int
        :return: interpolated contour points as x- and y-coordinates and
            normalized arc length
        :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
        """
        u = np.linspace(0, 1, n_points)
        x = self._sp_x(u)
        y = self._sp_y(u)
        return x, y, u
    
    def _evaluate_curvature(self, n_points: int) -> np.ndarray:
        """Spline-based evaluation of contour curavture.

        :param n_points: number of evaluation points on the contour
        :type n_points: int
        :return: curvature magnitude normalized to [0, 1]
        :rtype: np.ndarray
        """
        u = np.linspace(0, 1, n_points)
        dx = self._sp_x.derivative(1)(u)
        dy = self._sp_y.derivative(1)(u)
        ddx = self._sp_x.derivative(2)(u)
        ddy = self._sp_y.derivative(2)(u)
        curv = np.abs(dx*ddy - dy*ddx) / (dx**2 + dy**2)**1.5
        return curv / curv.max()


    def distribute_points(
            self,
            n_points: int,
            n_points_te: int=10,
            weight_upper: int=1.0,
            weight_curvature: int=4.0,
            weight_te: int=4.0,
            fraction_te: float=0.2,
            max_size_ratio: int=1.15,
            n_points_high_res: int=5000,
            max_relax_iter: int=100
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Distribute contour points for CFD-meshing.

        :param n_points: number of points to distribute, exluding TE
        :type n_points: int
        :param n_points_te: number of points on the TE, includes contour
            start and end points, defaults to 10
        :type n_points_te: int, optional
        :param weight_upper: bias point distribution toward upper side,
            0.0 yields an unbiased distribution, defaults to 1.0
        :type weight_upper: int, optional
        :param weight_curvature: bias point distribution toward regions
            of high absolute curvature, defaults to 4.0
        :type weight_curvature: int, optional
        :param weight_te: bias point distribution toward trailing edge,
            quadratic weight decay with respected to distance from TE,
            defaults to 4.0
        :type weight_te: int, optional
        :param fraction_te: fraction of chord in which apply TE bias,
            defaults to 0.2
        :type fraction_te: float, optional
        :param max_size_ratio: maximum allowed size ratio between two neighboring
            contour elements (larger over smaller element length), if exceeded, point
            distribution is smoothed iteratively, defaults to 1.15
        :type max_size_ratio: int, optional
        :param n_points_high_res: number of points for reference evaluation of contour
            properties, should be much higher than number of points to distribute,
            defaults to 5000
        :param max_relax_iter: maximum number of smoothing iterations to relax point
            distribution, defaults to 100
        :type max_relax_iter: int, optional
        :return: points on general contour and TE (x, y, x_te, y_te)
        :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        """
        x, y, u = self._evaluate_contour(n_points_high_res)
        ds = np.linalg.norm(np.vstack((np.diff(x), np.diff(y))), axis=0)
        s = np.hstack(([0], np.cumsum(ds)))
        curv = self._evaluate_curvature(n_points_high_res)
        chord = x.max() - x.min()
        dist_te = (x.max() - x) / chord
        weights = 1.0 + weight_curvature * curv \
            + weight_te * np.maximum((fraction_te  - dist_te) / fraction_te, 0.0)**2 \
            + weight_upper * (0.5 + 0.5 * np.tanh(s - np.mean(s)))
        weights[ y > 0] *= weight_upper
        weighted = np.hstack(([0], np.cumsum(weights[1:] * ds)))
        weighted /= weighted[-1]
        target_weights = np.linspace(0, 1, n_points)
        s_new = np.interp(target_weights, weighted, s)
        def stretching_statistics(s: np.ndarray) -> Tuple[float, float, float]:
            ds = np.diff(s)
            r = ds[1:] / ds[:-1]
            return np.mean(r), np.min(r), np.max(r)
        def relax_stretching(s: np.ndarray, max_ratio: int):
            _, rmin, rmax = stretching_statistics(s)
            limit = abs(1.0 - max_ratio)
            return max(abs(1.0 - rmin), abs(1.0 - rmax)) > limit
        logging.info(
            "length ratio of neighboring surface elements (mean, min, max): " +
            "{:1.4f}, {:1.4f}, {:1.4f}".format(*stretching_statistics(s_new))
            )
        if relax_stretching(s_new, max_size_ratio):
            logging.info("relaxing surface point distribution")
            uniform = np.linspace(0, 1, len(weighted))
            for i in range(max_relax_iter):
                weighted = 0.9 * weighted + 0.1 * uniform
                s_new = np.interp(target_weights, weighted, s)
                logging.info(f"iter. {i} (mean, min, max): " + "{:1.4f}, {:1.4f}, {:1.4f}".format(*stretching_statistics(s_new)))
                if not relax_stretching(s_new, max_size_ratio):
                    logging.info(f"relaxation converged after {i+1:d} iterations")
                    break
                if i==max_relax_iter-1:
                    logging.info(f"relaxation did not converge within {max_relax_iter} iterations")

        u_interp = np.interp(s_new, s, u)
        dx = self._sp_x.derivative(1)(u_interp)
        dy = self._sp_y.derivative(1)(u_interp)        
        normals = np.vstack([-dy, dx]).T
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals_unit = normals / norms
        x_te = np.linspace(x[0], x[-1], n_points_te)
        y_te = np.linspace(y[0], y[-1], n_points_te)
        return self._sp_x(u_interp), self._sp_y(u_interp), x_te, y_te, normals_unit
    
    @property
    def x(self) -> np.ndarray:
        """x-coordinates of original contour data.

        :return: x-coordinates of contour points
        :rtype: np.ndarray
        """
        return self._x
    
    @property
    def y(self) -> np.ndarray:
        """x-coordinates of original contour data.

        :return: y-coordinates of contour points
        :rtype: np.ndarray
        """
        return self._y


