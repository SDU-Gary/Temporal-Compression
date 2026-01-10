"""
Cubic Spline Interpolation Baseline

Fits independent cubic splines for each of the 27 SH coefficients.
Simple, deterministic baseline for time interpolation.
"""

from scipy.interpolate import CubicSpline
import numpy as np


class SplineBaseline:
    """Cubic spline interpolation baseline"""

    def __init__(self):
        self.splines = None

    def train(self, hours, sh_gt):
        """
        Fit cubic splines for each SH coefficient

        Args:
            hours: [N] array of training hours
            sh_gt: [N, 27] array of ground truth SH coefficients
        """
        self.splines = []
        for i in range(27):
            spline = CubicSpline(hours, sh_gt[:, i])
            self.splines.append(spline)

    def predict(self, hours_query):
        """
        Interpolate SH at query hours

        Args:
            hours_query: [M] array of query hours

        Returns:
            [M, 27] array of interpolated SH coefficients
        """
        if self.splines is None:
            raise ValueError("Model not trained. Call train() first.")

        predictions = []
        for spline in self.splines:
            predictions.append(spline(hours_query))

        return np.stack(predictions, axis=-1)  # [M, 27]
