import math
import numpy as np
import matplotlib.pyplot as plt


def affine(x: float, x1: float, x2: float, y1: float, y2: float) -> float:
    """Return the affine function that goes through (x1, y1) and (x2, y2)"""

    return ((y2 - y1) / (x2 - x1)) * (x - x1) + y1


class AffineFunc:
    def __init__(self, x1: float, x2: float, y1: float, y2: float):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def __call__(self, x: float) -> float:
        return ((self.y2 - self.y1) / (self.x2 - self.x1)) * (x - self.x1) + self.y1


class ERFFunc:
    """Generates a speed function based on a erf.
    x can be an angle or a distance... or anything else.
    """

    def __init__(
        self,
        max_speed: float,
        min_speed: float,
        x_for_max_speed: float,
        x_for_min_speed: float,
    ):
        self.max_speed = max_speed
        self.min_speed = min_speed
        self.x_for_max_speed = x_for_max_speed
        self.x_for_min_speed = x_for_min_speed

        self.affine = AffineFunc(x1=x_for_min_speed, x2=x_for_max_speed, y1=-2, y2=2)

    def __call__(self, x: float) -> float:
        """Returns the speed of the robot in function of x."""

        if x >= self.x_for_max_speed:
            return self.max_speed

        return (self.max_speed - self.min_speed) * self._erf_remap(x) + self.min_speed

    def _erf_remap(self, x: float) -> float:
        """erf remaped from [0, X_FOR_MAX_SPEED] to [0, 1]"""

        return (1 + math.erf(self.affine(x))) / 2

    def show(self):
        """Display a plot of the speed function."""
        dist = np.linspace(0.8 * self.x_for_min_speed, 1.2 * self.x_for_max_speed, 100)

        speed = np.vectorize(self)(dist)

        plt.plot(dist, speed)
        plt.xlabel("x")
        plt.ylabel("Speed")
        plt.title("Speed function")
        plt.show()
