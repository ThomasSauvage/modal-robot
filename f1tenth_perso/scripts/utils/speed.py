import math
from scipy.interpolate import lagrange

import numpy as np
import matplotlib.pyplot as plt


def affine(x: float, x1: float, x2: float, y1: float, y2: float) -> float:
    """Return the affine function that goes through (x1, y1) and (x2, y2)"""

    return ((y2 - y1) / (x2 - x1)) * (x - x1) + y1


class Function:
    def __call__(self, x: float):
        raise NotImplementedError

    def show(self):
        """Display a plot of the speed function."""
        dist = np.linspace(0.8 * self.x_for_min_speed, 1.2 * self.x_for_max_speed, 100)  # type: ignore

        speed = np.vectorize(self)(dist)

        plt.plot(dist, speed)
        plt.xlabel("x")
        plt.ylabel("Speed")
        plt.title("Speed function")
        plt.show()


class AffineFunc:
    def __init__(self, x1: float, x2: float, y1: float, y2: float):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def __call__(self, x: float) -> float:
        return ((self.y2 - self.y1) / (self.x2 - self.x1)) * (x - self.x1) + self.y1


class ERF(Function):
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


class ReversedERF(Function):
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
        if x_for_max_speed >= x_for_min_speed:
            raise ValueError("We should have x_for_max_speed < x_for_min_speed")

        self.max_speed = max_speed
        self.min_speed = min_speed
        self.x_for_max_speed = x_for_max_speed
        self.x_for_min_speed = x_for_min_speed

        self.x_mid = (x_for_max_speed + x_for_min_speed) / 2

        self.erf = ERF(max_speed, min_speed, x_for_min_speed, x_for_max_speed)

    def __call__(self, x: float) -> float:
        """Returns the speed of the robot in function of x."""

        new_x = self.x_mid + (self.x_mid - x)

        return self.erf(new_x)

    def show(self):
        """Display a plot of the speed function."""
        dist = np.linspace(self.x_for_min_speed, self.x_for_max_speed, 100)  # type: ignore
        print(dist)

        speed = np.vectorize(self)(dist)

        plt.plot(dist, speed)
        plt.xlabel("x")
        plt.ylabel("Speed")
        plt.title("Speed function")
        plt.show()


class _ClassicGauss(Function):
    def __init__(
        self,
        mu: float,
        sigma: float,
    ):
        self.mu = mu
        self.sigma = sigma

        self.norm_coef = 1 / (sigma * math.sqrt(2 * math.pi))
        self.int_coef = 1 / (2 * sigma**2)

    def __call__(self, x: float) -> float:
        return math.exp(-((x - self.mu) ** 2) * self.int_coef) * self.norm_coef


class Gauss(Function):
    """
    Generates a speed function based on a binomial.
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

        self.cgauss = _ClassicGauss(mu=x_for_max_speed, sigma=x_for_min_speed / 3)

    def __call__(self, x: float) -> float:
        """Returns the speed of the robot in function of x."""

        if abs(x) >= self.x_for_min_speed:
            return self.min_speed

        return (self.max_speed - self.min_speed) * self.cgauss(x) + self.min_speed


class Polynomial:
    def __init__(self, xs: np.ndarray, ys: np.ndarray):
        self.min_i = np.argmin(xs)
        self.max_i = np.argmax(xs)

        self.xs = xs
        self.ys = ys

        self.poly = lagrange(xs, ys)

    def __call__(self, x: float) -> float:

        if x < self.xs[self.min_i]:
            return self.ys[self.min_i]

        if x > self.xs[self.max_i]:
            return self.ys[self.max_i]

        return self.poly(x)

    def show(self):
        """Display a plot of the speed function."""

        dist = np.linspace(0.8 * self.xs[self.min_i], 1.2 * self.xs[self.max_i], 100)  # type: ignore

        speed = np.vectorize(self)(dist)

        plt.plot(dist, speed)
        plt.xlabel("x")
        plt.ylabel("Speed")
        plt.title("Speed function")
        plt.show()


if __name__ == "__main__":
    """
    poly = Polynomial(
        xs=np.array([0, 1, 0.5]),
        ys=np.array([7, 2]),
    )

    poly.show()
    """
    speed_function = ReversedERF(
        max_speed=7, min_speed=2, x_for_max_speed=0, x_for_min_speed=1.1
    )
    speed_function.show()
