import numpy as np


def circle_traj(speed: float, angular_speed: float, t: float) -> "tuple[float, float]":
    """Return the position after a duration t when the robot has:
    - A normal speed speed
    - An angular speed angular_speed
    """

    if angular_speed == 0:
        return speed * t, 0.0

    if angular_speed > 0:
        dir = 1
    else:
        dir = -1

    r = abs(speed / angular_speed)  # Radius of the circle
    theta = abs(angular_speed) * t

    if theta > 2 * np.pi:
        raise ValueError("Theta is superior to 2pi")

    return r * np.sin(theta), dir * (r - r * np.cos(theta))
