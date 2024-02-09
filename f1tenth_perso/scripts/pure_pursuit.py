#!/usr/bin/env python
import numpy as np
import math

import rospy
import tf2_ros
import tf2_geometry_msgs

from utils.speed import ERF, ReversedERF

from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Odometry


DRIVE_TOPIC = "/nav"  # "/vesc/ackermann_cmd_mux/input/navigation"
WAYPOINTS_FILENAME = "map.csv"

SHOW_WAYPOINTS = False

NBR_WAYPOINTS = 400

DIST_L = 1  # m
SMOOTH_ANGLE = 0.7

CURV_OVERHEAD = 15  # in indexes

speed_function = ReversedERF(
    max_speed=7, min_speed=1, x_for_max_speed=0, x_for_min_speed=1.22
)


def get_waypoints() -> np.ndarray:
    """Import waypoints"""

    with open(WAYPOINTS_FILENAME, "r") as f:
        waypoints_csv = f.readlines()

    print(len(waypoints_csv))

    waypoints = np.zeros((NBR_WAYPOINTS, 4))

    nbr_waypoints_per_line = len(waypoints_csv) // NBR_WAYPOINTS
    for i in range(NBR_WAYPOINTS):
        line = waypoints_csv[i * nbr_waypoints_per_line]
        waypoints[i] = line.split(",")

    return waypoints


# Structure of WAYPOINTS: x, y, ?, speed
WAYPOINTS = get_waypoints()


def get_marker_msg(
    x: float,
    y: float,
    id: int,
    color: "tuple[float, float, float]",
    frame_id: str = "base_link",
) -> Marker:
    """Return a Marker Message with the given position"""

    SCALE = 0.2

    marker_msg = Marker()
    marker_msg.header.frame_id = frame_id
    marker_msg.header.stamp = rospy.Time.now()
    marker_msg.ns = "marker1"
    marker_msg.id = id
    marker_msg.type = Marker.SPHERE
    marker_msg.action = Marker.ADD
    marker_msg.pose.position.x = x
    marker_msg.pose.position.y = y
    marker_msg.pose.position.z = 0
    marker_msg.pose.orientation.x = 0.0
    marker_msg.pose.orientation.y = 0.0
    marker_msg.pose.orientation.z = 0.0
    marker_msg.pose.orientation.w = 1.0
    marker_msg.scale.x = SCALE
    marker_msg.scale.y = SCALE
    marker_msg.scale.z = SCALE
    marker_msg.color.a = 1.0  # Don't forget to set the alpha!
    marker_msg.color.r = color[0]
    marker_msg.color.g = color[1]
    marker_msg.color.b = color[2]

    return marker_msg


def get_marker_array_waypoint() -> MarkerArray:
    """Return a Marker Message with the given position"""

    SCALE = 0.1

    marker_array = MarkerArray()

    for i, point in enumerate(WAYPOINTS):  # type: ignore
        marker_msg = Marker()
        marker_msg.header.frame_id = "map"
        # marker_msg.header.stamp = rospy.Time.now()
        marker_msg.ns = "marker1"
        marker_msg.id = i
        marker_msg.type = Marker.SPHERE
        marker_msg.pose.position.x = point[0]
        marker_msg.pose.position.y = point[1]
        marker_msg.pose.position.z = 0
        marker_msg.pose.orientation.x = 0.0
        marker_msg.pose.orientation.y = 0.0
        marker_msg.pose.orientation.z = 0.0
        marker_msg.pose.orientation.w = 1.0
        marker_msg.scale.x = SCALE
        marker_msg.scale.y = SCALE
        marker_msg.scale.z = SCALE
        marker_msg.color.a = 1  # Don't forget to set the alpha!
        marker_msg.color.r = 0.1
        marker_msg.color.g = 0.1
        marker_msg.color.b = 0.1

        # To only publish once and keep it persistent
        # marker_msg.lifetime = rospy.Duration() (doesn't work)

        marker_array.markers.append(marker_msg)  # type: ignore

    return marker_array


if SHOW_WAYPOINTS:
    WAYPOINTS_MARKER = get_marker_array_waypoint()


def get_nav_msg(angle: float, speed: float) -> AckermannDriveStamped:
    """Return an AckermannDriveStamped Message with the given steering angle"""

    drive_msg = AckermannDriveStamped()
    drive_msg.header.stamp = rospy.Time.now()
    drive_msg.header.frame_id = "base_link"
    drive_msg.drive.speed = speed
    drive_msg.drive.steering_angle = angle

    return drive_msg


class PurePursuit:
    """
    The class that handles pure pursuit.
    """

    def __init__(self):
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(buffer=self.tf_buffer)

        self.drive_pub = rospy.Publisher(
            DRIVE_TOPIC, AckermannDriveStamped, queue_size=10
        )
        self.marker_pub = rospy.Publisher("/dynamic_viz", Marker, queue_size=10)
        self.marker_array_pub = rospy.Publisher(
            "/dynamic_viz_array", MarkerArray, queue_size=10
        )

        rospy.Subscriber("/odom", Odometry, self.pose_callback)

        self.car_x = 0
        self.car_y = 0
        self.car_theta = 0

    def from_global_to_car_frame(self, x: float, y: float) -> np.ndarray:
        """Transform a point from the global frame to the car frame.

        Args:
            x (float): The x position of the point in the global frame.
            y (float): The y position of the point in the global frame.

        Returns:
            np.ndarray: The position of the point in the car frame.
        """

        norm = np.linalg.norm([x - self.car_x, y - self.car_y])
        new_x = norm * math.sin(self.car_theta)
        new_y = norm * math.cos(self.car_theta)

        return np.array([new_x, new_y])

    def get_target_point(
        self,
    ) -> "tuple[np.ndarray, np.ndarray]":
        """Get the target point for the pure pursuit algorithm.

        Returns:
            int: The index of the target point in the car frame.
            np.ndarray: The target point.
        """

        # Find the farthest waypoint of dist <= DIST_L and the closest waypoint of dist > DIST_L
        dists_to_car = np.linalg.norm(
            WAYPOINTS[:, :2] - np.array([self.car_x, self.car_y]), axis=1
        )
        values_far = dists_to_car.copy()
        values_far[values_far <= DIST_L] = np.inf

        while True:
            closest_waypoint_far = np.argmin(
                values_far
            )  # The closest waypoint that is outside the circle of radius L

            target_point = WAYPOINTS[closest_waypoint_far, :2]
            target_point_cf = self.from_global_to_car_frame(
                target_point[0], target_point[1]
            )

            # If the target point is in front of the car, return it
            if np.dot(target_point_cf, np.array([1, 0])) >= 0:  # type: ignore
                overhead_target = WAYPOINTS[
                    (closest_waypoint_far + CURV_OVERHEAD) % NBR_WAYPOINTS, :2  # type: ignore
                ]

                overhead_target_cf = self.from_global_to_car_frame(
                    overhead_target[0], overhead_target[1]
                )
                return target_point_cf, overhead_target_cf
            else:
                # print(" [INFO]: Target point behind the car, removing it from the list")
                values_far[closest_waypoint_far] = np.inf

    def get_speed(
        self, current_target_cf: np.ndarray, overhead_target_cf: np.ndarray
    ) -> float:
        """Return the speed of the car.

        Args:
            current_target_cf (np.ndarray): The current target point in the car frame.
            overhead_target_cf (np.ndarray): The overhead target point in the car frame.
        """

        overhead_angle = math.acos(
            np.dot(current_target_cf, overhead_target_cf)
            / (np.linalg.norm(current_target_cf) * np.linalg.norm(overhead_target_cf))  # type: ignore
        )

        print(overhead_angle)
        return speed_function(overhead_angle)

    def pose_callback(self, pose_msg: Odometry) -> None:
        # In lab ref, 0, 0 is the starting position
        self.car_x = pose_msg.pose.pose.position.x
        self.car_y = pose_msg.pose.pose.position.y
        self.car_theta = pose_msg.pose.pose.orientation.z

        if math.isnan(self.car_x) or math.isnan(self.car_y):
            raise ValueError("x or y is NaN")

        target_point_cf, overhead_target_cf = self.get_target_point()

        self.marker_pub.publish(
            get_marker_msg(target_point_cf[0], target_point_cf[1], 0, color=(1, 0, 0))
        )
        self.marker_pub.publish(
            get_marker_msg(
                overhead_target_cf[0],
                overhead_target_cf[1],
                1,
                color=(0.81, 0.17, 0.93),
            )
        )

        # Test

        self.marker_pub.publish(
            get_marker_msg(5, 5, 2, color=(0, 1, 0), frame_id="map")
        )
        test_point = self.from_global_to_car_frame(5, 5)
        self.marker_pub.publish(
            get_marker_msg(test_point[0], test_point[1], 3, color=(0, 0, 1))
        )

        # Calculate curvature/steering angle
        curvature = 2 * target_point_cf[1] / DIST_L**2

        # publish drive message, don't forget to limit the steering angle between -0.4189 and 0.4189 radians
        speed = self.get_speed(target_point_cf, overhead_target_cf)
        self.drive_pub.publish(get_nav_msg(curvature * SMOOTH_ANGLE, speed))

        if SHOW_WAYPOINTS:
            self.marker_array_pub.publish(WAYPOINTS_MARKER)
        # print(
        #    f"CAR {x:.2f} {y:.2f} | TFC {target_point_cf[0]:.2f} {target_point_cf[1]:.2f} | CURV {curvature:.2f} | SPEED {speed:.2f}"
        # )


def main():
    rospy.init_node("pure_pursuit_node")
    PurePursuit()
    rospy.spin()


if __name__ == "__main__":
    main()
