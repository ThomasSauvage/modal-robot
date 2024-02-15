#!/usr/bin/env python
import numpy as np
import math

import rospy
import tf2_ros
import tf2_geometry_msgs

from utils.speed import ReversedERF
from utils.traj import circle_traj
from utils.iterator import middle_range

from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

DRIVE_TOPIC = "/nav"  # "/vesc/ackermann_cmd_mux/input/navigation"

WAYPOINTS_FILENAME = "map.csv"

SHOW_WAYPOINTS = False
SHOW_ALL_DYN_VIZ_POINTS = True

NBR_WAYPOINTS = 400

LIDAR_NBR_POINTS_TO_KEEP = (
    10  # Keep only 1/LIDAR_NBR_POINTS_TO_KEEP of the points of the lidar
)

DIST_L = 1  # m
SMOOTH_ANGLE = 0.4

CURV_OVERHEAD = 15  # in indexes

# == Dynamic window parameters ==
NBR_TRAJ_DYN_WINDOW = 5
DTHETA_TRAJ_DYN_WINDOW = 0.4  # rad
NBR_POINTS_DYN_WINDOW = 10
DT_DYN_WINDOW = 0.1  # s
SAFE_DISTANCE_DYN_WINDOW = 0.2  # m

speed_function = ReversedERF(
    max_speed=5, min_speed=1, x_for_max_speed=0, x_for_min_speed=1.22
)


def get_waypoints() -> np.ndarray:
    """Import waypoints"""

    with open(WAYPOINTS_FILENAME, "r") as f:
        waypoints_csv = f.readlines()

    waypoints = np.zeros((NBR_WAYPOINTS, 4))

    nbr_waypoints_per_line = len(waypoints_csv) // NBR_WAYPOINTS
    for i in range(NBR_WAYPOINTS):
        line = waypoints_csv[i * nbr_waypoints_per_line]
        waypoints[i] = line.split(",")

    return waypoints


# Structure of WAYPOINTS: x, y, ?, speed
WAYPOINTS = get_waypoints()


def get_marker_msg(
    x: float, y: float, id: int, color: "tuple[float, float, float]"
) -> Marker:
    """Return a Marker Message with the given position"""

    SCALE = 0.2

    marker_msg = Marker()
    marker_msg.header.frame_id = "base_link"
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
        self.ranges: "np.ndarray | None" = None
        self.indexes_to_delete: "list[int] | None" = None
        self.angles: "np.ndarray | None" = None

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
        rospy.Subscriber("/scan", LaserScan, self.scan_callback)

    def from_global_to_car_frame(self, x: float, y: float) -> np.ndarray:
        """Transform a point from the global frame to the car frame.

        Args:
            x (float): The x position of the point in the global frame.
            y (float): The y position of the point in the global frame.

        Returns:
            np.ndarray: The position of the point in the car frame.
        """

        point = tf2_geometry_msgs.PoseStamped()
        point.pose.position.x = x
        point.pose.position.y = y
        point.pose.position.z = 0
        point.header.frame_id = "map"

        transformer = self.tf_buffer.lookup_transform("base_link", "map", rospy.Time())
        point_transformed = tf2_geometry_msgs.do_transform_pose(point, transformer)

        return np.array(
            [point_transformed.pose.position.x, point_transformed.pose.position.y]
        )

    def get_target_point(self, x: float, y: float) -> "tuple[np.ndarray, np.ndarray]":
        """Get the target point for the pure pursuit algorithm.

        Args:
            x (float): The x position of the car.
            y (float): The y position of the car.

        Returns:
            int: The index of the target point in the car frame.
            np.ndarray: The target point.
        """

        # Find the farthest waypoint of dist <= DIST_L and the closest waypoint of dist > DIST_L
        dists_to_car = np.linalg.norm(WAYPOINTS[:, :2] - np.array([x, y]), axis=1)
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
                    (closest_waypoint_far + CURV_OVERHEAD) % NBR_WAYPOINTS, :2
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

        # print(overhead_angle)
        return speed_function(overhead_angle)

    def show_dynamic_window(
        self,
        angle_target: float,
        speed: float,
    ):
        """Show the dynamic window of the car."""

        angular_speed = angle_target / DT_DYN_WINDOW

        for i in range(NBR_POINTS_DYN_WINDOW):
            traj_robot_ref_x, traj_robot_ref_y = circle_traj(
                speed, angular_speed, DT_DYN_WINDOW * i
            )

            self.marker_pub.publish(
                get_marker_msg(
                    traj_robot_ref_x, traj_robot_ref_y, 7000 + i, color=(0, 1, 0)
                )
            )

    def traj_is_valid(self, speed: float, angle_target: float, id: int) -> bool:
        """Return True if the trajectory is valid, False otherwise."""

        angular_speed = angle_target / DT_DYN_WINDOW

        for i in range(NBR_POINTS_DYN_WINDOW):
            traj_x, traj_y = circle_traj(speed, angular_speed, DT_DYN_WINDOW * i)

            obstacles = self.ranges * np.array(
                [np.cos(self.angles), np.sin(self.angles)]  # type: ignore
            )

            traj_points = np.zeros((2, self.angles.shape[0]))
            traj_points[0, :] = traj_x
            traj_points[1, :] = traj_y

            dist_to_closest_obstacle = np.min(
                np.linalg.norm(
                    obstacles - traj_points,
                    axis=0,
                )
            )

            if dist_to_closest_obstacle < SAFE_DISTANCE_DYN_WINDOW:
                if SHOW_ALL_DYN_VIZ_POINTS:
                    self.marker_pub.publish(
                        get_marker_msg(
                            traj_x, traj_y, 9000 + 100 * id + i, color=(0.7, 0.7, 0)
                        )
                    )
                return False

            if SHOW_ALL_DYN_VIZ_POINTS:
                self.marker_pub.publish(
                    get_marker_msg(traj_x, traj_y, 9000 + 100 * id + i, color=(0, 1, 0))
                )

        return True

    def scan_callback(self, scan: LaserScan):

        if not self.ranges is None and not self.indexes_to_delete is None:
            return  # Only define the ranges and indexes to delete once

        self.indexes_to_delete = []
        for i in range(len(scan.ranges)):
            if i % LIDAR_NBR_POINTS_TO_KEEP != 0:  # Keep only 1/n of the points
                self.indexes_to_delete.append(i)

        angles = np.linspace(scan.angle_min, scan.angle_max, len(scan.ranges))
        self.angles = np.delete(angles, self.indexes_to_delete)

        ranges = np.array(scan.ranges)
        self.ranges = np.delete(ranges, self.indexes_to_delete)

    def pose_callback(self, pose_msg: Odometry) -> None:
        if self.ranges is None or self.indexes_to_delete is None:
            print("[INFO] Waiting for the lidar to be ready")
            return

        x = pose_msg.pose.pose.position.x
        y = pose_msg.pose.pose.position.y

        if SHOW_WAYPOINTS:
            self.marker_array_pub.publish(WAYPOINTS_MARKER)

        if math.isnan(x) or math.isnan(y):
            raise ValueError("x or y is NaN")

        target_point_cf, overhead_target_cf = self.get_target_point(x, y)

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

        # Calculate curvature/steering angle
        curvature = 2 * target_point_cf[1] / DIST_L**2

        # publish drive message, don't forget to limit the steering angle between -0.4189 and 0.4189 radians
        speed = self.get_speed(target_point_cf, overhead_target_cf)
        target_angle = curvature * SMOOTH_ANGLE

        for i in middle_range(-NBR_TRAJ_DYN_WINDOW // 2, NBR_POINTS_DYN_WINDOW // 2):

            angle_dyn = target_angle + i * DTHETA_TRAJ_DYN_WINDOW

            if self.traj_is_valid(speed, angle_dyn, id=i):
                self.drive_pub.publish(get_nav_msg(angle_dyn, speed))
                return

        print("[ERROR]: No valid trajectory found, stopping the car")
        self.drive_pub.publish(get_nav_msg(0, 0))

        # print(
        #    f"CAR {x:.2f} {y:.2f} | TFC {target_point_cf[0]:.2f} {target_point_cf[1]:.2f} | CURV {curvature:.2f} | SPEED {speed:.2f}"
        # )


def main():
    rospy.init_node("pure_pursuit_node")
    PurePursuit()
    rospy.spin()


if __name__ == "__main__":
    main()
