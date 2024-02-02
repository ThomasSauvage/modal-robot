#!/usr/bin/env python
import numpy as np

import rospy
import tf2_ros
import tf2_geometry_msgs

from pprint import pprint

from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Odometry


DRIVE_TOPIC = "/nav"  # "/vesc/ackermann_cmd_mux/input/navigation"

DIST_L = 2  # m


def get_marker_msg(x: float, y: float):
    """Return a Marker Message with the given position"""

    SCALE = 0.2

    marker_msg = Marker()
    marker_msg.header.frame_id = "base_link"
    marker_msg.header.stamp = rospy.Time.now()
    marker_msg.ns = "marker1"
    marker_msg.id = 0
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
    marker_msg.color.r = 1.0
    marker_msg.color.g = 0.0
    marker_msg.color.b = 0.0

    return marker_msg


def get_marker_array_waypoint():
    """Return a Marker Message with the given position"""

    SCALE = 0.1

    marker_array = MarkerArray()

    for i, point in enumerate(WAYPOINTS):
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
        marker_msg.color.a = 0.5  # Don't forget to set the alpha!
        marker_msg.color.r = 1
        marker_msg.color.g = 1
        marker_msg.color.b = 1

        # To only publish once and keep it persistent
        # marker_msg.lifetime = rospy.Duration() (doesn't work)

        marker_array.markers.append(marker_msg)  # type: ignore

    return marker_array


def get_waypoints():
    """Import waypoints"""

    with open("/home/thomas/rcws/logs/wp-2024-01-26-16-05-00.csv", "r") as f:
        waypoints_csv = f.readlines()

    waypoints = np.zeros((len(waypoints_csv), 4))

    for i, line in enumerate(waypoints_csv):
        waypoints[i] = line.split(",")

    return waypoints


def get_target_point(x: float, y: float):
    """Get the target point for the pure pursuit algorithm.

    Args:
        x (float): The x position of the car.
        y (float): The y position of the car.

    Returns:
        np.ndarray: The target point.
    """

    # Find the farthest waypoint of dist <= DIST_L and the closest waypoint of dist > DIST_L
    dists_to_car = np.linalg.norm(WAYPOINTS[:, :2] - np.array([x, y]), axis=1)

    values_near = dists_to_car.copy()
    values_far = dists_to_car.copy()

    values_near[values_near > DIST_L] = -np.inf
    values_far[values_far <= DIST_L] = np.inf

    closest_waypoint_far = np.argmin(
        values_far
    )  # The closest waypoint that is outside the circle of radius L

    return WAYPOINTS[closest_waypoint_far, :2]

    farthest_waypoint_near = np.argmax(
        values_near
    )  # The farthest waypoint that is in the circle of radius L

    # Best point is the mean of the two
    target_point = (
        WAYPOINTS[farthest_waypoint_near, :2] + WAYPOINTS[closest_waypoint_far, :2]
    ) / 2

    return target_point


# Structure of WAYPOINTS: x, y, ?, speed
WAYPOINTS = get_waypoints()
WAYPOINTS_MARKER = get_marker_array_waypoint()


class PurePursuit(object):
    """
    The class that handles pure pursuit.
    """

    def __init__(self):
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(buffer=self.tf_buffer)

        self.starting_pos_x = None
        self.starting_pos_y = None

        self.drive_pub = rospy.Publisher(
            DRIVE_TOPIC, AckermannDriveStamped, queue_size=10
        )
        self.marker_pub = rospy.Publisher("/dynamic_viz", Marker, queue_size=10)
        self.marker_array_pub = rospy.Publisher(
            "/dynamic_viz_array", MarkerArray, queue_size=10
        )

        self.marker_array_pub.publish(get_marker_array_waypoint())

        rospy.Subscriber("/odom", Odometry, self.pose_callback)

    def from_global_to_car_frame(self, x: float, y: float):
        """Transform a point from the global frame to the car frame.

        Args:
            x (float): The x position of the point in the global frame.
            y (float): The y position of the point in the global frame.

        Returns:
            np.ndarray: The position of the point in the car frame.
        """

        if self.starting_pos_x is None or self.starting_pos_y is None:
            raise ValueError("Starting position not set")

        point = tf2_geometry_msgs.PoseStamped()
        point.pose.position.x = x + self.starting_pos_x
        point.pose.position.y = y + self.starting_pos_y
        point.pose.position.z = 0
        point.header.frame_id = "map"

        transformer = self.tf_buffer.lookup_transform("base_link", "map", rospy.Time())
        point_transformed = tf2_geometry_msgs.do_transform_pose(point, transformer)

        return point_transformed.pose.position.x, point_transformed.pose.position.y

    def pose_callback(self, pose_msg: Odometry):
        if self.starting_pos_x is None or self.starting_pos_y is None:
            self.starting_pos_x = pose_msg.pose.pose.position.x
            self.starting_pos_y = pose_msg.pose.pose.position.y

        # In lab ref, 0, 0 is the starting position
        x = pose_msg.pose.pose.position.x - self.starting_pos_x
        y = pose_msg.pose.pose.position.y - self.starting_pos_y

        target_point = get_target_point(x, y)

        target_point_car_frame = self.from_global_to_car_frame(
            target_point[0], target_point[1]
        )

        self.marker_pub.publish(
            get_marker_msg(target_point_car_frame[0], target_point_car_frame[1])
        )

        # self.marker_array_pub.publish(WAYPOINTS_MARKER)

        # TODO: transform goal point to vehicle frame of reference

        # TODO: calculate curvature/steering angle

        # TODO: publish drive message, don't forget to limit the steering angle between -0.4189 and 0.4189 radians

        print(
            f"CAR {x:.2f} {y:.2f} | T {target_point[0]:.2f} {target_point[1]:.2f} | T_F {target_point_car_frame[0]:.2f} {target_point_car_frame[1]:.2f}"
        )


def main():
    rospy.init_node("pure_pursuit_node")
    PurePursuit()
    rospy.spin()


if __name__ == "__main__":
    main()
