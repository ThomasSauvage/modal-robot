#!/usr/bin/env python
import numpy as np

import rospy
from pprint import pprint
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import LaserScan

from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker
from nav_msgs.msg import Odometry

DRIVE_TOPIC = "/nav"  # "/vesc/ackermann_cmd_mux/input/navigation"

DIST_L = 1  # m


# Import waypoints
def get_waypoints():
    with open("/home/thomas/rcws/logs/wp-2024-01-26-16-05-00.csv", "r") as f:
        waypoints_csv = f.readlines()

    # Structure: x, y, ?, speed
    waypoints = np.zeros((len(waypoints_csv), 4))

    for i, line in enumerate(waypoints_csv):
        waypoints[i] = line.split(",")

    return waypoints


waypoints = get_waypoints()


class PurePursuit(object):
    """
    The class that handles pure pursuit.
    """

    def __init__(self):
        rospy.Subscriber("/odom", Odometry, self.pose_callback)
        self.drive_pub = rospy.Publisher(
            DRIVE_TOPIC, AckermannDriveStamped, queue_size=10
        )
        self.marker_pub = rospy.Publisher("/dynamic_viz", Marker, queue_size=10)

        self.starting_pos_x = None
        self.starting_pos_y = None

    def pose_callback(self, pose_msg: Odometry):
        if self.starting_pos_x is None or self.starting_pos_y is None:
            self.starting_pos_x = pose_msg.pose.pose.position.x
            self.starting_pos_y = pose_msg.pose.pose.position.y

        # In lab ref, 0, 0 is the starting position
        x = pose_msg.pose.pose.position.x - self.starting_pos_x
        y = pose_msg.pose.pose.position.y - self.starting_pos_y

        # Find the current waypoint to track using methods mentioned in lecture

        # TODO: transform goal point to vehicle frame of reference

        # TODO: calculate curvature/steering angle

        # TODO: publish drive message, don't forget to limit the steering angle between -0.4189 and 0.4189 radians


def main():
    rospy.init_node("pure_pursuit_node")
    PurePursuit()
    rospy.spin()


if __name__ == "__main__":
    main()
