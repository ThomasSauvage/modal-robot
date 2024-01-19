#!/usr/bin/env python
import sys
import math
import numpy as np
import matplotlib.pyplot as plt

# ROS Imports
import rospy
from sensor_msgs.msg import Image, LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from visualization_msgs.msg import Marker

NBR_POINTS_SCAN = 100
BUBBLE_SIZE = 20

SAFE_DISTANCE = 0.5

SPEED = 2.0
CROP_SCAN = (
    200  # Number of points to remove from each side of the scan, must not be too big
)

GAP_DISTANCE_TRESHOLD = 3.0


def get_nav_msg(angle: float):
    """Return an AckermannDriveStamped Message with the given steering angle"""

    drive_msg = AckermannDriveStamped()
    drive_msg.header.stamp = rospy.Time.now()
    drive_msg.header.frame_id = "laser"
    drive_msg.drive.speed = SPEED
    drive_msg.drive.steering_angle = angle

    return drive_msg


def preprocess_scan(ranges_raw: list) -> np.ndarray:
    """Preprocess the LiDAR scan array. Expert implementation includes:
    1.Setting each value to the mean over some window
    2.Rejecting high values (eg. > 3m)
    """

    smaller_scan = np.zeros((NBR_POINTS_SCAN))
    ranges_raw = ranges_raw[CROP_SCAN:-CROP_SCAN]

    nbr_points_per_mean = int(len(ranges_raw) / NBR_POINTS_SCAN)
    for i in range(NBR_POINTS_SCAN):
        mean = 0
        for j in range(nbr_points_per_mean):
            data = ranges_raw[i * nbr_points_per_mean + j]

            if data == np.inf:
                data = 20

            mean += data

        smaller_scan[i] = mean / nbr_points_per_mean

    return smaller_scan


def get_biggest_gap(ranges: np.ndarray):
    """Return the left & right indicies of the biggest gap in ranges"""

    biggest_gap_left, biggest_gap_right = 0, 0
    biggest_gap = 0
    i = 0
    while i < len(ranges):
        size_current_gap = 0
        while i < len(ranges) and ranges[i] > GAP_DISTANCE_TRESHOLD:
            size_current_gap += 1
            i += 1

        if size_current_gap > biggest_gap:
            biggest_gap = size_current_gap
            biggest_gap_left = i - size_current_gap
            biggest_gap_right = i

        size_current_gap = 0
        i += 1

    if biggest_gap_right == len(ranges):
        biggest_gap_right = len(ranges) - 1

    return biggest_gap_left, biggest_gap_right


def int_to_angle(data: LaserScan, i: int) -> float:
    """Return the angle of the ith element of data.ranges"""

    return data.angle_min + i * data.angle_increment


def replace(string: str, index: int, char: str) -> str:
    """Return a string with the character at index replaced by char"""

    if not 0 <= index < len(string):
        raise ValueError("index out of range")

    return string[:index] + char + string[index + 1 :]


def get_fancy_lidar_string(
    biggest_gap_left: int,
    biggest_gap_right: int,
    biggest_gap_best: int,
    ranges: "list[int]",
):
    """Return a string representing the LiDAR scan with the biggest gap highlighted"""

    lidar_string = "-" * len(ranges)
    # for i in range(bubble_center - BUBBLE_SIZE, bubble_center + BUBBLE_SIZE):
    #    if 0 <= i < len_ranges:
    #        lidar_string = replace(lidar_string, i, "█")
    lidar_string = replace(lidar_string, len(ranges) // 2, "|")
    lidar_string = replace(lidar_string, biggest_gap_left, "#")
    lidar_string = replace(lidar_string, biggest_gap_right, "#")
    lidar_string = replace(lidar_string, biggest_gap_best, "X")

    for i in range(len(ranges)):
        if ranges[i] == 0:
            lidar_string = replace(lidar_string, i, "█")

    return lidar_string[::-1]


class ReactiveFollowGap:
    def __init__(self):
        # Topics & Subscriptions,Publishers
        rospy.Subscriber("/scan", LaserScan, self.scan_callback)
        self.drive_pub = rospy.Publisher("/nav", AckermannDriveStamped, queue_size=10)
        self.marker_pub = rospy.Publisher("/marker", Marker, queue_size=10)

    def scan_callback(self, data: LaserScan):
        """Process each LiDAR scan as per the Follow Gap algorithm & publish an AckermannDriveStamped Message"""

        ranges_no_bubble = preprocess_scan(data.ranges)  # type: ignore

        # Eliminate all points inside 'bubble' (set them to zero)
        ranges = ranges_no_bubble.copy()
        for i in range(len(ranges_no_bubble)):
            if ranges_no_bubble[i] < SAFE_DISTANCE:
                ranges[i - BUBBLE_SIZE : i + BUBBLE_SIZE] = 0

        # Find max length gap
        biggest_gap_left, biggest_gap_right = get_biggest_gap(ranges)

        # Find the best point in the gap
        biggest_gap_best = biggest_gap_left + np.argmax(
            ranges[biggest_gap_left:biggest_gap_right]
        )
        # print(np.argmax(ranges[biggest_gap_left:biggest_gap_right]))

        print(
            get_fancy_lidar_string(
                biggest_gap_left,
                biggest_gap_right,
                biggest_gap_best,
                ranges,
            )
        )

        """
        plt.plot(range(len(ranges)), ranges)
        plt.plot(range(len(ranges)), ranges_no_bubble)
        plt.scatter(biggest_gap_best, ranges[biggest_gap_best], c="r")
        plt.scatter(biggest_gap_right, ranges[biggest_gap_right], c="g")
        plt.scatter(biggest_gap_left, ranges[biggest_gap_left], c="g")
        plt.show()
        """

        # Create drive message
        nbr_points_per_mean = int(len(data.ranges) / NBR_POINTS_SCAN)
        index_biggest_gap_raw = biggest_gap_best * nbr_points_per_mean

        drive_msg = get_nav_msg(int_to_angle(data, index_biggest_gap_raw))

        # print(f"Nbr points per mean : {nbr_points_per_mean}")
        # print(f"closest index : {biggest_gap_best}/{len(ranges)}")
        # print(f"closest index raw : {index_biggest_gap_raw}/{len(data.ranges)}")
        # print(f"steering angle : {int_to_angle(data, index_biggest_gap_raw)}")
        # print("--------------------------")

        self.drive_pub.publish(drive_msg)


def main(args):
    rospy.init_node("FollowGap_node", anonymous=True)
    ReactiveFollowGap()
    rospy.sleep(0.1)
    rospy.spin()


if __name__ == "__main__":
    main(sys.argv)
