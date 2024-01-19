#!/usr/bin/env python
import sys
import math
import numpy as np

# ROS Imports
import rospy
from sensor_msgs.msg import Image, LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive

NBR_POINTS_SCAN = 100
BUBBLE_SIZE = 5
SPEED = 2.0
CROP_SCAN = 400  # Number of points to remove from each side of the scan

GAP_DISTANCE_TRESHOLD = 3.0


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
            mean += ranges_raw[i * nbr_points_per_mean + j]

        smaller_scan[i] = mean / nbr_points_per_mean

        if smaller_scan[i] > 5:
            smaller_scan[i] = 5

    return smaller_scan


def int_to_angle(data: LaserScan, i: int) -> float:
    """Return the angle of the ith element of data.ranges"""

    return data.angle_min + i * data.angle_increment


class ReactiveFollowGap:
    def __init__(self):
        # Topics & Subscriptions,Publishers
        rospy.Subscriber("/scan", LaserScan, self.scan_callback)
        self.drive_pub = rospy.Publisher("/nav", AckermannDriveStamped, queue_size=10)

    def find_max_gap(self, free_space_ranges):
        """Return the start index & end index of the max gap in free_space_ranges"""
        return None

    def find_best_point(self, start_i, end_i, ranges):
        """Start_i & end_i are start and end indicies of max-gap range, respectively
        Return index of best point in ranges
        Naive: Choose the furthest point within ranges and go there
        """
        return None

    def scan_callback(self, data: LaserScan):
        """Process each LiDAR scan as per the Follow Gap algorithm & publish an AckermannDriveStamped Message"""
        ranges = preprocess_scan(data.ranges)  # type: ignore

        # Find closest point to LiDAR
        closest_index = np.argmin(ranges)

        # Eliminate all points inside 'bubble' (set them to zero)
        ranges[closest_index - BUBBLE_SIZE : closest_index + BUBBLE_SIZE] = 0

        # Find max length gap
        index_biggest_gap = 0
        i = 0
        while i < len(ranges):
            size_current_gap = 0
            while i < len(ranges) and ranges[i] > GAP_DISTANCE_TRESHOLD:
                size_current_gap += 1
                i += 1

            if size_current_gap > ranges[index_biggest_gap]:
                index_biggest_gap = i - size_current_gap // 2

            size_current_gap = 0
            i += 1

        print(index_biggest_gap, len(ranges))

        # Find the best point in the gap

        # Publish Drive message
        nbr_points_per_mean = int(len(ranges) / NBR_POINTS_SCAN)

        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = rospy.Time.now()
        drive_msg.header.frame_id = "laser"
        drive_msg.drive.speed = SPEED
        drive_msg.drive.steering_angle = int_to_angle(
            data, int(index_biggest_gap * (len(data.ranges) / NBR_POINTS_SCAN))
        )

        print(int(index_biggest_gap * (len(data.ranges) / NBR_POINTS_SCAN)))
        print(drive_msg.drive.steering_angle)

        self.drive_pub.publish(drive_msg)


def main(args):
    rospy.init_node("FollowGap_node", anonymous=True)
    rfgs = ReactiveFollowGap()
    rospy.sleep(0.1)
    rospy.spin()


if __name__ == "__main__":
    main(sys.argv)
