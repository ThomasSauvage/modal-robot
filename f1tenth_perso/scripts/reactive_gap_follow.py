#!/usr/bin/env python
import sys
import math
import numpy as np
import matplotlib.pyplot as plt

# ROS Imports
import rospy
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker

print("Version : 2.24")
DEBUG = False

DRIVE_TOPIC = "/nav"  # "/vesc/ackermann_cmd_mux/input/navigation"
NBR_POINTS_SCAN = 20

GAP_DISTANCE_TRESHOLD = 1.5  # m
CROP_SCAN = (
    200  # Number of points to remove from each side of the scan, must not be too big
)

BEST_POINT_METHOD = "middle"  # "farthest" or "middle"

# Angle param
SMOOTH_ANGLE = 1

K_LIVE = 0.9
K_PAST = 0.1

# Speed params
MAX_SPEED = 3  # m/s
MIN_SPEED = 0.8  # m/s
DISTANCE_FOR_MAX_SPEED = 2  # m

# Bubble params
SAFE_DISTANCE = 0.4  # m
BUBBLE_RADIUS = 0.4  # m


def affine(x: float, x1: float, x2: float, y1: float, y2: float) -> float:
    """Return the affine function that goes through (x1, y1) and (x2, y2)"""

    return ((y2 - y1) / (x2 - x1)) * (x - x1) + y1


def erf_remap(x: float) -> float:
    """erf remaped from [0, DISTANCE_FOR_MAX_SPEED] to [0, 1]"""

    return (1 + math.erf(affine(x, 0, DISTANCE_FOR_MAX_SPEED, -2, 2))) / 2


def speed_function(distance: float) -> float:
    """Returns the speed of the robot in function of the distance."""

    if distance >= DISTANCE_FOR_MAX_SPEED:
        return MAX_SPEED
    else:
        return (MAX_SPEED - MIN_SPEED) * erf_remap(distance) + MIN_SPEED
        # Affine
        # return ((MAX_SPEED - MIN_SPEED) / DISTANCE_FOR_MAX_SPEED) * distance + MIN_SPEED


def test_speed_function():
    dist = np.linspace(0, DISTANCE_FOR_MAX_SPEED - 0.01, 100)

    speed = np.vectorize(speed_function)(dist)

    plt.plot(dist, speed)
    plt.show()


# test_speed_function()


def get_nav_msg(angle: float, distance: float):
    """Return an AckermannDriveStamped Message with the given steering angle"""

    drive_msg = AckermannDriveStamped()
    drive_msg.header.stamp = rospy.Time.now()
    drive_msg.header.frame_id = "laser"
    drive_msg.drive.speed = speed_function(distance) * abs(np.cos(angle * 0.5))
    drive_msg.drive.steering_angle = angle * SMOOTH_ANGLE

    return drive_msg


def get_marker_msg(x: float, y: float):
    """Return a Marker Message with the given position"""

    SCALE = 0.5

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
    biggest_gap_size = 0
    i = 0
    while i < len(ranges):
        size_current_gap = 0
        while i < len(ranges) and ranges[i] > GAP_DISTANCE_TRESHOLD:
            size_current_gap += 1
            i += 1

        if size_current_gap > biggest_gap_size:
            biggest_gap_size = size_current_gap
            biggest_gap_left = i - size_current_gap
            biggest_gap_right = i

        size_current_gap = 0
        i += 1

    if biggest_gap_right == len(ranges):
        biggest_gap_right = len(ranges) - 1

    if biggest_gap_size == 0:
        print("\a")
        raise ValueError("No gap found")

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
    ranges: np.ndarray,
    distance: float,
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
            if i == biggest_gap_best:
                lidar_string = replace(lidar_string, i, "$")
            else:
                lidar_string = replace(lidar_string, i, "█")

            if ranges[i] > GAP_DISTANCE_TRESHOLD:
                lidar_string = replace(lidar_string, i, "o")

    return lidar_string[::-1] + f" Speed: {speed_function(distance)}"


class ReactiveFollowGap:
    def __init__(self):
        # Topics & Subscriptions,Publishers
        rospy.Subscriber("/scan", LaserScan, self.scan_callback)
        self.drive_pub = rospy.Publisher(
            DRIVE_TOPIC, AckermannDriveStamped, queue_size=10
        )
        self.marker_pub = rospy.Publisher("/dynamic_viz", Marker, queue_size=10)

        self.latest_biggest_gap_best = None

    def scan_callback(self, data: LaserScan):
        """Process each LiDAR scan as per the Follow Gap algorithm & publish an AckermannDriveStamped Message"""

        nbr_points_per_mean = int((len(data.ranges) - 2 * CROP_SCAN) / NBR_POINTS_SCAN)
        ranges_no_bubble = preprocess_scan(data.ranges)  # type: ignore

        # Eliminate all points inside 'bubble' (set them to zero)
        ranges = ranges_no_bubble.copy()
        for i in range(len(ranges_no_bubble)):
            if ranges_no_bubble[i] < SAFE_DISTANCE:
                if ranges_no_bubble[i] != 0:
                    theta_to_ban = np.arctan(BUBBLE_RADIUS / ranges_no_bubble[i])
                else:
                    theta_to_ban = np.pi / 2

                indexes_to_ban = int(
                    theta_to_ban / (data.angle_increment * nbr_points_per_mean)
                )

                ranges[i - indexes_to_ban : i + indexes_to_ban] = 0

        # Find max length gap
        biggest_gap_left, biggest_gap_right = get_biggest_gap(ranges)

        # Find the best point in the gap
        if BEST_POINT_METHOD == "farthest":
            biggest_gap_best_present = int(
                biggest_gap_left
                + np.argmax(ranges[biggest_gap_left : biggest_gap_right + 1])  # type: ignore
            )
        elif BEST_POINT_METHOD == "middle":
            biggest_gap_best_present = (biggest_gap_left + biggest_gap_right) // 2

        if self.latest_biggest_gap_best is None:
            biggest_gap_best = biggest_gap_best_present
        else:
            biggest_gap_best = round(
                biggest_gap_best_present * K_LIVE
                + self.latest_biggest_gap_best * K_PAST
            )

        self.latest_biggest_gap_best = biggest_gap_best

        print(
            get_fancy_lidar_string(
                biggest_gap_left,
                biggest_gap_right,
                biggest_gap_best,
                ranges,
                ranges[biggest_gap_best],
            )
        )

        # Create drive message

        biggest_gap_best_raw = CROP_SCAN + biggest_gap_best * nbr_points_per_mean
        angle = int_to_angle(data, biggest_gap_best_raw)

        drive_msg = get_nav_msg(angle, ranges[biggest_gap_best])

        if DEBUG:
            plt.plot(range(len(ranges)), ranges, label="Ranges")
            plt.plot(range(len(ranges)), ranges_no_bubble, label="Ranges no bubble")
            plt.scatter(biggest_gap_best, ranges[biggest_gap_best], c="r")
            plt.scatter(biggest_gap_right, ranges[biggest_gap_right], c="g")
            plt.scatter(biggest_gap_left, ranges[biggest_gap_left], c="g")
            plt.legend()
            plt.show()

            resized = CROP_SCAN + np.arange(len(ranges)) * nbr_points_per_mean
            plt.plot(range(len(data.ranges)), data.ranges)
            plt.plot(resized, ranges)
            plt.scatter(biggest_gap_best_raw, data.ranges[biggest_gap_best_raw], c="r")
            plt.show()

        # print(f"Nbr points per mean : {nbr_points_per_mean}")
        # print(f"closest index : {biggest_gap_best}/{len(ranges)}")
        # print(f"closest index raw : {index_biggest_gap_raw}/{len(data.ranges)}")
        # print(f"steering angle : {int_to_angle(data, index_biggest_gap_raw)}")
        # print("--------------------------")

        self.drive_pub.publish(drive_msg)
        x = data.ranges[biggest_gap_best_raw] * math.cos(angle)
        y = data.ranges[biggest_gap_best_raw] * math.sin(angle)
        self.marker_pub.publish(get_marker_msg(x, y))


def main(args):
    rospy.init_node("FollowGap_node", anonymous=True)
    ReactiveFollowGap()
    rospy.sleep(0.1)
    rospy.spin()


if __name__ == "__main__":
    main(sys.argv)
