#!/usr/bin/env python
import sys
import math
import numpy as np

# ROS Imports
import rospy
from sensor_msgs.msg import Image, LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive

# PID CONTROL PARAMS
K_P = 0.9
K_D = 0.01
K_I = 0.0001


# WALL FOLLOW PARAMS
DESIRED_DISTANCE_WALL = 0.55
VELOCITY = 2.00  # meters per second
PROPORTION_SPEED_TURN = 1  # 0 < ... <= 1
TURN_THRESHOLD = 10 * np.pi / 180

THETA = 60 * math.pi / 180
L = 0.2


def get_dist_dir(data: LaserScan, angle: float) -> float:
    """Return the distance to the closest obje
    in the direction angle.

    - Angle unit: Rad

    - 0 is in front (?)
    - pi is behind (?)

    # data: single message from topic /scan
    # angle: between -45 to 225 degrees, where 0 degrees is directly to the right
    # Outputs length in meters to object with angle in lidar scan field of view
    #make sure to take care of nans etc.
    """

    if not data.angle_min <= angle <= data.angle_max:
        raise ValueError("Angle asked out of LIDAR range")

    i = int((angle - data.angle_min) / data.angle_increment)

    return data.ranges[i]


class WallFollow:
    """Implement Wall Following on the car"""

    def __init__(self):
        # Topics & Subs, Pubs
        rospy.Subscriber("/scan", LaserScan, self.scan_callback)
        self.drive_pub = rospy.Publisher("/nav", AckermannDriveStamped, queue_size=10)

        self.err_integral = 0
        self.prev_error = 0
        self.error = 0

    def pid_control(self):
        """Implement the PID controller"""

        u_angle = (
            K_P * self.error
            + K_I * self.err_integral
            + K_D * (self.error - self.prev_error)
        )

        if not -np.pi <= u_angle <= np.pi:
            print("[ WALL FOLLOW ] Warning: u_angle out of [-pi, pi]")

        if abs(u_angle) > TURN_THRESHOLD:
            speed = VELOCITY * PROPORTION_SPEED_TURN
        else:
            speed = VELOCITY

        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = rospy.Time.now()
        drive_msg.header.frame_id = "laser"
        drive_msg.drive.steering_angle = u_angle
        drive_msg.drive.speed = speed
        self.drive_pub.publish(drive_msg)

    def scan_callback(self, data: LaserScan):
        """Callback for the LIDAR"""

        b = get_dist_dir(data, np.pi / 2)
        a = get_dist_dir(data, np.pi / 2 - THETA)

        alpha = math.atan((a * math.cos(THETA) - b) / (a * math.sin(THETA)))

        D_t_pp = b * math.cos(alpha) + L * math.sin(alpha)

        # Update errors
        self.prev_error = self.error
        self.error = D_t_pp - DESIRED_DISTANCE_WALL
        self.err_integral += self.error

        self.pid_control()


def main(args):
    rospy.init_node("WallFollow_node", anonymous=True)
    wf = WallFollow()
    rospy.sleep(0.1)
    rospy.spin()


if __name__ == "__main__":
    main(sys.argv)
