#!/usr/bin/env python
import math
import rospy

from std_msgs.msg import Bool
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped

SAFE_BREAKING_DURATION = 1  # s


def get_ttc_min(speed: float, scan_msg: LaserScan) -> float:
    ''' Return the smallest TTC '''

    ttc_min = float('+inf')
    for i, dist in enumerate(scan_msg.ranges):
        theta = scan_msg.angle_min + scan_msg.angle_increment*i
        speed_dir = abs(max(speed * math.cos(theta), 0))

        if speed_dir == 0:
            ttc = float('+inf')
        else:
            ttc = dist/speed_dir

        if ttc < ttc_min:
            ttc_min = ttc

    return ttc_min


class Safety:
    """
    The class that handles emergency braking.
    """

    def __init__(self):
        """
        One publisher should publish to the /brake topic with a AckermannDriveStamped brake message.

        One publisher should publish to the /brake_bool topic with a Bool message.

        You should also subscribe to the /scan topic to get the LaserScan messages and
        the /odom topic to get the current speed of the vehicle.

        The subscribers should use the provided odom_callback and scan_callback as callback methods

        NOTE that the x component of the linear velocity in odom is the speed
        """
        self.speed = 0
        # Create ROS subscribers and publishers.
        rospy.Subscriber("scan", LaserScan, self.scan_callback)
        rospy.Subscriber("odom", Odometry, self.odom_callback)

        self.brake_bool_pub = rospy.Publisher(
            '/brake_bool', Bool, queue_size=10)

        self.brake_pub = rospy.Publisher(
            '/brake', AckermannDriveStamped, queue_size=10)

    def odom_callback(self, odom_msg: Odometry):
        ''' Update current speed '''

        # Can be negative !
        self.speed = odom_msg.twist.twist.linear.x

    def scan_callback(self, scan_msg: LaserScan):
        if self.speed == 0:
            return

        ttc_min = get_ttc_min(self.speed, scan_msg)

        if ttc_min < SAFE_BREAKING_DURATION:
            print('[ EMERGENCY BRAKE ] Stopped')

            bool_msg = Bool()
            bool_msg.data = True

            ack_msg = AckermannDriveStamped()
            ack_msg.drive.speed = 0

            self.brake_bool_pub.publish(bool_msg)
            self.brake_pub.publish(ack_msg)
        else:
            print('[ EMERGENCY BRAKE ] RAS')
            bool_msg = Bool()
            bool_msg.data = False

            self.brake_bool_pub.publish(bool_msg)


if __name__ == '__main__':
    rospy.init_node('safety_node')
    Safety()
    rospy.spin()
