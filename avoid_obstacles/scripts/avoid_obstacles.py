#!/usr/bin/env python

import rospy


#from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

SAFE_DISTANCE = 0.25
    
def get_dist_dir(data, angle: float) -> float:
    ''' Return the distance to the closest object
        in the direction angle.

        - Angle unit: Rad

        - 0 is in front
        - pi is behind
    '''

    if not data.angle_min <= angle <= data.angle_max:
        raise ValueError('Angle asked out of LIDAR range')
    

    i =  int((angle - data.angle_min) / data.angle_increment)

    return data.ranges[i]

class AvoidObstacles:   
    def __init__(self):
        rospy.init_node('listener', anonymous=True)
        rospy.Subscriber("scan", LaserScan, self.callback)

        self.velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)   

    
    def start(self):
        rospy.spin()
    
    def callback(self, data):
        dist_front = get_dist_dir(data, 0)

        print(dist_front)

        if dist_front < SAFE_DISTANCE:
            vel_msg = Twist()

            vel_msg.linear.x = 0
            vel_msg.linear.y = 0
            vel_msg.linear.z = 0
            vel_msg.angular.x = 0
            vel_msg.angular.y = 0
            vel_msg.angular.z = 0


            self.velocity_publisher.publish(vel_msg)
            print('[CONTACT]')
        else:
            vel_msg = Twist()

            vel_msg.linear.x = 0.2
            vel_msg.linear.y = 0
            vel_msg.linear.z = 0
            vel_msg.angular.x = 0
            vel_msg.angular.y = 0
            vel_msg.angular.z = 0  

            self.velocity_publisher.publish(vel_msg)  


if __name__ == '__main__':
    runner = AvoidObstacles()
    runner.start()
