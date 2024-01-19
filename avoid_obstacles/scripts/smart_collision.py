#!/usr/bin/env python
from matplotlib import colors
import rospy
import matplotlib.pyplot as plt
import numpy as np

# from std_msgs.msg import String
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry


'''
Avoid collisions with a smarter algorithm.

To start this program, it needs:
roscore
roslaunch turtlebot3_gazebo turtlebot3_world.launch
roslaunch turtlebot3_slam turtlebot3_slam.launch
'''

OBSTACLE_DETECTION_DURATION = 4
NBR_POINTS_TRAJECTORY = 10
SAFE_DISTANCE = 0.5
ANGULAR_VELOCITIES = np.linspace(0, 0.3, 10)
SPEED = 0.2


def rotation_matrix(angle: float):
    return np.array([[np.cos(angle), -np.sin(angle)],
                     [np.sin(angle), np.cos(angle)]])


def zero_twist():
    vel_msg = Twist()

    vel_msg.linear.x = 0
    vel_msg.linear.y = 0
    vel_msg.linear.z = 0
    vel_msg.angular.x = 0
    vel_msg.angular.y = 0
    vel_msg.angular.z = 0

    return vel_msg


def circle_traj(speed: float, angular_speed: float, t: float) -> 'tuple[float, float]':
    ''' Return the position after a duration t when the robot has:
         - A normal speed speed
         - An angular speed angular_speed
    '''

    if angular_speed == 0:
        return speed*t, 0.

    if angular_speed > 0:
        dir = 1
    else:
        dir = -1

    r = abs(speed/angular_speed)  # Radius of the circle
    theta = abs(angular_speed)*t

    if theta > 2*np.pi:
        raise ValueError('Theta is superior to 2pi')

    return r*np.sin(theta), dir*(r - r*np.cos(theta))


def test_circle_traj():
    for angular_speed in [-0.01, 0.01]:
        for speed in [0.01, 0.02, 0.03]:
            traj_x = []
            traj_y = []
            for t in range(100):
                x, y = circle_traj(speed, angular_speed, t)
                traj_x.append(x)
                traj_y.append(y)

            plt.plot(traj_x, traj_y, label=f'v={speed}, w={angular_speed}')

    plt.legend()
    plt.show()


class Map:
    def __init__(self, raw_data: OccupancyGrid, pos_x: float, pos_y: float, orientation: float):
        self.raw: OccupancyGrid = raw_data

        self.width = self.raw.info.width
        self.height = self.raw.info.height
        self.resolution = self.raw.info.resolution

        self.origin_x = self.raw.info.origin.position.x
        self.origin_y = self.raw.info.origin.position.y

        self.pos_x = pos_x
        self.pos_y = pos_y

        self.orientation = orientation

    def __getitem__(self, coords: 'tuple[int, int]'):
        i, j = coords
        return self.raw.data[i*self.width + j]

    def get(self, coords: 'tuple[float, float]'):
        ''' Like getitem, but uses coordinates in a
        referential centered on the robot, using real coords.
        It is also rotated by the angle of the robot (using it's speed)
         '''
        x, y = coords

        i, j = self.robot_ref_to_map(x, y)

        return self[int(i), int(j)]

    def to_np_array(self):
        array = np.zeros((self.width, self.height))

        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                array[i, j] = self[i, j]

        return array

    def show(self):
        cmap = colors.ListedColormap(['white', 'red', 'blue'])
        bounds = [-1.5, -0.5, 90, 110]
        norm = colors.BoundaryNorm(bounds, cmap.N)

        img = plt.imshow(self.to_np_array(), interpolation='nearest', origin='lower',
                         cmap=cmap, norm=norm)

        plt.colorbar(img, cmap=cmap, norm=norm, boundaries=bounds)

        i_robot, j_robot = self.robot_ref_to_map(0, 0)

        plt.scatter([i_robot], [j_robot], label='Robot position')
        plt.legend()
        plt.show()

    def robot_ref_to_map(self, x: float, y: float) -> 'tuple[int, int]':
        ''' Transforms coordinates in a referential centered on the 
        robot, to integer coords '''

        vect = np.array([x, y])
        rot = rotation_matrix(self.orientation)

        vect_rotated = np.matmul(rot, vect)

        i = (vect_rotated[0] + self.pos_x -
             self.origin_x) / self.resolution
        j = (vect_rotated[1] + self.pos_y -
             self.origin_y) / self.resolution

        if not 0 <= i < self.width:
            raise ValueError('Row out of map')

        if not 0 <= j < self.height:
            raise ValueError('Column out of map')

        return int(i), int(j)

    def get_occupied_points(self):
        occupied_points_list = []
        for i in range(self.width):
            for j in range(self.height):
                if self[i, j] > 95:
                    occupied_points_list.append((i, j))

        return np.array(occupied_points_list)


class SmartCollision:
    def __init__(self):
        rospy.init_node('smart_collision', anonymous=True)
        rospy.Subscriber("/map", OccupancyGrid, self.map_callback)
        rospy.Subscriber('/odom', Odometry, self.set_pos_callback)

        # Position of the robot
        self.pos_x: float = 0.
        self.pos_y: float = 0.

        # Orientation of the robot, in radian, between -pi and pi (0 is the x axis, in front of the robot)
        # ! Negative is to the left, positive is to the right
        self.orientation: float = 0.

        self.speed_x: float = 0.
        self.speed_y: float = 0.

        self.angular_speed: float = 0.

        self.velocity_publisher = rospy.Publisher(
            '/cmd_vel', Twist, queue_size=10)

    def start(self):
        rospy.spin()

    def set_pos_callback(self, data: Odometry):
        self.pos_x = data.pose.pose.position.x
        self.pos_y = data.pose.pose.position.y

        self.orientation = data.pose.pose.orientation.z*np.pi

        self.speed_x = data.twist.twist.linear.x
        self.speed_y = data.twist.twist.linear.y

        # It is the normal vector around the rotation
        self.angular_speed = data.twist.twist.angular.z

    def show_trajs(self, map: Map):
        plt.imshow(map.to_np_array(), vmin=95, vmax=110)

        for sign in [-1, 1]:
            for angular_velocity in ANGULAR_VELOCITIES:
                traj_is = []
                traj_js = []
                for i in range(NBR_POINTS_TRAJECTORY):
                    traj_robot_ref_x, traj_robot_ref_y = circle_traj(
                        SPEED, sign*angular_velocity, OBSTACLE_DETECTION_DURATION*i/NBR_POINTS_TRAJECTORY)

                    traj_i, traj_j = map.robot_ref_to_map(
                        traj_robot_ref_x, traj_robot_ref_y)

                    traj_is.append(traj_i)
                    traj_js.append(traj_j)

                plt.scatter(traj_is, traj_js,
                            label=f'w={sign*angular_velocity}')

        i_robot, j_robot = map.robot_ref_to_map(0, 0)
        plt.scatter([i_robot], [j_robot], label='Robot position')
        plt.legend()
        plt.show()

    def map_callback(self, data: OccupancyGrid):
        map = Map(data, self.pos_x, self.pos_y, self.orientation)

        # self.show_trajs(map)
        # return

        # List of points on the map
        occupied_points = map.get_occupied_points()

        # Try first the straight trajectory, if it work, do it.
        # Then try the slightly to the right... Then to the left...
        # Then more to the right... more to the left etc...
        for angular_velocity in ANGULAR_VELOCITIES:
            for angular_velocity_sign in [-1, 1]:
                trajectory_works = True
                for i in range(NBR_POINTS_TRAJECTORY):
                    traj_robot_ref_x, traj_robot_ref_y = circle_traj(
                        SPEED, angular_velocity_sign*angular_velocity, OBSTACLE_DETECTION_DURATION*i/NBR_POINTS_TRAJECTORY)

                    traj_i, traj_j = map.robot_ref_to_map(
                        traj_robot_ref_x, traj_robot_ref_y)

                    dists_squared = (occupied_points[:, 0] - traj_i)**2 + \
                        (occupied_points[:, 1] - traj_j)**2

                    min_dist_squared = np.min(dists_squared)

                    if min_dist_squared < SAFE_DISTANCE**2:
                        print(
                            f'[TOO CLOSE] For trajectory(speed={SPEED}, angular_velocity={angular_velocity_sign*angular_velocity})')
                        trajectory_works = False
                        break

                if trajectory_works:
                    print(
                        f' -> [OK] For trajectory(speed={SPEED}, angular_velocity={angular_velocity})')

                    vel_msg = zero_twist()
                    vel_msg.linear.x = SPEED
                    vel_msg.angular.z = angular_velocity_sign*angular_velocity
                    self.velocity_publisher.publish(vel_msg)

                    return

        print(' -> [NO TRAJECTORY] All trajectories collided')


if __name__ == '__main__':
    runner = SmartCollision()
    runner.start()
