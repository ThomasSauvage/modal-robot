from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker, MarkerArray


def get_marker_msg(
    x: float,
    y: float,
    id: int,
    color: "tuple[float, float, float]",
    scale: float = 0.2,
) -> Marker:
    """Return a Marker Message with the given position"""

    marker_msg = Marker()
    marker_msg.header.frame_id = "base_link"
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
    marker_msg.scale.x = scale
    marker_msg.scale.y = scale
    marker_msg.scale.z = scale
    marker_msg.color.a = 1.0  # Don't forget to set the alpha!
    marker_msg.color.r = color[0]
    marker_msg.color.g = color[1]
    marker_msg.color.b = color[2]

    return marker_msg


def get_marker_array(points, scale: float = 0.1) -> MarkerArray:
    """Return a Marker Message with the given position"""

    marker_array = MarkerArray()

    for i, point in enumerate(points):  # type: ignore
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
        marker_msg.scale.x = scale
        marker_msg.scale.y = scale
        marker_msg.scale.z = scale
        marker_msg.color.a = 1  # Don't forget to set the alpha!
        marker_msg.color.r = 0.1
        marker_msg.color.g = 0.1
        marker_msg.color.b = 0.1

        # To only publish once and keep it persistent
        # marker_msg.lifetime = rospy.Duration() (doesn't work)

        marker_array.markers.append(marker_msg)  # type: ignore

    return marker_array


def get_nav_msg(angle: float, speed: float) -> AckermannDriveStamped:
    """Return an AckermannDriveStamped Message with the given steering angle"""

    drive_msg = AckermannDriveStamped()
    drive_msg.header.frame_id = "base_link"
    drive_msg.drive.speed = speed
    drive_msg.drive.steering_angle = angle

    return drive_msg
