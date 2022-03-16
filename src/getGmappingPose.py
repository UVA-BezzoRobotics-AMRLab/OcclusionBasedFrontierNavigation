#! /usr/bin/env python
import sys
import rospy
import tf2_ros
import tf2_geometry_msgs

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped

odom = None


def pose_callback(data):
    global odom
    odom = data


def main():
    global odom
    rospy.init_node("gmapping_pose_publisher", anonymous=True)
    rospy.Subscriber("/odometry/filtered", Odometry, pose_callback)

    pose_pub = rospy.Publisher("gmapping/odometry", Odometry, queue_size=10)

    tf_buffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tf_buffer)

    rate = rospy.Rate(20)
    i = 0

    transform = None
    while not rospy.is_shutdown():
        rate.sleep()

        if odom == None:
            continue

        tmp = PoseStamped()
        tmp.pose = odom.pose.pose
        tmp.header.frame_id = "/odom"
        tmp.header.stamp = rospy.Time.now()

        try:
            if i % 10 == 0:
                transform = tf_buffer.lookup_transform(
                    "map", "odom", rospy.Time(0), rospy.Duration(1.0)
                )
                i = 0

            pose_transformed = tf2_geometry_msgs.do_transform_pose(tmp, transform)
            odom.pose.pose = pose_transformed.pose

        except Exception as e:
            rospy.loginfo(e)
            continue

        pose_pub.publish(odom)


if __name__ == "__main__":
    main()
