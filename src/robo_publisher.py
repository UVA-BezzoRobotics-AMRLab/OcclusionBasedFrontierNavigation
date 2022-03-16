#! /usr/bin/env python
import sys
import math
import rospy
import numpy as np

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from tf.transformations import euler_from_quaternion

jackal_pos = None
length = .5

def getJackalPosOdom(data):
	global jackal_pos
	jackal_pos = data


def main():
	global jackal_pos, length

	rospy.init_node("robo_publisher", anonymous=True)
	rospy.Subscriber("/gmapping/odometry", Odometry, getJackalPosOdom)

	robo_pub = rospy.Publisher("/robo_pose", Marker, queue_size=100)

	rate = rospy.Rate(30)

	prevx = None
	prevy = None
	while not rospy.is_shutdown():
		rate.sleep()

		if jackal_pos == None:
			continue

		x = jackal_pos.pose.pose.position.x
		y = jackal_pos.pose.pose.position.y

		if prevx == None:
			prevx=x
			prevy=y

		if (prevx-x)**2 + (prevy-y)**2 < 1 and (prevx-x)**2 + (prevy-y)**2 > .25:
			robo_pub.publish(msg)
			continue

		msg = Marker()
		msg.header.frame_id="map"
		msg.header.stamp = rospy.Time.now()
		msg.ns = "robo"
		msg.id = 1
		msg.scale.x = length
		msg.scale.y = .4
		msg.scale.z = .3
		msg.action = Marker.ADD
		msg.type = Marker.ARROW
		msg.color.r, msg.color.g, msg.color.b, msg.color.a = (1, 1, 0, 1)

		jackal_orientation = (
			jackal_pos.pose.pose.orientation.x,
			jackal_pos.pose.pose.orientation.y,
			jackal_pos.pose.pose.orientation.z,
			jackal_pos.pose.pose.orientation.w,
		)
		jackal_orient_euler = euler_from_quaternion(jackal_orientation)
		jackal_yaw = jackal_orient_euler[2]

		p1 = [x-length*np.cos(jackal_yaw), y-length*np.sin(jackal_yaw)]
		p2 = [x + length*np.cos(jackal_yaw), y+length*np.sin(jackal_yaw)]

		start = Point()
		start.x = p1[0]
		start.y = p1[1]

		end = Point()
		end.x = p2[0]
		end.y = p2[1]

		msg.points.append(start)
		msg.points.append(end)

		robo_pub.publish(msg)

		prevx = x
		prevy = y

if __name__ == "__main__":
	main()