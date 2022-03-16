#! /usr/bin/env python
import sys
import time
import math
import rospy
import tf2_ros
import numpy as np

from std_msgs.msg import Bool
from collections import defaultdict
from geometry_msgs.msg import Pose, Point
from tf.transformations import euler_from_quaternion
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Odometry


crumbs = None
odom = None

# Calculate the euclidian distance in n-space of the route r traversing cities c, ending at the path start.
path_distance = lambda r, c: np.sum(
    [np.linalg.norm(c[r[p]] - c[r[p - 1]]) for p in range(len(r))]
)
# Reverse the order of all elements from element i to element k in array r.
two_opt_swap = lambda r, i, k: np.concatenate(
    (r[0:i], r[k : -len(r) + i - 1 : -1], r[k + 1 : len(r)])
)

def two_opt(
    cities, improvement_threshold
):  # 2-opt Algorithm adapted from https://en.wikipedia.org/wiki/2-opt
    route = np.arange(
        cities.shape[0]
    )  # Make an array of row numbers corresponding to cities.
    improvement_factor = 1  # Initialize the improvement factor.
    best_distance = path_distance(
        route, cities
    )  # Calculate the distance of the initial path.
    while (
        improvement_factor > improvement_threshold
    ):  # If the route is still improving, keep going!
        distance_to_beat = (
            best_distance  # Record the distance at the beginning of the loop.
        )
        for swap_first in range(
            1, len(route) - 2
        ):  # From each city except the first and last,
            for swap_last in range(
                swap_first + 1, len(route)
            ):  # to each of the cities following,
                new_route = two_opt_swap(
                    route, swap_first, swap_last
                )  # try reversing the order of these cities
                new_distance = path_distance(
                    new_route, cities
                )  # and check the total distance with this modification.
                if (
                    new_distance < best_distance
                ):  # If the path distance is an improvement,
                    route = new_route  # make this the accepted best route
                    best_distance = new_distance  # and update the distance corresponding to this route.
        improvement_factor = (
            1 - best_distance / distance_to_beat
        )  # Calculate how much the route has improved.
    return route  # When the route is no longer improving substantially, stop searching and return the route.


def generatePathMarker(cities):

	ret = Marker()
	ret.header.frame_id = "map"
	ret.header.stamp = rospy.Time.now()
	ret.ns = "final_path"
	ret.id = 1337
	ret.action = Marker.ADD
	ret.type = Marker.LINE_LIST
	ret.scale.x = 0.5
	ret.color.r,ret.color.g,ret.color.b,ret.color.a=(1,.7,0,.7)

	for ind, city in enumerate(cities):

		if ind == cities.shape[0]-1:
			break

		point1 = Point()
		point1.x = city[0]
		point1.y = city[1]
		ret.points.append(point1)

		point2 = Point()
		point2.x = cities[(ind+1)%cities.shape[0]][0]
		point2.y = cities[(ind+1)%cities.shape[0]][1]
		ret.points.append(point2)

	return ret


def getCrumbs(data):
	global crumbs

	if crumbs == None:
		crumbs = [m for m in data.markers if len(m.points) > 0]

		for crumb in crumbs:

			crumb.color.r, crumb.color.g, crumb.color.b, crumb.color.a = (1,0,0,1)


def getOdom(data):
	global odom

	odom = data.pose.pose


def main():
	global crumbs, odom

	rospy.init_node("final_crumbs")

	rospy.Subscriber("/gmapping/odometry", Odometry, getOdom)
	# rospy.Subscriber("/crumbs", MarkerArray, getCrumbs)
	rospy.Subscriber("/breadcrumbs", MarkerArray, getCrumbs)
	# rospy.Subscriber("/crumbs_9", MarkerArray, getCrumbs)

	crumb_pub = rospy.Publisher("/remaining_crumbs", MarkerArray, queue_size=10)
	path_pub = rospy.Publisher("/remaining_path", Marker, queue_size=10)

	rate = rospy.Rate(10)

	threshold = .4
	path_marker = None
	path = None
	cities = None

	delete_msg = Marker()
	delete_msg.header.frame_id = "map"
	delete_msg.header.stamp = rospy.Time.now()
	delete_msg.ns = "visibilities"
	delete_msg.action = Marker.DELETEALL

	count = 0
	while not rospy.is_shutdown():
		rate.sleep()

		if odom == None or crumbs == None:
			continue

		if path == None:

			cities = [[odom.position.x, odom.position.y]]
			for crumb in crumbs:
				cx = (crumb.points[0].x + crumb.points[1].x)/2
				cy = (crumb.points[0].y + crumb.points[1].y)/2

				cities.append([cx,cy])

			cities = np.asarray(cities)
			path=[i for i in range(cities.shape[0])]
			# path = two_opt(cities, .001)
			cities = np.delete(cities, 0,0)
			# print(cities)
			path = [p-1 for p in path if p > 0]

			# print(cities[path])
			path_marker = generatePathMarker(cities[path])

			# print(cities)
		
		# sys.exit()
		for ind, crumb in enumerate(crumbs):
			if crumb == None:
				continue

			cx = (crumb.points[0].x + crumb.points[1].x)/2
			cy = (crumb.points[0].y + crumb.points[1].y)/2

			jx = odom.position.x
			jy = odom.position.y

			if (cx-jx)**2 + (cy-jy)**2 < threshold:
				crumbs[ind] = None


				path_marker = generatePathMarker(cities[path])
				# tmp = list(path)
				path=[p for p in path if p != ind]

				# if ind == 6 and count == 0:
				# 	count = 1
				# 	path = tmp
				# 	crumbs[ind] = None

				# if ind == 0 and count == 1:
				# 	count = 0
				# 	path = [p for p in path if p != 6]
				# 	tmp = [p for p in tmp if p != 6]
				# 	path_marker = generatePathMarker(cities[tmp])

				if len(path) == 0:
					path_marker = generatePathMarker(cities[path])

				print(ind)

		msg = MarkerArray()
		msg.markers = [delete_msg]
		for ind, crumb in enumerate(crumbs):
			if crumb == None:
				continue

			msg.markers.append(crumb)

		crumb_pub.publish(msg)
		path_pub.publish(path_marker)

if __name__ == "__main__":
	main()
