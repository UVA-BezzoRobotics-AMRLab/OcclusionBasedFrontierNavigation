#! /usr/bin/env python
import sys
import math
import rospy
import numpy as np

from copy import deepcopy
from geometry_msgs.msg import Pose
from shapely.ops import unary_union
from shapely.geometry import Polygon
from sensor_msgs.msg import LaserScan
from shapely.geometry import Point as shaPoint
from geometry_msgs.msg import Point as geoPoint
from nav_msgs.msg import OccupancyGrid, Odometry
from tf.transformations import euler_from_quaternion
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PolygonStamped as geoPoly
from std_msgs.msg import Bool, Float32MultiArray, MultiArrayDimension

lidar = None
jackal_pos = None

MAX_RANGE = None

JACKAL_LENGTH = 0.508
THRESHOLD = 3 * (2 * JACKAL_LENGTH) ** 2
STEP = 12
NUM_BREADCRUMBS = None
Hz = .5

minA = -3 * math.pi / 4
maxA = 3 * math.pi / 4
SAMPLES = 720
angles = None  # [minA + ind * (maxA - minA) / (SAMPLES - 1) for ind in range(SAMPLES)]

poses = []
map_ = None
map_meta = None
global_poly = None
BUFFER_LENGTH = 150
CAP_RANGE = 5

initial_pos = None
initial_ori = None
initial_poly = None

experiment = None
switch = True
start = False

def getJackalPos(data):
    global jackal_pos
    jackal_pos = data


def getJackalPosOdom(data):
    global jackal_pos, initial_pos, initial_ori

    jackal_pos = data.pose.pose

    if initial_pos == None:

        initial_pos = jackal_pos.position
        initial_ori = jackal_pos.orientation


def getLidarData(data):
    global lidar, angles, SAMPLES, MAX_RANGE
    lidar = data.ranges
    SAMPLES = len(lidar)
    MAX_RANGE = data.range_max

    if angles == None:
        angles = [data.angle_min + i * data.angle_increment for i in range(SAMPLES)]


def update_poses():
    global global_poly, initial_poly, experiment

    points = generate_polygon()

    vis_polygon = Polygon(points).simplify(0.05, preserve_topology=False)

    if initial_poly == None:
        initial_poly = vis_polygon


    if not experiment:
        origin = map_meta.origin
        resolution = map_meta.resolution
        w, h = map_meta.width, map_meta.height

        x = jackal_pos.position.x
        y = jackal_pos.position.y

        wi, hi = (x - JACKAL_LENGTH / 2 - origin.position.x) / resolution, (
            y - JACKAL_LENGTH / 2 - origin.position.y
        ) / resolution
        wf, hf = (x + JACKAL_LENGTH / 2 - origin.position.x) / resolution, (
            y + JACKAL_LENGTH / 2 - origin.position.y
        ) / resolution

        # Check if width / height are NaN or inf
        if math.fabs(wi) == float("inf") or wi != wi:
            wi = w
        if math.fabs(hi) == float("inf") or hi != hi:
            hi = h

        if math.fabs(wf) == float("inf") or wf != wf:
            wf = w
        if math.fabs(hf) == float("inf") or hf != hf:
            hf = h

        wi, hi = int(min(wi, w)), int(min(h, hi))
        wf, hf = int(min(w, wf)), int(min(hf, h))

        square = map_[wi:wf+hf*w]
        if np.sum(square[square > 0]) > 0:
            return False

    for poly, pos, ori in poses:

        if (pos.x - jackal_pos.position.x) ** 2 + (
            pos.y - jackal_pos.position.y
        ) ** 2 < 1:
            return False

        # Check if new pose is too similar to old pose in terms of visibility
        int_area = vis_polygon.intersection(poly).area
        if int_area / vis_polygon.area > 0.95 or int_area / poly.area > 0.95:
            return False

    poses.append((vis_polygon, jackal_pos.position, jackal_pos.orientation))

    if len(poses) == BUFFER_LENGTH:
        del poses[0]

    # if len(poses) > 1:
    global_poly = unary_union([poly[0] for poly in poses])
    # else:
    # 	global_poly = vis_polygon

    return True


def updateMap(data):
    global map_, map_meta
    map_meta = data.info
    map_ = data.data


def generate_polygon():

    global CAP_RANGE
    jackal_orientation = (
        jackal_pos.orientation.x,
        jackal_pos.orientation.y,
        jackal_pos.orientation.z,
        jackal_pos.orientation.w,
    )
    jackal_orient_euler = euler_from_quaternion(jackal_orientation)
    jackal_yaw = jackal_orient_euler[2]

    x = jackal_pos.position.x
    y = jackal_pos.position.y

    points = []
    for i in range(0, len(lidar), STEP):
        d = lidar[i]
        if d == float("inf"):
            continue
            # d = MAX_RANGE
        elif d != d:
            continue

        d = min(CAP_RANGE,d)
        px = d * np.cos(angles[i] + jackal_yaw) + x
        py = d * np.sin(angles[i] + jackal_yaw) + y

        points.append((px, py))

    points.append((x, y))

    return Polygon(points)


def generate_outline(polygon_points):
    msg = Marker()
    msg.header.frame_id = "map"
    msg.header.stamp = rospy.Time.now()
    msg.ns = "border"
    msg.id = 1
    msg.scale.x = 0.1
    msg.scale.y = 0.1
    msg.color.r, msg.color.g, msg.color.b, msg.color.a = (1, 0, 0, 1)
    msg.action = Marker.ADD
    msg.type = Marker.POINTS
    msg.points = [geoPoint(x, y, 1) for (x, y) in polygon_points]

    return msg


def generate_message(pos, ori, _id, color=(1,0,0,1)):
    msg = Marker()
    msg.header.frame_id = "map"
    msg.header.stamp = rospy.Time.now()
    msg.ns = "visibilities"
    msg.id = _id
    msg.scale.x = 0.2
    msg.scale.y = 0.5
    msg.scale.z = 0.2
    msg.color.r, msg.color.g, msg.color.b, msg.color.a = color
    msg.action = Marker.ADD
    msg.type = Marker.ARROW
    msg.points = []

    start = geoPoint()
    end = geoPoint()

    start.x = pos.x
    start.y = pos.y
    start.z = 1

    # Compute jackal heading
    jackal_orientation = (
        ori.x,
        ori.y,
        ori.z,
        ori.w,
    )
    jackal_orient_euler = euler_from_quaternion(jackal_orientation)
    jackal_yaw = jackal_orient_euler[2]

    end.x = pos.x + JACKAL_LENGTH * np.cos(jackal_yaw)
    end.y = pos.y + JACKAL_LENGTH * np.sin(jackal_yaw)
    end.z = 1

    msg.points.append(start)
    msg.points.append(end)

    return msg


def newUpdate(marker_arr, areas):

    global_area = global_poly.area

    if global_area == 0:
        return []

    crumbs = [(initial_poly, initial_pos, initial_ori)]
    # crumbs = []

    max_area = 0
    # prev_area = 0
    search_space = list(poses)

    while max_area / global_area < .95 and len(crumbs) < len(poses):
        # print("crumbs: {}\tposes: {}".format(len(crumbs), len(poses)))
        max_area = 0


        if len(crumbs) > 0:
            max_area = unary_union([item[0] for item in crumbs]).area

        prev_crum_len = len(crumbs)

        champ_ind = -1
        for ind, (vis_poly, pos, ori) in enumerate(search_space):
            
            if vis_poly == None or vis_poly.area != vis_poly.area:
                continue

            too_close = False
            for i, (vp, p, o) in enumerate(crumbs):

                if (p.x - pos.x)**2 + (p.y - pos.y)**2 < THRESHOLD:
                    too_close = True
                    break

            if too_close:
                continue

            crumbs.append((vis_poly, pos, ori))
            new_poly = unary_union([item[0] for item in crumbs])

           
            if new_poly.area >= max_area:
                champ_ind = ind
                max_area = new_poly.area
                
            
            del crumbs[-1]

        # if prev_area / max_area > .95:
        #     break

        if champ_ind != -1:
            crumbs.append(deepcopy(search_space[champ_ind]))
            search_space[champ_ind] = (None, None, None)

        if len(crumbs) == prev_crum_len:
            break

        # Move any "champion" crumb to the top of the poses stack so it isn't deleted 
        # anytime soon.
        poses.append(poses.pop(champ_ind))

    print("done: covered area is {}\tglobal area is {}".format(max_area, global_area))
    print(np.array(crumbs)[:,[1]])
    return crumbs


# Calculate the euclidian distance in n-space of the route r traversing cities c, ending at the path start.
path_distance = lambda r, c: np.sum(
    [np.linalg.norm(c[r[p]] - c[r[p - 1]]) for p in range(len(r))]
)
# Reverse the order of all elements from element i to element k in array r.
two_opt_swap = lambda r, i, k: np.concatenate(
    (r[0:i], r[k : -len(r) + i - 1 : -1], r[k + 1 : len(r)])
)


def two_opt(cities, improvement_threshold):
    route = np.arange(cities.shape[0])
    improvement_factor = 1
    best_distance = path_distance(route, cities)
    while improvement_factor > improvement_threshold:
        distance_to_beat = best_distance
        for swap_first in range(1, len(route) - 2):
            for swap_last in range(swap_first + 1, len(route)):
                new_route = two_opt_swap(route, swap_first, swap_last)
                new_distance = path_distance(new_route, cities)
                if new_distance < best_distance:
                    route = new_route
                    best_distance = new_distance

        improvement_factor = 1 - best_distance / distance_to_beat

    return route


def check_obstacle_intersection(p0, p1):

    # convert p0 and p1 to grid cells
    origin = np.array([map_meta.origin.position.x, map_meta.origin.position.y])
    w, h = map_meta.width, map_meta.height
    r = map_meta.resolution

    q0 = (p0 - origin) / r
    q1 = (p1 - origin) / r

    v = q1 - q0
    D = 2 * int(v[1]) - int(v[0])
    y = int(q0[1])

    for x in range(int(q0[0]), int(q1[0]) + 1):
        if map_[x + y * w] > 80:
            return True

        if D > 0:
            y = y + 1
            D = D - 2 * v[0]

        D = D + 2 * v[1]

    return False


# Courtesy of: https://stackoverflow.com/questions/48562739/reducing-number-of-nodes-in-polygons-using-python
def reduce_polygon(polygon, crumbs, angle_th=20):

    angle_th_rad = np.deg2rad(angle_th)
    points_removed = [0]
    while len(points_removed):
        points_removed = list()
        for i in range(len(polygon)):
            v01 = polygon[i - 1] - polygon[i]
            v12 = polygon[i] - polygon[(i + 1) % len(polygon)]

            d01 = np.linalg.norm(v01)
            d12 = np.linalg.norm(v12)

            angle = np.arccos(np.sum(v01 * v12) / (d01 * d12))

            vec = polygon[(i + 1) % len(polygon)] - polygon[i-1]
            # theta = math.atan2(vec[1],vec[0])
            theta = math.atan(vec[1]/vec[0])

            ori = crumbs[i][2]
            crumb_ori = euler_from_quaternion((ori.x,ori.y,ori.z,ori.w))
            crumb_yaw = crumb_ori[2]

            if check_obstacle_intersection(
                polygon[i], polygon[i - 1]
            ) or check_obstacle_intersection(
                polygon[i], polygon[(i + 1) % len(polygon)]
            ):
                # print("Obstacle intersection with point {}".format(polygon[i]))
            	continue

            if angle < angle_th_rad:# and math.fabs(theta - crumb_yaw) < angle_th_rad:
            	# print("crumb i-1 is ({})".format(polygon[i-1]))
            	# print("crumb i is ({})".format(polygon[i]))
            	# print("crumb i+1 is ({})".format(polygon[(i+1)%len(polygon)]))

            	# print("Angle b/w i-1 and i+1 is {}".format(theta))
            	# print("Angle of i is {}".format(crumb_yaw))
            	# print("---------------------------------------------------------")
                points_removed.append(i)

        for ind in reversed(list(points_removed)):
            del crumbs[ind]

        polygon = np.delete(polygon, points_removed, axis=0)

    return polygon, points_removed


def getNavigated(data):
    global experiment, switch, poses

    if switch:
        poses = []
        switch = False

    experiment = data.data


def getStart(data):
    global start
    start = data.data


def main():

    global experiment

    rospy.init_node("lidar_area_mapper", anonymous=True)

    rospy.Subscriber("/map", OccupancyGrid, updateMap)
    rospy.Subscriber("/front/scan", LaserScan, getLidarData)

    rospy.Subscriber("/navigating_crumbs", Bool, getNavigated)
    rospy.Subscriber("/start_recording", Bool, getStart)

    rospy.Subscriber("/gmapping/odometry", Odometry, getJackalPosOdom)

    if not experiment:
        path_pub = rospy.Publisher("crumb_path", Marker, queue_size=10)
        marker_pub = rospy.Publisher("crumbs", MarkerArray, queue_size=100)

    poly_pub = rospy.Publisher("coverage_outline", geoPoly, queue_size=100)

    

    rate = rospy.Rate(Hz)

    marker_arr = MarkerArray()
    marker_arr.markers = []

    areas = []

    boundary = geoPoly()
    boundary.header.frame_id = "map"
    x, y = 0, 0

    delete_msg = Marker()
    delete_msg.header.frame_id = "map"
    delete_msg.header.stamp = rospy.Time.now()
    delete_msg.ns = "visibilities"
    delete_msg.action = Marker.DELETEALL

    while not rospy.is_shutdown():
        rate.sleep()

        if jackal_pos == None or lidar == None or map_ == None:
            continue


        if not experiment:
            print("no experiment yet")


        if experiment and not start:
            print("nost started yet")
            marker_pub.publish(marker_arr)
            continue

        # pose list didn't update, no need to rerun anything
        if not update_poses():
            marker_pub.publish(marker_arr)
            continue

        if experiment:
            
            print(len(poses))
            if not global_poly.is_empty:

                if global_poly.type == "MultiPolygon":
                    print("Geometry is MultiPolygon, can't pulish boundary")
                else:
                    x, y = global_poly.exterior.xy
                    z = [0 for elem in x]
                    points = []

                    for x, y, z in zip(x, y, z):
                        points.append(geoPoint(x, y, z))

                    boundary.header.stamp = rospy.Time.now()
                    boundary.polygon.points = points
                    marker_pub.publish(marker_arr)
                    poly_pub.publish(boundary)

                    print("pub time")

            continue

        marker_arr.markers = [delete_msg]
        crumbs = []
        points_removed = []
        crumbs = newUpdate(marker_arr.markers, areas)
        all_crumbs = list(crumbs)

        lines = Marker()
        if len(crumbs) > 1:

            cities = np.array([[c[1].x, c[1].y] for c in crumbs])
            route = two_opt(cities, 0.001)
            crumbs = np.array(crumbs)
            crumbs = crumbs[route]
            crumbs = crumbs.tolist()
            cities = cities[route]

            if len(crumbs) > 2:
	            print("Crumb len before: {}".format(len(crumbs)))
	            cities, points_removed = reduce_polygon(cities, crumbs)
	            print("Crumb len after: {}".format(len(crumbs)))

            lines.header.frame_id = "map"
            lines.header.stamp = rospy.Time.now()
            lines.ns = "path"
            lines.id = 420
            lines.action = Marker.ADD
            lines.type = Marker.LINE_LIST
            lines.scale.x = 0.5
            lines.color.r, lines.color.g, lines.color.b, lines.color.a = (1, 0, 0, 1)

            for i, city in enumerate(cities):
                point1 = geoPoint()
                point1.x = city[0]
                point1.y = city[1]
                lines.points.append(point1)

                point2 = geoPoint()
                point2.x = cities[(i + 1)% len(cities)][0]
                point2.y = cities[(i + 1)% len(cities)][1]
                lines.points.append(point2)


        for ind, (poly, pos, ori) in enumerate(crumbs):
            msg = generate_message(pos, ori, ind)
            marker_arr.markers.append(msg)


        for ind in points_removed:
        	poly, pos, ori = all_crumbs[ind]
        	msg = generate_message(pos, ori, ind+len(crumbs),color=(1,0,0,1))
        	marker_arr.markers.append(msg)


        poly = unary_union([el[0] for el in all_crumbs])

        if not poly.is_empty:

            if poly.type == "MultiPolygon":
                print("Geometry is MultiPolygon, can't pulish boundary")
            else:
                x, y = poly.exterior.xy
                z = [0 for elem in x]
                points = []

                for x, y, z in zip(x, y, z):
                    points.append(geoPoint(x, y, z))

                boundary.header.stamp = rospy.Time.now()
                boundary.polygon.points = points
                poly_pub.publish(boundary)

        marker_pub.publish(marker_arr)
        path_pub.publish(lines)


if __name__ == "__main__":
    main()
