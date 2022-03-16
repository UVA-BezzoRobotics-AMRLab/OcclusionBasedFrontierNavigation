#! /usr/bin/env python

import sys
import time
import math
import rospy
import random
import tf2_ros
import actionlib

from GoalManager import *
from std_msgs.msg import Bool
from OcclusionManager import *
from sklearn.neighbors import KDTree
from sensor_msgs.msg import LaserScan
from actionlib_msgs.msg import GoalStatus
from std_msgs.msg import Float32MultiArray
from tf.transformations import euler_from_quaternion
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Pose, Point, TransformStamped
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from nav_msgs.msg import Odometry, MapMetaData, OccupancyGrid


jackal_pos = None
initial_jackal_pos = None

OBSTACLE_PROB_THRESH = 80

# Dimensions in meters
JACKAL_WIDTH = 0.43
JACKAL_LENGTH = 0.508

# Lidar Sensor Data
MAX_RANGE = None  # 30
MIN_READING = math.sqrt(JACKAL_WIDTH ** 2 + JACKAL_LENGTH ** 2)

SAMPLES = None
angles = None
incA = None
minA = None

regions = None
counter = 0
points = 0

marker_pub = marker_pub1 = None

occlusion_buffer = []
frontiers = None
delete_buffer = []
last_window_pos = None

debug = True

client = None

occ_man = None
tmp_count = 0

ns = "occlusion_detector"


def getLidarData(data):
    global regions, minA, incA, SAMPLES, last_window_pos, MAX_RANGE

    regions = data.ranges

    MAX_RANGE = data.range_max
    SAMPLES = len(regions)
    last_window_pos = SAMPLES - 2
    minA = data.angle_min
    incA = data.angle_increment

    occ_man.update_lidar(data)


def getJackalPos(data):
    global jackal_pos, initial_jackal_pos

    if initial_jackal_pos == None:
        initial_jackal_pos = data

    jackal_pos = data


def getJackalPosOdom(data):
    global jackal_pos, initial_jackal_pos

    if initial_jackal_pos == None:
        initial_jackal_pos = data.pose.pose

    jackal_pos = data.pose.pose


def getJackalPosVicon(data):

    if use_vicon:
        global jackal_pos, initial_jackal_pos

        jackal_pos = Pose()
        jackal_pos.position.x = data.transform.translation.x
        jackal_pos.position.y = data.transform.translation.y
        jackal_pos.position.z = data.transform.translation.z

        jackal_pos.orientation.x = data.transform.rotation.x
        jackal_pos.orientation.y = data.transform.rotation.y
        jackal_pos.orientation.z = data.transform.rotation.z
        jackal_pos.orientation.w = data.transform.rotation.w

        if initial_jackal_pos == None:
            initial_jackal_pos = jackal_pos

    else:
        global vicon_pose

        vicon_pose = Pose()
        vicon_pose.position.x = data.transform.translation.x
        vicon_pose.position.y = data.transform.translation.y
        vicon_pose.position.z = data.transform.translation.z

        vicon_pose.orientation.x = data.transform.rotation.x
        vicon_pose.orientation.y = data.transform.rotation.y
        vicon_pose.orientation.z = data.transform.rotation.z
        vicon_pose.orientation.w = data.transform.rotation.w


def get_frontiers(data):
    global frontiers
    stride = data.layout.dim[1].stride
    frontiers = []

    for i in range(0, len(data.data), stride):

        x = data.data[i]
        y = data.data[i + 1]

        occ = Occlusion(
            mx=x,
            my=y,
            color=OcclusionState.colors[OcclusionState.frontier],
            iden=0,
            shape=Marker.CUBE,
            scale=.8,
            pointA=None,
            pointB=None,
            state=OcclusionState.frontier,
            time_created=rospy.get_time(),
        )
        frontiers.append(occ)

    print("There are {} frontiers".format(len(frontiers)))


def identify_occlusions():

    global counter, angles, regions, initial_jackal_pos, jackal_pos, last_window_pos, MAX_RANGE, occ_man

    colors = OcclusionState.colors
    if initial_jackal_pos == None or regions == None:
        print("darn")
        return

    # Number of meters for threshold
    threshold = occlusion_threshold

    epsilon = 0.3

    window_size = 50

    change_flag = False

    occlusions = []
    blacklisted_rays = [0] * SAMPLES

    obstacle_regions = []
    freespace_regions = []

    last_obs = -1
    last_free = -1

    if include_occluded_space:
        for window in range(SAMPLES - 1):

            if regions[window] != float("inf"):

                if last_obs + 1 < window:
                    if len(obstacle_regions) != 0 and len(obstacle_regions[-1]) < 5:
                        del obstacle_regions[-1]

                    obstacle_regions.append([])

                if len(obstacle_regions) == 0:
                    obstacle_regions.append([])

                last_obs = window

                if (
                    len(obstacle_regions[-1]) == 0
                    or math.fabs(obstacle_regions[-1][-1][0] - regions[window]) <= 0.15
                ):
                    obstacle_regions[-1].append((regions[window], angles[window]))

                else:
                    if len(obstacle_regions[-1]) < 5:
                        del obstacle_regions[-1]

                    obstacle_regions.append([])

                    obstacle_regions[-1].append((regions[window], angles[window]))

            if regions[window] >= 5:

                if last_free + 1 < window:
                    if len(freespace_regions) != 0 and len(freespace_regions[-1]) < 5:
                        del freespace_regions[-1]

                    freespace_regions.append([])

                if len(freespace_regions) == 0:
                    freespace_regions.append([])

                freespace_regions[-1].append((regions[window], angles[window]))
                last_free = window

            # if window has two infinite readings, do not try to plot the point
            if regions[window] == float("inf") and regions[window + 1] == float("inf"):
                pass
            elif regions[window] < MIN_READING or regions[window + 1] < MIN_READING:
                pass
            elif (
                math.fabs(regions[window] - regions[window + 1])
                > threshold *epsilon * (regions[window] + regions[window + 1]) / 2
            ):

                jackal_orientation = (
                    jackal_pos.orientation.x,
                    jackal_pos.orientation.y,
                    jackal_pos.orientation.z,
                    jackal_pos.orientation.w,
                )
                jackal_orient_euler = euler_from_quaternion(jackal_orientation)
                jackal_yaw = jackal_orient_euler[2]

                angle1 = angles[window] + jackal_yaw
                angle2 = angles[window + 1] + jackal_yaw

                # find the points in the global space to be plotted in rviz
                x, y = (
                    regions[window] * math.cos(angle1) + jackal_pos.position.x,
                    regions[window] * math.sin(angle1) + jackal_pos.position.y,
                )

                # Case 1
                is_valid = True
                if regions[window] > regions[(window + 1) % SAMPLES]:

                    for k in range(1, window_size + 1):

                        angle1 = angles[(window - k) % SAMPLES] + jackal_yaw
                        angle2 = angles[(window - k + 1) % SAMPLES] + jackal_yaw

                        xw, yw = (
                            regions[(window - k) % SAMPLES] * math.cos(angle1)
                            + jackal_pos.position.x,
                            regions[(window - k + 1) % SAMPLES] * math.sin(angle2)
                            + jackal_pos.position.y,
                        )

                        if (x - xw) * (x - xw) + (y - yw) * (
                            y - yw
                        ) < JACKAL_WIDTH * JACKAL_WIDTH * 16 / 9:
                            is_valid = False
                            break

                # Case 2
                elif regions[window] < regions[(window + 1) % SAMPLES]:

                    for k in range(1, window_size + 1):

                        angle1 = angles[(window + k) % SAMPLES] + jackal_yaw
                        angle2 = angles[(window + k + 1) % SAMPLES] + jackal_yaw

                        xw, yw = (
                            regions[(window + k) % SAMPLES] * math.cos(angle1)
                            + jackal_pos.position.x,
                            regions[(window + k + 1) % SAMPLES] * math.sin(angle2)
                            + jackal_pos.position.y,
                        )

                        if (x - xw) * (x - xw) + (y - yw) * (
                            y - yw
                        ) < JACKAL_WIDTH * JACKAL_WIDTH * 16 / 9:
                            is_valid = False
                            break

                if not is_valid:
                    continue

                a, b = regions[window], regions[window + 1]

                state = OcclusionState.nearby

                if a == float("inf"):
                    state = OcclusionState.horizon
                    a = MAX_RANGE
                    continue
                elif b == float("inf"):
                    state = OcclusionState.horizon
                    b = MAX_RANGE
                    continue

                color = colors[state]

                x1, y1, x2, y2, mx, my, scale = calcOcclusion(window, jackal_pos, a, b)

                # Box dimensions are 5.3m tall by 4m wide
                if use_vicon and use_gate:

                    if math.fabs(mx) > 3.7 / 2 or math.fabs(my) > 6 / 2:
                        # color = (1,0,0,1)
                        continue

                elif use_gate:

                    # Need to get position based on vicon location
                    tmpx1, tmpy1, tmpx2, tmpy2, tmpx, tmpy, scale = calcOcclusion(
                        window, vicon_pose, a, b
                    )

                    if math.fabs(tmpx) > 3.7 / 2 or math.fabs(tmpy) > 6 / 2:
                        # color = (1,0,0,1)
                        continue

                if (
                    scale ** 2 > min_occlusion_size
                ):  # (JACKAL_LENGTH ** 2 + JACKAL_WIDTH ** 2)**.5:

                    occ = Occlusion(
                        mx,
                        my,
                        color,
                        0,
                        Marker.SPHERE,
                        scale,
                        (x1, y1),
                        (x2, y2),
                        state,
                        rospy.get_time(),
                    )

                    occlusions.append(occ)
                    blacklisted_rays[window] = 1
                    blacklisted_rays[window + 1] = 1

    # occ_man.checkIntersections(regions, angles, jackal_pos, blacklisted_rays)

    if include_shadow_space:

        shadowCentroids, shadow_line_list = calculateShadowCentroids(obstacle_regions)
        for (x, y) in shadowCentroids:
            occ = Occlusion(
                mx=x,
                my=y,
                color=OcclusionState.colors[OcclusionState.centroid],
                iden=0,
                shape=Marker.CUBE,
                scale=1.0,
                pointA=None,
                pointB=None,
                state=OcclusionState.centroid,
                time_created=rospy.get_time(),
            )
            occlusions.append(occ)

    if include_open_space:

        openCentroids, open_line_list = calculateOpenCentroids(freespace_regions)
        for (x, y) in openCentroids:
            occ = Occlusion(
                mx=x,
                my=y,
                color=OcclusionState.colors[OcclusionState.free_centroid],
                iden=0,
                shape=Marker.CUBE,
                scale=1.0,
                pointA=None,
                pointB=None,
                state=OcclusionState.free_centroid,
                time_created=rospy.get_time(),
            )
            occlusions.append(occ)

    if include_frontiers and frontiers != None:

        for frontier in frontiers:
            occlusions.append(frontier)

    occ_man.updateBuffer(occlusions, jackal_pos)

    if include_open_space:
        occ_man.publishShadows(open_line_list)
    if include_shadow_space:
        occ_man.publishShadows(shadow_line_list)


def calcOcclusion(window, pos, a, b):
    jackal_orientation = (
        pos.orientation.x,
        pos.orientation.y,
        pos.orientation.z,
        pos.orientation.w,
    )
    jackal_orient_euler = euler_from_quaternion(jackal_orientation)
    jackal_yaw = jackal_orient_euler[2]

    angle1 = angles[window] + jackal_yaw
    angle2 = angles[window + 1] + jackal_yaw

    # find the points in the global space to be plotted in rviz
    x1, y1 = (
        a * math.cos(angle1) + pos.position.x,
        a * math.sin(angle1) + pos.position.y,
    )
    x2, y2 = (
        b * math.cos(angle2) + pos.position.x,
        b * math.sin(angle2) + pos.position.y,
    )

    scale = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) / 2

    mx = (x1 + x2) / 2
    my = (y1 + y2) / 2

    return x1, y1, x2, y2, mx, my, scale


# Function to calculate centroids of provided regions
def calculateShadowCentroids(obstacle_regions):
    global jackal_pos
    centroids = []

    jackal_orientation = (
        jackal_pos.orientation.x,
        jackal_pos.orientation.y,
        jackal_pos.orientation.z,
        jackal_pos.orientation.w,
    )
    jackal_orient_euler = euler_from_quaternion(jackal_orientation)
    jackal_yaw = jackal_orient_euler[2]

    jackal_vicon_yaw = None
    if use_gate and not use_vicon:
        ori = (
            vicon_pose.orientation.x,
            vicon_pose.orientation.y,
            vicon_pose.orientation.z,
            vicon_pose.orientation.w,
        )

    line_list = Marker()
    line_list.header.frame_id = "map"
    line_list.header.stamp = rospy.Time.now()
    line_list.ns = "shadows"
    line_list.id = 1
    line_list.type = Marker.LINE_LIST
    line_list.scale.x = 0.1
    line_list.color.r = 0.18
    line_list.color.b = 0.31
    line_list.color.g = 0.31
    line_list.color.a = 0.2
    line_list.action = Marker.ADD

    for ind, region in enumerate(obstacle_regions):
        x_m, y_m = 0, 0

        for ind, (dist, angle) in enumerate(region):
            ang = angle + jackal_yaw
            x1, y1 = (
                dist * math.cos(ang) + jackal_pos.position.x,
                dist * math.sin(ang) + jackal_pos.position.y,
            )

            x2, y2 = (
                (dist+4) * math.cos(ang) + jackal_pos.position.x,
                (dist+4) * math.sin(ang) + jackal_pos.position.y,
            )

            x_m += x1 + x2
            y_m += y1 + y2

            p1 = Point()
            p1.x = x1
            p1.y = y1
            p1.z = 1

            p2 = Point()
            p2.x = x2
            p2.y = y2
            p2.z = 1

            line_list.points.append(p1)
            line_list.points.append(p2)

        centroids.append((x_m / (2 * len(region)), y_m / (2 * len(region))))

    return centroids, line_list


def calculateOpenCentroids(free_regions):
    centroids = []

    jackal_orientation = (
        jackal_pos.orientation.x,
        jackal_pos.orientation.y,
        jackal_pos.orientation.z,
        jackal_pos.orientation.w,
    )
    jackal_orient_euler = euler_from_quaternion(jackal_orientation)
    jackal_yaw = jackal_orient_euler[2]

    line_list = Marker()
    line_list.header.frame_id = "map"
    line_list.header.stamp = rospy.Time.now()
    line_list.ns = "freespace"
    line_list.id = 1
    line_list.type = Marker.LINE_LIST
    line_list.scale.x = 0.1
    line_list.color.r = 1
    line_list.color.g = 0.8
    line_list.color.b = 0.79
    line_list.color.a = 0.1
    line_list.action = Marker.ADD

    for ind, region in enumerate(free_regions):
        x_m, y_m = 0, 0

        for ind, (dist, angle) in enumerate(region):
            # if dist > MAX_RANGE:
            #     dist = MAX_RANGE
            if dist > 10:
                dist = 10

            ang = angle + jackal_yaw
            x1, y1 = (
                jackal_pos.position.x,
                jackal_pos.position.y,
            )

            x2, y2 = (
                dist * math.cos(ang) + jackal_pos.position.x,
                dist * math.sin(ang) + jackal_pos.position.y,
            )

            x_m += x1 + x2
            y_m += y1 + y2

            p1 = Point()
            p1.x = x1
            p1.y = y1
            p1.z = 1

            p2 = Point()
            p2.x = x2
            p2.y = y2
            p2.z = 1

            line_list.points.append(p1)
            line_list.points.append(p2)

        centroids.append((x_m / (2 * len(region)), y_m / (2 * len(region))))

    return centroids, line_list


def updateMap(data):
    if occ_man == None:
        return

    occ_man.update_map(data)


def findEscapeGoal():

    if regions == None or jackal_pos == None:
        return

    move_base_goal = occ_man.current_goal
    move_base_goal[0] -= jackal_pos.position.x
    move_base_goal[1] -= jackal_pos.position.y

    jackal_orientation = (
        jackal_pos.orientation.x,
        jackal_pos.orientation.y,
        jackal_pos.orientation.z,
        jackal_pos.orientation.w,
    )
    jackal_orient_euler = euler_from_quaternion(jackal_orientation)
    jackal_yaw = jackal_orient_euler[2]

    # lidar = np.array(regions)
    # angs = np.array(angles)

    # X = np.multiply(lidar-JACKAL_LENGTH,np.cos(angs+jackal_yaw))
    # Y = np.multiply(lidar-JACKAL_LENGTH,np.sin(angs+jackal_yaw))
    # points = np.array((X,Y)).T

    # optimal_goal_ind = np.argmax(points.dot(move_base_goal))
    # rospy.loginfo("Best goal is {}".format(points[optimal_goal_ind]))
    # return points[optimal_goal_ind]
    # sys.exit()

    # Find closest point to obstacle, then move in opposite direction to that point.
    lidar = np.array(regions)
    closest_ind = np.argmax(regions)

    theta = angles[closest_ind] + jackal_yaw - math.pi

    x = lidar[closest_ind] * math.cos(theta) + jackal_pos.position.x
    y = lidar[closest_ind] * math.sin(theta) + jackal_pos.position.y

    return [x, y]


def stop_client():
    global client

    if client != None:
        rospy.loginfo("stopping navigation!")
        client.cancel_goal()

    occ_man.clear_occlusions()


def main():
    global marker_pub, marker_pub1, client, JACKAL_LENGTH, JACKAL_WIDTH, occ_man, jackal_pos, angles

    rospy.init_node(ns, anonymous=True)

    # Read in rosparams
    global include_frontiers
    include_frontiers = (
        rospy.get_param(ns + "/include_frontiers")
        if rospy.has_param(ns + "/include_frontiers")
        else False
    )
    global include_open_space
    include_open_space = (
        rospy.get_param(ns + "/include_open_space")
        if rospy.has_param(ns + "/include_open_space")
        else False
    )
    global include_shadow_space
    include_shadow_space = (
        rospy.get_param(ns + "/include_shadow_space")
        if rospy.has_param(ns + "/include_shadow_space")
        else False
    )
    global include_occluded_space
    include_occluded_space = (
        rospy.get_param(ns + "/include_occluded_space")
        if rospy.has_param(ns + "/include_occluded_space")
        else True
    )

    global occlusion_threshold
    occlusion_threshold = (
        rospy.get_param(ns + "/occlusion_threshold")
        if rospy.has_param(ns + "/occlusion_threshold")
        else False
    )
    global min_occlusion_size
    min_occlusion_size = (
        rospy.get_param(ns + "/min_occlusion_size")
        if rospy.has_param(ns + "/min_occlusion_size")
        else 1.5
    )

    global use_vicon
    use_vicon = (
        rospy.get_param(ns + "/use_vicon")
        if rospy.has_param(ns + "/use_vicon")
        else False
    )
    global use_gate
    use_gate = (
        rospy.get_param(ns + "/use_gate")
        if rospy.has_param(ns + "/use_gate")
        else False
    )

    autonomous = (
        rospy.get_param(ns + "/autonomous")
        if rospy.has_param(ns + "/autonomous")
        else False
    )

    default_goals = []

    marker_pub = rospy.Publisher(
        "visualization_marker_array", MarkerArray, queue_size=1000
    )

    # vel_pub = rospy.Publisher(
    #     "/jackal_velocity_controller/cmd_vel", Twist, queue_size=1
    # )

    client = actionlib.SimpleActionClient("move_base", MoveBaseAction)
    client.wait_for_server()

    occ_man = OcclusionManager(
        client,
        default_goals,
        JACKAL_WIDTH,
        JACKAL_LENGTH,
        marker_pub,
        autonomous=autonomous,
    )

    # rospy.Subscriber("/safety", Bool, occ_man.terminateGoal)
    # rospy.Subscriber("/jackal/global_pos", Pose, getJackalPos)

    if use_vicon or use_gate:
        rospy.Subscriber("/vicon/jackal4/jackal4", TransformStamped, getJackalPosVicon)
    if not use_vicon:
        # rospy.Subscriber("/odometry/filtered", Odometry, getJackalPosOdom)
        rospy.Subscriber("gmapping/odometry", Odometry, getJackalPosOdom)

    if include_frontiers:
        rospy.Subscriber("frontier_centroids", Float32MultiArray, get_frontiers)

    rospy.Subscriber("/scan", LaserScan, getLidarData)
    rospy.Subscriber("/map", OccupancyGrid, updateMap)

    rate = rospy.Rate(5)
    rospy.on_shutdown(stop_client)

    count = 0

    # default_goal = MoveBaseGoal()
    # default_goal.target_pose.header.frame_id = "map"
    # default_goal.target_pose.header.stamp = rospy.Time.now()
    # default_goal.target_pose.pose.position.x = -9
    # default_goal.target_pose.pose.position.y = 2
    # default_goal.target_pose.pose.orientation.w = 1.0

    # goal = Goal(default_goal, 1)
    # default_goals.append(goal)

    escape_goal = None
    while not rospy.is_shutdown():
        rate.sleep()

        if angles == None:

            if incA == None:
                continue

            angles = [minA + i * incA for i in range(SAMPLES)]

        # if include_frontiers and frontiers == None:
        #     continue

        identify_occlusions()
        # findEscapeGoal()
        # rospy.loginfo(client.get_state())
        # occ_man.set_goal(jackal_pos)
        occ_man.publishOcclusions(debug)

        if occ_man.state == ManagerState.AVOIDING_OBSTACLE:

            if escape_goal == None:
                escape_goal = findEscapeGoal()

            # figure out trajectory for next time step

        # if include_frontiers and len(frontiers) == 0:
        #     rospy.loginfo("Exploration complete")
        #     break

    client.cancel_goal()


if __name__ == "__main__":
    main()
