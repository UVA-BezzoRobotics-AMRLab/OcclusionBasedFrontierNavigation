#! /usr/bin/env python
import sys
import yaml
import time
import math
import rospy
import random
import tf2_ros
import actionlib
import matplotlib.pyplot as plt

from GoalManager import *
from std_msgs.msg import Bool
from OcclusionManager import *
from yaml import CLoader as Loader
from collections import defaultdict
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Pose, Point
from actionlib_msgs.msg import GoalStatus
from tf.transformations import euler_from_quaternion
from visualization_msgs.msg import Marker, MarkerArray
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from nav_msgs.msg import Odometry, MapMetaData, OccupancyGrid

markers = None

occupancy_grid = None

# Calculate the euclidian distance in n-space of the route r traversing cities c, ending at the path start.
path_distance = lambda r, c: np.sum(
    [np.linalg.norm(c[r[p]] - c[r[p - 1]]) for p in range(len(r))]
)
# Reverse the order of all elements from element i to element k in array r.
two_opt_swap = lambda r, i, k: np.concatenate(
    (r[0:i], r[k : -len(r) + i - 1 : -1], r[k + 1 : len(r)])
)

client = None
start = False

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


def get_crumb_msg(crumbs, ends):

    markers_p = MarkerArray()
    markers_p.markers = []

    for ind, (crumb, end) in enumerate(zip(crumbs, ends)):
        msg = Marker()
        msg.header.frame_id = "map"
        msg.header.stamp = rospy.Time.now()
        msg.ns = "crumbs"
        msg.id = ind
        msg.scale.x = 0.4
        msg.scale.y = 1
        msg.scale.z = 0.2
        msg.color.r, msg.color.g, msg.color.b, msg.color.a = (1, 1, 0, 1)
        msg.action = Marker.ADD
        msg.type = Marker.ARROW
        msg.points = []

        start_p = Point()
        end_p = Point()

        start_p.x = crumb[0]
        start_p.y = crumb[1]
        start_p.z = 1

        end_p.x = end[0]
        end_p.y = end[1]
        end_p.z = 1

        msg.points.append(start_p)
        msg.points.append(end_p)

        markers_p.markers.append(msg)

    return markers_p


def getJackalPos(data):
    global jackal_pos

    jackal_pos = data

def getJackalPosOdom(data):
    global jackal_pos

    jackal_pos = data.pose.pose

def updateMap(data):
    global occupancy_grid

    occupancy_grid = data


def dist2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

def getGoalCoords(mx,my):

    global jackal_pos, occupancy_grid

    if occupancy_grid is None:
        return (float("inf"), float("inf"))

    # Get corners of occupancy_grid
    origin = occupancy_grid.info.origin
    resolution = occupancy_grid.info.resolution
    delta = 0.995
    w, h = (
        occupancy_grid.info.width * resolution,
        occupancy_grid.info.height * resolution,
    )

    A = np.array([origin.position.x, origin.position.y + h])  # tl
    B = np.array([origin.position.x + w, origin.position.y + h])  # tr
    C = np.array([origin.position.x + w, origin.position.y])  # br
    D = np.array([origin.position.x, origin.position.y])  # bl

    P = np.array([mx, my])

    if P[0] >= A[0] and P[0] <= B[0] and P[1] >= D[1] and P[1] <= A[1]:
        # rospy.loginfo("Goal is inside occupancy grid")
        return (mx, my)
    # else:
    #     return (float("inf"), float("inf"))

    # Find the line between jackal and occlusion
    J = np.array([jackal_pos.position.x, jackal_pos.position.y])

    intersection = None
    if isSegIntersecting(A, B, J, P):
        print("Intersects top")
        intersection = findSegInt(A, B, J, P)
    elif isSegIntersecting(A, D, J, P):
        print("Intersects left")
        intersection = findSegInt(A, D, J, P)
    elif isSegIntersecting(C, D, J, P):
        print("Intersects bottom")
        intersection = findSegInt(C, D, J, P)
    elif isSegIntersecting(B, C, J, P):
        print("Intersects right")
        intersection = findSegInt(B, C, J, P)

    if intersection != None:
        max_x = (origin.position.x + w) * delta
        max_y = (origin.position.y + h) * delta
        min_x = origin.position.x * delta
        min_y = origin.position.y * delta
        return (
            min(max(intersection[0], min_x), max_x),
            min(max(intersection[1], min_y), max_y),
        )

    rospy.loginfo("Error occured in calculation of getGoalCoords()")
    return (float("inf"), float("inf"))


# Courtesy of stack overflow
def isSegIntersecting(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def perp(a):
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b


# line segment a given by endpoints a1, a2
# line segment b given by endpoints b1, b2
def findSegInt(a1, a2, b1, b2):
    da = a2 - a1
    db = b2 - b1
    dp = a1 - b1
    dap = perp(da)
    denom = np.dot(dap, db)
    num = np.dot(dap, dp)
    return (num / denom.astype(float)) * db + b1


def send_goal(goalx, goaly):

    angle = math.atan2(goaly, goalx) + np.pi / 2
    if angle > 2 * np.pi:
        angle -= 2 * np.pi

    goal_heading = angle
    quat = quaternion_from_euler(0, 0, angle)

    msg = MoveBaseGoal()
    msg.target_pose.header.frame_id = "map"
    msg.target_pose.header.stamp = rospy.Time.now()
    msg.target_pose.pose.position.x = goalx
    msg.target_pose.pose.position.y = goaly
    msg.target_pose.pose.orientation.x = quat[0]
    msg.target_pose.pose.orientation.y = quat[1]
    msg.target_pose.pose.orientation.z = quat[2]
    msg.target_pose.pose.orientation.w = quat[3]

    client.send_goal(msg)


def getCrumbs(data):
    global markers

    markers = data.markers


def stop_client():
    global client, crumbs, route

    if client != None:
        rospy.loginfo("stopping navigation!")
        client.cancel_goal()

    plt.scatter(crumbs[route][:, 0], crumbs[route][:, 1])
    plt.plot(crumbs[route][:, 0], crumbs[route][:, 1])
    dist = 0

    for ind, point in enumerate(crumbs[route]):

        if ind == len(crumbs[route]) - 1:
            dist += dist2(point, crumbs[route][0]) ** 0.5
            break

        dist += dist2(point, crumbs[route][ind + 1]) ** 0.5

        plt.annotate(
            str(ind), point, textcoords="offset points", xytext=(0, 10), ha="right"
        )

    plt.title("Walk length is {}m".format(round(dist, 2)))
    plt.show()


def main():

    global client, markers, crumbs, route, start

    experiment = len(sys.argv) > 1

    rospy.init_node("navigate_crumbs", anonymous=True)

    marker_pub = rospy.Publisher("breadcrumbs", MarkerArray, queue_size=1000)
    bool_pub = rospy.Publisher("/navigating_crumbs", Bool, queue_size=10)
    start_pub = rospy.Publisher("/start_recording", Bool, queue_size=10)

    rospy.Subscriber("/gmapping/odometry", Odometry, getJackalPosOdom)
    rospy.Subscriber("/map", OccupancyGrid, updateMap)
    # rospy.Subscriber("/jackal/global_pos", Pose, getJackalPos)

    if experiment:

        rospy.Subscriber("/crumbs", MarkerArray, getCrumbs)

    client = actionlib.SimpleActionClient("move_base", MoveBaseAction)
    # client.wait_for_server()

    rate = rospy.Rate(2)
    rospy.on_shutdown(stop_client)

    crumbs = None
    end = None

    route = None
    data = None

    modified_goal = False
    goalx, goaly = 0,0
    target_ind = 0

    while not rospy.is_shutdown():
        rate.sleep()

        if jackal_pos == None:
            continue


        if crumbs == None:

            if not experiment:
                
                with open("/home/nm9ur/catkin_ws/src/jackal_nodes/experiments/cluttered_env/bags/crumbs.yaml", "r") as f:

                    try:
                        data = yaml.load(f, Loader=Loader)
                    except yaml.YAMLError as exc:
                        print(exc)
                        sys.exit()

                data = data['markers'][1:]
                crumbs = []
                end = []

                for crumb in data:
                    if len(crumb['points']) == 0:
                        continue

                    p1 = [float(crumb['points'][0]['x']),float(crumb['points'][0]['y'])]
                    p2 = [float(crumb['points'][1]['x']),float(crumb['points'][1]['y'])]
                    crumbs.append(p1)

                    end.append(p2)

            elif experiment:

                if markers == None:
                    continue

                crumbs = []
                end = []

                for marker in markers:
                    if len(marker.points) == 0:
                        continue

                    p1 = [float(marker.points[0].x), float(marker.points[0].y)]
                    p2 = [float(marker.points[1].x), float(marker.points[1].y)]
                    crumbs.append(p1)

                    end.append(p2)

            # crumbs.append(crumbs[0])
            p = [jackal_pos.position.x, jackal_pos.position.y]
            crumbs.insert(0,p)
            crumbs = np.array(crumbs)
            end = np.array(end)

            route = two_opt(crumbs, 0.001)
            print("before: ", route)
            crumbs = crumbs[1:]
            route = [r -1 for r in route if r > 0]

            print(crumbs)
            print(route)
            print(target_ind)

            tx = crumbs[route[target_ind]][0]
            ty = crumbs[route[target_ind]][1]
            ax,ay = getGoalCoords(tx*1.1,ty*1.1)

            if math.fabs(ax-tx*1.1) > .001 or math.fabs(ay-ty*1.1) > .001:
                modified_goal = True
                goalx,goaly = ax,ay
            else:
                goalx,goaly = tx,ty

            send_goal(goalx, goaly)
            rospy.loginfo("sent first goal {}".format(crumbs[route[target_ind]]))


        crumbs_msg = get_crumb_msg(crumbs, end)
        marker_pub.publish(crumbs_msg)

        client_state = client.get_state()
        if (
            client_state != GoalStatus.ABORTED
            and client_state != GoalStatus.LOST
            and client_state != GoalStatus.SUCCEEDED
        ):

            jackal_orientation = (
                jackal_pos.orientation.x,
                jackal_pos.orientation.y,
                jackal_pos.orientation.z,
                jackal_pos.orientation.w,
            )
            jackal_orient_euler = euler_from_quaternion(jackal_orientation)
            jackal_yaw = jackal_orient_euler[2]
            # rospy.loginfo(
            #     "Goal heading is {} and jackal heading is {}".format(
            #         goal_heading, jackal_yaw
            #     )
            # )

            if modified_goal:

                tx = crumbs[route[target_ind]][0]
                ty = crumbs[route[target_ind]][1]
                ax,ay = getGoalCoords(tx*1.1,ty*1.1)

                if math.fabs(ax-tx*1.1) <= .001 and math.fabs(ay-ty*1.1) <= .001:
                    modified_goal = False
                    client.cancel_goal()
                    goalx, goaly = tx, ty
                    send_goal(tx, ty)

                    rospy.loginfo("costmap updated, sending new actual goal ({},{})".format(goalx, goaly))

        distance = dist2(
            [goalx,goaly],
            [jackal_pos.position.x, jackal_pos.position.y],
        )
        if distance < 0.2 or client_state == GoalStatus.SUCCEEDED or client_state == GoalStatus.ABORTED or client_state == GoalStatus.LOST: #and math.fabs(jackal_yaw - goal_heading) < 0.1:
            rospy.loginfo([jackal_pos.position.x, jackal_pos.position.y])
            client.cancel_goal()
            target_ind += 1
            start = True

            tx = crumbs[route[target_ind]][0]
            ty = crumbs[route[target_ind]][1]
            ax,ay = getGoalCoords(tx*1.1,ty*1.1)

            if math.fabs(ax-tx*1.1) > .001 or math.fabs(ay-ty*1.1) > .001:
                modified_goal = True
                goalx,goaly = ax,ay
            else:
                goalx,goaly = tx,ty

            send_goal(goalx, goaly)

            if client_state == GoalStatus.LOST or client_state == GoalStatus.ABORTED:
                rospy.loginfo("Failed to reach goal!")

            rospy.loginfo("Goal reached, sent new goal ({},{})".format(goalx,goaly))


        msg = Bool()
        msg.data = True
        bool_pub.publish(msg)

        msg = Bool()
        msg.data = start
        start_pub.publish(msg)

if __name__ == "__main__":
    main()
