#! /usr/bin/env python
import sys
import time
import math
import copy
import rospy
import random
import tf2_ros
import actionlib
import numpy as np

from GoalManager import *
from collections import deque
from geometry_msgs.msg import Pose
from sklearn.neighbors import KDTree
from sensor_msgs.msg import LaserScan
from scipy.spatial.distance import cdist
from actionlib_msgs.msg import GoalStatus
from visualization_msgs.msg import Marker, MarkerArray
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from nav_msgs.msg import Odometry, MapMetaData, OccupancyGrid
from tf.transformations import euler_from_quaternion, quaternion_from_euler


class OcclusionState:

    default = 0
    nearby = 1
    horizon = 2
    goal = 3
    free_space = 4
    centroid = 5
    free_centroid = 6
    frontier = 7
    invalid = 8

    colors = [
        (0, 0, 0, 0),
        (0, 0, 1, 0.5),
        (0, 1, 0, 0.5),
        (1, 1, 0, 0.5),
        (1, 0, 1, 0.5),
        (0, 1, 1, 0.5),
        (1, 0, 0, 0.5),
        (1, 0.5, 0.5, 0.5),
        (1, 0, 0, 1),
    ]


class ManagerState:

    WAITING_FOR_GOAL = 1
    GOING_TO_GOAL = 2
    MOVE_TO_DEFAULT_GOAL = 3
    AVOIDING_OBSTACLE = 4
    ROTATING = 5


class Occlusion:
    def __init__(
        self, mx, my, color, iden, shape, scale, pointA, pointB, state, time_created
    ):

        self.mx = mx
        self.my = my
        self.color = color
        self.iden = iden
        self.shape = shape
        self.scale = scale
        self.pointA = pointA
        self.pointB = pointB

        # If counter reaches theta, we know it is a valid occlusion
        self.counter = 0
        # self.is_valid = False
        self.is_valid = True
        # self.threshold = 3
        self.threshold = 0

        self.max_priority = 20
        self.priority = 0

        self.is_goal = False
        self.state = state
        self.initial_state = state

        self.times_goal = 0
        self.times_failed = 0

        self.time_created = time_created

    def set_goal(self):
        self.is_goal = True
        self.state = OcclusionState.goal
        self.color = OcclusionState.colors[self.state]

    def unset_goal(self):
        self.is_goal = False
        self.state = self.initial_state
        self.color = OcclusionState.colors[self.state]

    def get_coords(self):
        return (self.mx, self.my)

    def updateCounter(self):

        if self.is_valid:
            return

        self.counter += 1

        if self.counter > self.threshold:
            self.is_valid = True

    def updatePriority(self):

        if self.priority < self.max_priority:
            self.priority += 1
        else:
            return

        if self.color == (1, 1, 0, 1) or (1, 0, 1, 0.5):
            return

        if self.color[1] != 0:
            self.color = (
                self.priority / self.max_priority,
                1 - (self.priority / self.max_priority),
                0,
                1,
            )
        elif self.color[2] != 0:
            self.color = (
                self.priority / self.max_priority,
                0,
                1 - (self.priority / self.max_priority),
                1,
            )

    def __str__(self):
        return "({},{}): id = {}".format(
            round(self.mx, 2), round(self.my, 2), self.iden
        )


class OcclusionManager:
    def __init__(
        self,
        client,
        default_goals,
        jackal_width,
        jackal_length,
        publisher,
        autonomous=True,
    ):
        self.occlusion_buffer = []
        self.publish_buffer = []
        self.delete_buffer = []

        # parameters about occlusions
        self.epsilon = 0.5
        self.buffer_len = 100
        self.threshold = 1

        self.counter = 0
        self.max_counter = 30
        self.is_updated = False
        self.client = client
        self.state = ManagerState.WAITING_FOR_GOAL

        self.default_goals = default_goals
        self.deleted_goals = []

        self.jackal_width = jackal_width
        self.jackal_length = jackal_length

        self.occupancy_grid = None
        self.publisher = publisher

        self.current_default_goal = 0
        self.current_goal = [float("inf"), float("inf")]
        self.current_goal_occ = None

        self.autonomous = autonomous
        self.jackalpos = None
        self.ranges = [float("inf")] * 720

        self.timespan = 30
        self.freq = 5
        self.previous_positions = deque(maxlen=self.timespan * self.freq)
        self.max_attempts = self.timespan * self.freq
        self.positions_to_try = None

        self.visited = []

    def update_map(self, data):

        self.occupancy_grid = data

    def update_lidar(self, data):

        self.ranges = data.ranges

    def set_goal(self, jackal_pos, goal_deleted=False):

        # return True

        client_state = self.client.get_state()

        if (
            self.state == ManagerState.AVOIDING_OBSTACLE
            or self.state == ManagerState.ROTATING
        ):

            # Robot has yet to complete its recovery goal, do not try to set another goal yet
            if (
                client_state != GoalStatus.SUCCEEDED
                and client_state != GoalStatus.LOST
                and client_state != GoalStatus.ABORTED
            ):
                return False

        # Naive, come back and do something more intelligent than this
        if goal_deleted:
            self.visited.append([jackal_pos.position.x, jackal_pos.position.y])
            # rospy.loginfo("goal has been deleted")
            self.state = ManagerState.WAITING_FOR_GOAL

        if len(self.default_goals) == 0 and len(self.occlusion_buffer) == 0:
            rospy.loginfo("no goals to navigate towards")
            self.state = ManagerState.WAITING_FOR_GOAL
            return False

        if self.state != ManagerState.WAITING_FOR_GOAL:

            if client_state == GoalStatus.ABORTED:

                if (
                    self.state != ManagerState.AVOIDING_OBSTACLE
                    and self.state != ManagerState.ROTATING
                ):
                    rospy.loginfo("Navigation aborted, starting recovery behavior")
                    self.state = ManagerState.ROTATING
                    self.positions_to_try = copy.deepcopy(self.previous_positions)
                    self.current_goal_occ.times_failed += 1

                if self.state == ManagerState.AVOIDING_OBSTACLE:
                    self.state = ManagerState.ROTATING

                    # Begin rotating behavior, try to rotate 180 degrees, twice
                    jackal_orientation = (
                        jackal_pos.orientation.x,
                        jackal_pos.orientation.y,
                        jackal_pos.orientation.z,
                        jackal_pos.orientation.w,
                    )
                    jack_orient = list(euler_from_quaternion(jackal_orientation))
                    jack_orient[2] = (jack_orient[2] + 180) % 360

                    goal_orientation = quaternion_from_euler(
                        jack_orient[0], jack_orient[1], jack_orient[2]
                    )

                    rospy.loginfo(
                        "Trying to rotate 180 degrees in place: {}".format(
                            goal_orientation
                        )
                    )
                    msg = MoveBaseGoal()
                    msg.target_pose.header.frame_id = "map"
                    msg.target_pose.header.stamp = rospy.Time.now()
                    msg.target_pose.pose.position.x = jackal_pos.position.x
                    msg.target_pose.pose.position.y = jackal_pos.position.y
                    msg.target_pose.pose.orientation.x = goal_orientation[0]
                    msg.target_pose.pose.orientation.y = goal_orientation[1]
                    msg.target_pose.pose.orientation.z = goal_orientation[2]
                    msg.target_pose.pose.orientation.w = goal_orientation[3]

                    # self.client.send_goal(msg)
                    return True

                elif self.state == ManagerState.ROTATING:
                    self.state = ManagerState.AVOIDING_OBSTACLE

                    if len(self.positions_to_try) == 0:
                        rospy.signal_shutdown(
                            "Failed to unstuck robot. Shutting down node"
                        )

                    x, y = self.positions_to_try.popleft()

                    rospy.loginfo(
                        "Robot is stuck, setting goal to ({},{})".format(x, y)
                    )

                    msg = MoveBaseGoal()
                    msg.target_pose.header.frame_id = "map"
                    msg.target_pose.header.stamp = rospy.Time.now()
                    msg.target_pose.pose.position.x = x
                    msg.target_pose.pose.position.y = y
                    msg.target_pose.pose.orientation.w = 1

                    # self.client.send_goal(msg)
                    return True

            if client_state == GoalStatus.LOST or client_state == GoalStatus.SUCCEEDED:

                if (
                    self.state == ManagerState.AVOIDING_OBSTACLE
                    or self.state == ManagerState.ROTATING
                ):

                    if client_state == GoalStatus.SUCCEEDED:
                        rospy.loginfo("Successfully recovered from being stuck")
                        self.positions_to_try = None
                        self.state = ManagerState.WAITING_FOR_GOAL

                else:
                    if client_state == GoalStatus.LOST:
                        rospy.loginfo("Robot gave up on reaching occlusion")
                    else:
                        rospy.loginfo("Reached occlusion!")
                        self.visited.append(
                            [jackal_pos.position.x, jackal_pos.position.y]
                        )

                    self.current_goal_occ = None
                    self.current_goal = [float("inf"), float("inf")]
                    # In case it hasn't been deleted already, remove occlusion from buffer.
                    for ind, occ in enumerate(self.occlusion_buffer):
                        if occ.is_goal:
                            self.delete_buffer.append(occ)
                            del self.occlusion_buffer[ind]

                self.state = ManagerState.WAITING_FOR_GOAL

        # Search all occlusions for next goal (except free_space and invalid occlusions)
        occlusions = np.array(
            [
                [occ.mx, occ.my]
                if occ.state != OcclusionState.free_space
                and occ.state != OcclusionState.invalid
                else [float("inf"), float("inf")]
                for occ in self.occlusion_buffer
            ]
        )

        occ_dist = closest_occ = None
        if len(occlusions) > 0:
            occ_dists = cdist(
                occlusions, np.array([[jackal_pos.position.x, jackal_pos.position.y]])
            )
            closest_occ = occ_dists.argmin()

        if closest_occ == None:
            rospy.loginfo("No closest occlusion")

            return
        goal = self.occlusion_buffer[closest_occ]
        if self.current_goal_occ != None and self.current_goal_occ.iden == goal.iden:

            # if self.state == ManagerState.WAITING_FOR_GOAL:
            #     x, y = getGoalCoords(goal, self.occupancy_grid, jackal_pos)

            #     msg = MoveBaseGoal()
            #     msg.target_pose.header.frame_id = "map"
            #     msg.target_pose.header.stamp = rospy.Time.now()
            #     msg.target_pose.pose.position.x = x
            #     msg.target_pose.pose.position.y = y
            #     msg.target_pose.pose.orientation.w = 1.0

            #     if self.client.get_state() == GoalStatus.ACTIVE:
            #         self.client.cancel_goal()

            #     self.state = ManagerState.GOING_TO_GOAL
            #     self.client.send_goal(msg)

            #     return True
            # else:
            self.state = ManagerState.GOING_TO_GOAL
            return False

        self.occlusion_buffer[closest_occ].times_goal += 1

        self.state = ManagerState.GOING_TO_GOAL
        goal.set_goal()

        x, y = getGoalCoords(goal, self.occupancy_grid, jackal_pos)
        rospy.loginfo("Coords are {} {}".format(x, y))

        if x == float("inf") and y == float("inf"):
            rospy.loginfo("Couldn't resolve goal coordinates")
            return False

        self.current_goal = [x, y]
        rospy.loginfo(
            "Setting current goal to ({},{})".format(
                self.current_goal[0], self.current_goal[1]
            )
        )

        if self.current_goal_occ != None:
            self.current_goal_occ.unset_goal()

        self.current_goal_occ = goal

        msg = MoveBaseGoal()
        msg.target_pose.header.frame_id = "map"
        msg.target_pose.header.stamp = rospy.Time.now()
        msg.target_pose.pose.position.x = x
        msg.target_pose.pose.position.y = y
        msg.target_pose.pose.orientation.w = 1.0

        if self.client.get_state() == GoalStatus.ACTIVE:
            self.client.cancel_goal()

        self.client.send_goal(msg)

        return True

    def tsp_set_goal(self, jackal_pos, goal_deleted=False):

        client_state = self.client.get_state()

        if goal_deleted:
            rospy.loginfo("goal has been deleted")
            self.state = ManagerState.WAITING_FOR_GOAL

        if len(self.occlusion_buffer) == 0:
            rospy.loginfo("no goals to navigate towards")
            self.state = ManagerState.WAITING_FOR_GOAL
            return False

        if self.state != ManagerState.WAITING_FOR_GOAL:

            if client_state == GoalStatus.ABORTED:
                rospy.loginfo("Navigation aborted")
                sys.exit()

            if client_state == GoalStatus.LOST or client_state == GoalStatus.SUCCEEDED:

                if client_state == GoalStatus.LOST:
                    rospy.loginfo("Robot gave up on reaching occlusion")

                elif client_state == GoalStatus.SUCCEEDED:
                    rospy.loginfo("Reached occlusion!")
                    self.visited.append([jackal_pos.position.x, jackal_pos.position.y])

                self.current_goal_occ = None
                self.current_goal = [float("inf"), float("inf")]
                # In case it hasn't been deleted already, remove occlusion from buffer.
                for ind, occ in enumerate(self.occlusion_buffer):
                    if occ.is_goal:
                        self.delete_buffer.append(occ)
                        del self.occlusion_buffer[ind]

                self.state = ManagerState.WAITING_FOR_GOAL

        # Search all occlusions for next goal (except free_space and invalid occlusions)
        occlusions = np.array(
            [
                [occ.mx, occ.my]
                if occ.state != OcclusionState.free_space
                and occ.state != OcclusionState.invalid
                else [float("inf"), float("inf")]
                for occ in self.occlusion_buffer
            ]
        )

        np.insert(occlusions, 0, [jackal_pos.position.x, jackal_pos.position.y])
        # occlusions.append([jackal_pos.position.x, jackal_pos.position.y])

        if len(occlusions) == 0:
            rospy.loginfo("No closest occlusion")
            return False

        route = two_opt(occlusions, 0.001)
        idx = route[0]

        goal = self.occlusion_buffer[idx]

        if self.current_goal_occ != None and self.current_goal_occ.iden == goal.iden:
            self.state = ManagerState.GOING_TO_GOAL
            return False

        self.occlusion_buffer[idx].times_goal += 1
        self.state = ManagerState.GOING_TO_GOAL
        goal.set_goal()

        x, y = getGoalCoords(goal, self.occupancy_grid, jackal_pos)
        rospy.loginfo("Coords are {} {}".format(x, y))

        if x == float("inf") and y == float("inf"):
            rospy.loginfo("Couldn't resolve goal coordinates")
            return False

        self.current_goal = [x, y]
        rospy.loginfo(
            "Setting current goal to ({},{})".format(
                self.current_goal[0], self.current_goal[1]
            )
        )

        if self.current_goal_occ != None:
            self.current_goal_occ.unset_goal()

        self.current_goal_occ = goal

        msg = MoveBaseGoal()
        msg.target_pose.header.frame_id = "map"
        msg.target_pose.header.stamp = rospy.Time.now()
        msg.target_pose.pose.position.x = x
        msg.target_pose.pose.position.y = y
        msg.target_pose.pose.orientation.w = 1.0

        if self.client.get_state() == GoalStatus.ACTIVE:
            self.client.cancel_goal()

        self.client.send_goal(msg)

        return True

    def entropyGoal(self, jackal_pos, goal_deleted=False):

        client_state = self.client.get_state()

        if goal_deleted:
            rospy.loginfo("goal has been deleted")
            self.state = ManagerState.WAITING_FOR_GOAL

        if len(self.occlusion_buffer) == 0:
            rospy.loginfo("no goals to navigate towards")
            self.state = ManagerState.WAITING_FOR_GOAL
            return False

        if self.state != ManagerState.WAITING_FOR_GOAL:

            if client_state == GoalStatus.ABORTED:
                rospy.loginfo("Navigation aborted")
                sys.exit()

            if client_state == GoalStatus.LOST or client_state == GoalStatus.SUCCEEDED:

                if client_state == GoalStatus.LOST:
                    rospy.loginfo("Robot gave up on reaching occlusion")

                elif client_state == GoalStatus.SUCCEEDED:
                    rospy.loginfo("Reached occlusion!")
                    self.visited.append([jackal_pos.position.x, jackal_pos.position.y])

                self.current_goal_occ = None
                self.current_goal = [float("inf"), float("inf")]
                # In case it hasn't been deleted already, remove occlusion from buffer.
                for ind, occ in enumerate(self.occlusion_buffer):
                    if occ.is_goal:
                        self.delete_buffer.append(occ)
                        del self.occlusion_buffer[ind]

                self.state = ManagerState.WAITING_FOR_GOAL

        # Search all occlusions for next goal (except free_space and invalid occlusions)
        occlusions = np.array(
            [
                cal_entropy(occ, self.occupancy_grid)
                if occ.state != OcclusionState.free_space
                and occ.state != OcclusionState.invalid
                else [float("inf"), float("inf")]
                for occ in self.occlusion_buffer
            ]
        )

        if len(occlusions) == 0:
            rospy.loginfo("No closest occlusion")
            return False

        idx = np.argmax(occlusions)
        goal = self.occlusion_buffer[idx]

        if self.current_goal_occ != None and self.current_goal_occ.iden == goal.iden:
            self.state = ManagerState.GOING_TO_GOAL
            return False

        self.occlusion_buffer[idx].times_goal += 1
        self.state = ManagerState.GOING_TO_GOAL
        goal.set_goal()

        x, y = getGoalCoords(goal, self.occupancy_grid, jackal_pos)
        rospy.loginfo("Coords are {} {}".format(x, y))

        if x == float("inf") and y == float("inf"):
            rospy.loginfo("Couldn't resolve goal coordinates")
            return False

        self.current_goal = [x, y]
        rospy.loginfo(
            "Setting current goal to ({},{})".format(
                self.current_goal[0], self.current_goal[1]
            )
        )

        if self.current_goal_occ != None:
            self.current_goal_occ.unset_goal()

        self.current_goal_occ = goal

        msg = MoveBaseGoal()
        msg.target_pose.header.frame_id = "map"
        msg.target_pose.header.stamp = rospy.Time.now()
        msg.target_pose.pose.position.x = x
        msg.target_pose.pose.position.y = y
        msg.target_pose.pose.orientation.w = 1.0

        if self.client.get_state() == GoalStatus.ACTIVE:
            self.client.cancel_goal()

        self.client.send_goal(msg)

        return True

    def terminateGoal(self, data):

        backup_dist = 0.2  # meters

        if data.data == True and self.state == ManagerState.AVOIDING_OBSTACLE:
            return

        # Callback from safety policy node
        if data.data == True and self.state != ManagerState.AVOIDING_OBSTACLE:

            self.state = ManagerState.AVOIDING_OBSTACLE

            rospy.loginfo(
                "Too close to obstacle, stopping movement. Manager state is {}".format(
                    self.state
                )
            )

            self.client.cancel_goal()

            if self.current_goal_occ != None:
                self.current_goal_occ.is_goal = False
                self.current_goal_occ.state = OcclusionState.invalid
                self.current_goal_occ.color = OcclusionState.colors[
                    OcclusionState.invalid
                ]
                self.current_goal_occ = None

        elif data.data == False and self.state == ManagerState.AVOIDING_OBSTACLE:
            self.client.cancel_goal()
            rospy.loginfo(
                "No longer near obstacle, changing manager state {}".format(data.data)
            )
            self.state = ManagerState.WAITING_FOR_GOAL

    def set_goal2(self, jackal_pos, goal_deleted=False):

        client_state = self.client.get_state()

        if goal_deleted:
            # rospy.loginfo("goal has been deleted")
            self.state = ManagerState.WAITING_FOR_GOAL

        if len(self.occlusion_buffer) == 0:
            rospy.loginfo("no goals to navigate towards")
            self.state = ManagerState.WAITING_FOR_GOAL
            return False

        if self.state != ManagerState.WAITING_FOR_GOAL:

            if client_state == GoalStatus.ABORTED:
                rospy.loginfo("Navigation aborted")
                
                self.current_goal_occ = None
                self.current_goal = [float("inf"), float("inf")]
                # In case it hasn't been deleted already, remove occlusion from buffer.
                for ind, occ in enumerate(self.occlusion_buffer):
                    if occ.is_goal:
                        self.delete_buffer.append(occ)
                        del self.occlusion_buffer[ind]

            if client_state == GoalStatus.LOST or client_state == GoalStatus.SUCCEEDED:

                if client_state == GoalStatus.LOST:
                    rospy.loginfo("Robot gave up on reaching occlusion")

                elif client_state == GoalStatus.SUCCEEDED:
                    rospy.loginfo("Reached occlusion!")
                    self.visited.append([jackal_pos.position.x, jackal_pos.position.y])

                self.current_goal_occ = None
                self.current_goal = [float("inf"), float("inf")]
                # In case it hasn't been deleted already, remove occlusion from buffer.
                for ind, occ in enumerate(self.occlusion_buffer):
                    if occ.is_goal:
                        self.delete_buffer.append(occ)
                        del self.occlusion_buffer[ind]

                self.state = ManagerState.WAITING_FOR_GOAL

        occlusions = []
        count = [0] * len(OcclusionState.colors)
        for idx, occ in enumerate(self.occlusion_buffer):

            if (
                occ.state != OcclusionState.free_space
                and occ.state != OcclusionState.invalid
            ):
                occlusions.append(self.scoreOcc(occ, jackal_pos, idx))
                count[occ.state] += 1

        # occlusions = np.array(
        #     [
        #         self.scoreOcc(occ, jackal_pos,idx)
        #         if occ.state == OcclusionState.nearby
        #         else float("inf")
        #         for (idx,occ) in enumerate(self.occlusion_buffer)
        #     ]
        # )

        # centroids = np.array(
        #     [
        #         self.scoreOcc(occ, jackal_pos,idx)
        #         if occ.state == OcclusionState.centroid
        #         else float("inf")
        #         for (idx,occ) in enumerate(self.occlusion_buffer)
        #     ]
        # )

        # frontiers = np.array(
        #     [
        #         self.scoreOcc(occ, jackal_pos,idx)
        #         if occ.state == OcclusionState.frontier
        #         else float("inf")
        #         for (idx,occ) in enumerate(self.occlusion_buffer)
        #     ]
        # )

        occlusions = np.array(
            [
                self.scoreOcc(occ, jackal_pos, idx)
                if occ.state != OcclusionState.invalid
                and occ.state != OcclusionState.free_centroid
                else float("inf")
                for (idx, occ) in enumerate(self.occlusion_buffer)
            ]
        )
        idx = np.argmin(occlusions)
        # idx = -1
        # if len(occlusions) == 0:

        #     if len(centroids) == 0:

        #         if len(frontiers) == 0:

        #             rospy.loginfo("No waypoints")
        #             return False

        #         else:

        #             idx = np.argmin(frontiers)

        #     else:

        #         idx = np.argmin(centroids)

        # else:

        #     idx = np.argmin(occlusions)

        goal = self.occlusion_buffer[idx]

        if self.current_goal_occ != None and self.current_goal_occ.iden == goal.iden:
            self.state = ManagerState.GOING_TO_GOAL
            return False

        self.occlusion_buffer[idx].times_goal += 1
        self.state = ManagerState.GOING_TO_GOAL
        goal.set_goal()

        x, y = getGoalCoords(goal, self.occupancy_grid, jackal_pos)
        # rospy.loginfo("Coords are {} {}".format(x,y))

        if x == float("inf") and y == float("inf"):
            rospy.loginfo("Couldn't resolve goal coordinates")
            return False

        self.current_goal = [x, y]
        # rospy.loginfo(
        # "Setting current goal to ({},{})".format(
        # self.current_goal[0], self.current_goal[1]
        # )
        # )

        if self.current_goal_occ != None:
            self.current_goal_occ.unset_goal()

        self.current_goal_occ = goal

        msg = MoveBaseGoal()
        msg.target_pose.header.frame_id = "map"
        msg.target_pose.header.stamp = rospy.Time.now()
        msg.target_pose.pose.position.x = x
        msg.target_pose.pose.position.y = y
        msg.target_pose.pose.orientation.w = 1.0

        if self.client.get_state() == GoalStatus.ACTIVE:
            self.client.cancel_goal()

        self.client.send_goal(msg)

        return True

    def scoreOcc(self, occ, jackal_pos, idx):
        Ts = 1
        Hs = 0.1 #0.35
        Ds = 0.25
        pos = jackal_pos.position

        jackal_orientation = (
            jackal_pos.orientation.x,
            jackal_pos.orientation.y,
            jackal_pos.orientation.z,
            jackal_pos.orientation.w,
        )
        jackal_orient_euler = euler_from_quaternion(jackal_orientation)
        jackal_yaw = jackal_orient_euler[2] + np.pi
        occ_heading = math.atan2(pos.y - occ.my, pos.x - occ.mx)

        err = jackal_yaw - occ_heading
        if err > np.pi:
            err -= 2 * np.pi
        elif err < -np.pi:
            err += 2 * np.pi

        dist = (pos.y - occ.my) ** 2 + (pos.x - occ.mx) ** 2

        t = 1 - 1 / (1 + np.exp(Ts * (occ.time_created - rospy.get_time())))
        # h = 1-1/(1+np.exp(Hs*err))
        h = Hs * math.fabs(err) / np.pi
        d = 1 - 1 / (1 + np.exp(Ds * dist))
        # score = (t + h + d)/3
        score = (h + d) / 2

        # print("Score is ----- " +str(score) + ": Ts -- " + str(t) + ": Hs -- " + str(h) + ": Ds -- " + str(d))
        # print("^^ For ({},{}): err = {} ---- yaw = {} -- occ_head = {}".format(occ.mx, occ.my, err, jackal_yaw, occ_heading))

        # self.occlusion_buffer[idx].color = (score, occ.color[1], occ.color[2], 1)
        return score

    def updateBuffer(self, occlusions, jackal_pos):

        # self.previous_positions.append((jackal_pos.position.x, jackal_pos.position.y))
        self.jackalpos = jackal_pos

        is_goal_valid = True

        # Update occlusion buffer based on some things
        for ind, occ in reversed(list(enumerate(self.occlusion_buffer))):
            # If the robot is inside any buffered occlusion, remove it.
            if (
                cal_dist(jackal_pos.position.x, jackal_pos.position.y, occ.mx, occ.my)
                <= occ.scale * 3 / 4
            ):  # 2*(JACKAL_LENGTH**2 + JACKAL_WIDTH**2):

                if occ.state == OcclusionState.free_space:
                    continue

                # Mark that the goal has just been deleted
                is_goal_valid = not occ.is_goal
                print(
                    cal_dist(
                        jackal_pos.position.x, jackal_pos.position.y, occ.mx, occ.my
                    )
                )
                rospy.loginfo("Occ is too close to jackal")
                self.delete_buffer.append(self.occlusion_buffer[ind])
                del self.occlusion_buffer[ind]

            # If obstacle is too close to occlusion, remove it.
            elif overlapping_obstacle(occ, self.occupancy_grid):

                if occ.state == OcclusionState.free_centroid:
                    continue

                # Mark that the goal has just been deleted
                is_goal_valid = not occ.is_goal
                rospy.loginfo("Occlusion is too close to obstacle")
                self.delete_buffer.append(self.occlusion_buffer[ind])
                del self.occlusion_buffer[ind]

            elif (
                occ.state == OcclusionState.centroid
                and free_space_amount(occ.mx, occ.my, occ.scale, self.occupancy_grid)
                > 0.5
            ):

                is_goal_valid = not occ.is_goal
                self.delete_buffer.append(self.occlusion_buffer[ind])
                del self.occlusion_buffer[ind]

            elif (
                occ.state == OcclusionState.frontier
                and free_space_amount(occ.mx, occ.my, occ.scale, self.occupancy_grid)
                > .70 #0.90
            ):

                is_goal_valid = not occ.is_goal
                self.delete_buffer.append(self.occlusion_buffer[ind])
                del self.occlusion_buffer[ind]

            elif (
                occ.state == OcclusionState.nearby
                and free_space_amount(occ.mx, occ.my, occ.scale, self.occupancy_grid)
                > .60 #0.95
            ):

                rospy.loginfo("Occlusion is in free space")
                is_goal_valid = not occ.is_goal
                self.delete_buffer.append(self.occlusion_buffer[ind])
                del self.occlusion_buffer[ind]

            # If the occlusion has been targeted as a goal too many times, there is likely
            # an oscillation. Remove the occlusion.

            elif occ.times_goal > 5:
                is_goal_valid = not occ.is_goal
                self.delete_buffer.append(self.occlusion_buffer[ind])
                del self.occlusion_buffer[ind]

            elif occ.times_failed > 1:
                is_goal_valid = not occ.is_goal
                self.delete_buffer.append(self.occlusion_buffer[ind])
                del self.occlusion_buffer[ind]

        # Now that occlusion buffer has been updated, it's time to check which new occlusions should be added to the buffer
        is_modified = False
        for occlusion in occlusions:
            x, y = occlusion.mx, occlusion.my

            if (
                cal_dist(jackal_pos.position.x, jackal_pos.position.y, x, y)
                <= occlusion.scale * 3 / 4
            ):

                continue

            # if new occlusion is overlapping an obstacle, ignore it
            if overlapping_obstacle(occlusion, self.occupancy_grid):

                if occlusion.state != OcclusionState.free_centroid:
                    # rospy.loginfo("Occlusion at (%.2f, %.2f) is overlapping obstacle, not adding",x,y)

                    continue

            # if occlusion is the centroid of a shadow and it's in free space, ignore it
            if (
                occlusion.state == OcclusionState.centroid
                and free_space_amount(x, y, occlusion.scale, self.occupancy_grid) > 0.5
            ):
                continue

            if (
                occlusion.state == OcclusionState.frontier
                and free_space_amount(x, y, occlusion.scale, self.occupancy_grid) > .70 #0.9
            ):
                continue

            if (
                occlusion.state == OcclusionState.nearby
                and free_space_amount(x, y, occlusion.scale, self.occupancy_grid) > .60 #0.95
            ):
                continue

            add_flag = True
            for occ in self.occlusion_buffer:
                if cal_dist(occ.mx, occ.my, x, y) < self.epsilon:# * cal_dist(
                #occ.mx, occ.my, jackal_pos.position.x, jackal_pos.position.y
                #):
                    add_flag = False
                    break

            # If occlusion is really close to another occlusion, ignore it. This threshold should be low enough that only noise fluctuations are discarded
            if not add_flag:
                continue

            # if midpoint of new sphere intersects any buffered spheres, remove them and plot this new one
            for ind, data in reversed(list(enumerate(self.occlusion_buffer))):

                if is_intersecting(data, occlusion):

                    # Do not add frontier/centroid if occlusion is already present there
                    if (
                        occlusion.state == OcclusionState.frontier
                        and data.state == OcclusionState.nearby
                    ):
                        add_flag = False
                        break

                    if (
                        occlusion.state == OcclusionState.centroid
                        and data.state == OcclusionState.nearby
                    ):
                        add_flag = False
                        break

                    occlusion.times_goal = data.times_goal
                    if data.is_goal:

                        is_goal_valid = False

                    if data.state == occlusion.state:
                        occlusion.color = data.color

                    self.delete_buffer.append(self.occlusion_buffer[ind])
                    del self.occlusion_buffer[ind]

                elif (
                    data.state == OcclusionState.centroid
                    and occlusion.state == OcclusionState.centroid
                    and cal_dist(data.mx, data.my, x, y) <= 2
                ):

                    occlusion.times_goal = data.times_goal
                    if data.is_goal:
                        is_goal_valid = False

                    self.delete_buffer.append(self.occlusion_buffer[ind])
                    del self.occlusion_buffer[ind]

            if add_flag:
                is_modified = True
                occlusion.iden = self.counter
                self.occlusion_buffer.append(occlusion)

                if len(self.occlusion_buffer) > self.buffer_len:
                    rospy.loginfo(
                        "deleting "
                        + str(self.occlusion_buffer[0])
                        + " because buffer is full"
                    )
                    is_goal_valid = self.occlusion_buffer[0].is_goal
                    self.delete_buffer.append(self.occlusion_buffer[0])
                    del self.occlusion_buffer[0]

                self.counter += 1
                self.is_updated = True

        for ind, occ in enumerate(self.occlusion_buffer):
            self.scoreOcc(occ, jackal_pos, ind)

        if self.autonomous:
            # self.entropyGoal(jackal_pos, not is_goal_valid)
            # self.set_goal(jackal_pos, not is_goal_valid)
            self.set_goal2(jackal_pos, not is_goal_valid)
            # self.tsp_set_goal(jackal_pos, not is_goal_valid)

    def publishOcclusions(self, debug=False):

        if not self.is_updated and len(self.delete_buffer) == 0:
            return

        marker_arr = MarkerArray()
        marker_arr.markers = []

        for ind, occ in list(enumerate(self.delete_buffer)):
            marker_arr.markers.append(self.generateDeleteMsg(occ))

        if debug:
            msg = self.generatePublishMsg(Marker.DELETEALL, namespace="point")
            marker_arr.markers.append(msg)

        for ind, occ in list(enumerate(self.occlusion_buffer)):

            if occ.is_valid:
                marker_arr.markers.append(self.generatePublishMsg(Marker.ADD, occ))

            if debug and occ.state == OcclusionState.nearby:
                msg1, msg2 = self.plotOcclusionPoints(occ)
                marker_arr.markers.append(msg1)
                marker_arr.markers.append(msg2)

        self.publisher.publish(marker_arr)
        self.is_updated = False

        self.delete_buffer = []

        # rospy.loginfo([str(i) for i in self.occlusion_buffer])

    def publishShadows(self, line_list):
        if not self.is_updated:
            return

        marker_arr = MarkerArray()
        marker_arr.markers = [line_list]

        self.publisher.publish(marker_arr)


    def clear_occlusions(self):

        marker_arr = MarkerArray()
        for occ in self.occlusion_buffer:
            if occ.state == OcclusionState.frontier:
                marker_arr.markers.append(self.generateDeleteMsg(occ))

        self.publisher.publish(marker_arr)


    def generateDeleteMsg(self, occlusion):
        mx = occlusion.mx
        my = occlusion.my

        msg = Marker()
        msg.header.frame_id = "map"
        msg.header.stamp = rospy.Time.now()
        msg.ns = "occlusion"
        msg.id = occlusion.iden
        msg.action = Marker.DELETE

        return msg

    def plotOcclusionPoints(self, occlusion):
        p1 = Occlusion(
            occlusion.pointA[0],
            occlusion.pointA[1],
            (1, 0, 0, 1),
            2 * occlusion.iden,
            occlusion.shape,
            0.25,
            None,
            None,
            OcclusionState.nearby,
            rospy.get_time(),
        )
        p2 = Occlusion(
            occlusion.pointB[0],
            occlusion.pointB[1],
            (1, 0, 0, 1),
            2 * occlusion.iden + 1,
            occlusion.shape,
            0.25,
            None,
            None,
            OcclusionState.nearby,
            rospy.get_time(),
        )

        msg1 = self.generatePublishMsg(Marker.ADD, p1, "point")
        msg2 = self.generatePublishMsg(Marker.ADD, p2, "point")

        return (msg1, msg2)

    def generatePublishMsg(self, action, occlusion=None, namespace=None):

        mx = my = 0
        if occlusion != None:
            mx, my = occlusion.mx, occlusion.my

        msg = Marker()
        msg.header.frame_id = "map"
        msg.header.stamp = rospy.Time.now()
        if namespace == None:
            msg.ns = "occlusion"
        else:
            msg.ns = namespace
        # when counter > buffer_len, we are rewritting old spheres
        if occlusion != None:
            msg.id = occlusion.iden
        msg.action = action

        if action == Marker.ADD:

            msg.type = occlusion.shape
            msg.pose.position.x = mx
            msg.pose.position.y = my
            msg.pose.position.z = 1
            msg.pose.orientation.x = 0
            msg.pose.orientation.y = 0
            msg.pose.orientation.z = 0
            msg.pose.orientation.w = 1

            msg.lifetime = rospy.Duration()

            # Set scale of sphere to some percentage of distance between the two identified points
            msg.scale.x = msg.scale.y = msg.scale.z = occlusion.scale

            msg.color.r, msg.color.g, msg.color.b, msg.color.a = occlusion.color

        return msg

    # Intersection logic courtesy of https://codereview.stackexchange.com/questions/86421/line-segment-to-circle-collision-algorithm
    def checkIntersections(self, regions, angles, curr_pos, blacklist):
        jackal_orientation = (
            jackal_pos.orientation.x,
            jackal_pos.orientation.y,
            jackal_pos.orientation.z,
            jackal_pos.orientation.w,
        )
        jackal_orient_euler = euler_from_quaternion(jackal_orientation)
        jackal_yaw = jackal_orient_euler[2]

        pri_counts = [0] * len(self.occlusion_buffer)

        # not looping occ buffer first because then we would need to recalculate p2 for each range, really inefficient and would require
        # more space to keep a map of those values

        for ind, (dist, angle) in enumerate(zip(regions, angles)):
            if blacklist[ind] == 1:
                continue

            if dist == float("inf"):
                dist = MAX_RANGE

            rot = angle + jackal_yaw
            p1 = (curr_pos.position.x, curr_pos.position.y)
            p2 = (
                dist * math.cos(rot) + curr_pos.position.x,
                dist * math.sin(rot) + curr_pos.position.y,
            )

            P = (p2[0] - p1[0], p2[1] - p1[1])

            for i, occ in enumerate(self.occlusion_buffer):

                if (
                    1.5
                    * (
                        (curr_pos.position.x - occ.mx) ** 2
                        + (curr_pos.position.y - occ.my) ** 2
                    )
                    > dist
                ):
                    continue

                # occlusion is already at max priority, so this update is useless for it
                if occ.priority == occ.max_priority:
                    continue

                # calculate quadratic equation coefficients
                a = dist
                vecP1_C = (p1[0] - occ.mx, p1[1] - occ.my)
                b = 2 * (P[0] * vecP1_C[0] + P[1] * vecP1_C[1])
                c = (
                    (p1[0] * p1[0] + p1[1] * p1[1])
                    + (occ.mx * occ.mx + occ.my * occ.my)
                    - 2 * (p1[0] * occ.mx + p1[1] * occ.my)
                    - (occ.scale / 2) ** 2
                )

                discrim = b ** 2 - 4 * a * c
                if discrim < 0:
                    continue

                rt_discirm = math.sqrt(discrim)
                t1 = (-b + rt_discirm) / (2 * a)
                t2 = (-b - rt_discirm) / (2 * a)

                # If the solutions are less than 1, that means the segment is not intersecting the sphere, but
                # would intersect if it were longer
                # if not (0 <= t1 <= 1 or 0 <= t2 <= 1):
                if (t1 > 1 or t1 < 0) and (t2 > 1 or t2 < 0):
                    continue

                pri_counts[i] += 1
                # prevent a decrease in priority due to the robot moving in such a way to reduce line segment intersections
                if pri_counts[i] > occ.priority:
                    occ.updatePriority()


def cal_dist(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


# Check if two spheres (occlusions) intersect one another
def is_intersecting(occ1, occ2):
    # if occ1.state != Occ
    return cal_dist(occ1.mx, occ1.my, occ2.mx, occ2.my) <= (occ1.scale + occ2.scale) / 2


def overlapping_obstacle(occlusion, occupancy_grid):
    mx, my = occlusion.mx, occlusion.my
    scale = occlusion.scale

    if occupancy_grid is None:
        return

    # if occlusion is outside occupancy grid don't bother checking
    OBSTACLE_PROB_THRESH = 80

    # Get grid coordinates of occlusion
    origin = occupancy_grid.info.origin
    resolution = occupancy_grid.info.resolution
    w, h = occupancy_grid.info.width, occupancy_grid.info.height

    # wi,hi = (mx-jackal_width - origin.position.x)/resolution, (my-jackal_length - origin.position.y)/resolution
    # wf,hf = (mx+jackal_width - origin.position.x)/resolution, (my+jackal_length - origin.position.y)/resolution
    wi, hi = (mx - scale / 2 - origin.position.x) / resolution, (
        my - scale / 2 - origin.position.y
    ) / resolution
    wf, hf = (mx + scale / 2 - origin.position.x) / resolution, (
        my + scale / 2 - origin.position.y
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

    if wf >= w or wi <= 0:
        return False

    if hf >= h or hi <= 0:
        return False

    for i in range(wi, wf):
        if i >= w:
            break
        for j in range(hi, hf):
            if j >= h:
                break
            if i + j * w >= w * h:
                break

            # If chance of obstacle being at point is too high, we are overlapping
            # the obstacle
            if occupancy_grid.data[i + j * w] > OBSTACLE_PROB_THRESH:
                return True

    return False


def free_space_amount(mx, my, scale, occupancy_grid):

    if occupancy_grid is None:
        return

    origin = occupancy_grid.info.origin
    resolution = occupancy_grid.info.resolution
    w, h = occupancy_grid.info.width, occupancy_grid.info.height

    radius = scale / 2

    wi, hi = int((mx - radius - origin.position.x) / resolution), int(
        (my - radius - origin.position.y) / resolution
    )
    wf, hf = int((mx + radius - origin.position.x) / resolution), int(
        (my + radius - origin.position.y) / resolution
    )
    wi, hi = max(0, wi), max(0, hi)
    wf, hf = min(w, wf), min(h, hf)

    if wf >= w or wi <= 0:
        return False

    if hf >= h or hi <= 0:
        return False

    area = 0.0
    count = 0
    for i in range(wi, wf):
        if i >= w:
            break
        for j in range(hi, hf):
            if j >= h:
                break

            # idx = i + j*w
            idx = i + j * w

            # if radius**2 < i**2 + j**2:
            #     continue
            if idx >= w * h:
                break

            if occupancy_grid.data[idx] < 0:
                count += 1
                continue

            count += 1
            # area += (1-occupancy_grid.data[i+j*w])*resolution**2
            area += (100 - occupancy_grid.data[i + j * w]) / 100

    # return area/(math.pi*radius**2) > .8
    if count == 0:
        return 0

    if wf == wi or hf == hi:
        return 1.0

    return area / count  # ((wf - wi) * (hf - hi))


def getGoalCoords(occ, occupancy_grid, jackal_pos):

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

    P = np.array([occ.mx, occ.my])

    if P[0] >= A[0] and P[0] <= B[0] and P[1] >= D[1] and P[1] <= A[1]:
        # rospy.loginfo("Goal is inside occupancy grid")
        return (occ.mx, occ.my)
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


# Calculate the euclidian distance in n-space of the route r traversing cities c, ending at the path start.
path_distance = lambda r, c: np.sum(
    [
        np.linalg.norm([c[r[p]][0] - c[r[p - 1]][0], c[r[p]][1] - c[r[p - 1]][1]])
        for p in range(len(r))
    ]
)
# Reverse the order of all elements from element i to element k in array r.
two_opt_swap = lambda r, i, k: np.concatenate(
    (r[0:i], r[k : -len(r) + i - 1 : -1], r[k + 1 : len(r)])
)


def two_opt(
    cities, improvement_threshold
):  # 2-opt Algorithm adapted from https://en.wikipedia.org/wiki/2-opt
    route = np.arange(
        len(cities)
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
            1, len(route) - 1
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


def cal_entropy(occlusion, occupancy_grid):

    if occupancy_grid is None:
        return

    if occlusion.mx == float("inf") or occlusion.my == float("inf"):
        return

    origin = occupancy_grid.info.origin
    resolution = occupancy_grid.info.resolution
    w, h = occupancy_grid.info.width, occupancy_grid.info.height

    radius = occlusion.scale / 2

    wi, hi = int((occlusion.mx - radius - origin.position.x) / resolution), int(
        (occlusion.my - radius - origin.position.y) / resolution
    )
    wf, hf = int((occlusion.mx + radius - origin.position.x) / resolution), int(
        (occlusion.my + radius - origin.position.y) / resolution
    )
    wi, hi = max(0, wi), max(0, hi)
    wf, hf = min(w, wf), min(h, hf)

    if wf >= w or wi <= 0:
        return False

    if hf >= h or hi <= 0:
        return False

    grid = np.reshape(np.array(occupancy_grid.data), [w, h])
    grid = grid[wi:wf, hi:hf]
    grid[grid < 0] = 50
    grid = grid / 100.0

    entropy = 0

    try:
        entropy = -1 * np.sum(
            grid * np.log2(grid + 0.001) + (1 - grid) * np.log2(1 - grid + 0.001)
        )
    except:
        entropy = -1

    if math.isnan(entropy):
        np.set_printoptions(threshold=sys.maxsize)
        print(grid)
        print(entropy)
        sys.exit()

    return entropy
