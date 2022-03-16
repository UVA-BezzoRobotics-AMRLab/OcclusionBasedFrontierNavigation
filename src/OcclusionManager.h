#include <actionlib/client/simple_action_client.h>
#include <actionlib_msgs/GoalStatus.h>
#include <boost/tuple/tuple.hpp>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/TransformStamped.h>
#include <iostream>
#include <math.h>
#include <move_base_msgs/MoveBaseAction.h>
#include <move_base_msgs/MoveBaseGoal.h>
#include <nav_msgs/MapMetaData.h>
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/Odometry.h>
#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>
#include <string>
#include <tf2/utils.h>
#include <vector>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

typedef unsigned int uint_t;
typedef boost::tuple<float, float, float, float> color_t;

struct OcclusionState {
  enum State {
    def,
    nearby,
    horizon,
    goal,
    freeSpace,
    centroid,
    freeCentroid,
    invalid
  };

  color_t colors[8] = {
          [0] = color_t(0, 0, 0, 0),  [1] = color_t(0, 0, 1, .5),
          [2] = color_t(0, 1, 0, .5), [3] = color_t(1, 1, 0, .5),
          [4] = color_t(1, 0, 1, .5), [5] = color_t(0, 1, 1, .5),
          [6] = color_t(1, 0, 0, .5), [7] = color_t(1, 0, 0, 1),
  };
};
typedef struct OcclusionState OcclusionState_t;

struct ManagerState {
  enum State {
    waitingForGoal,
    goingToGoal,
    moveToDefaultGoal,
    avoidingObstacle,
    rotating
  };
};
typedef struct ManagerState ManagerState_t;

class Occlusion {
public:
  Occlusion(float mx, float my, color_t color, uint_t id, uint_t shape,
            float scale, boost::tuple<float, float> pointA,
            boost::tuple<float, float> pointB, uint_t state);
  ~Occlusion();

  void setGoal();
  void unsetGoal();
  boost::tuple<float, float> getCoords();
  void updateCounter();
  void updatePriority();

  friend std::ostream &operator<<(std::ostream &strm, const Occlusion &occ);

  float mx, my, scale;
  color_t color;
  uint_t id, shape, counter, threshold, priority, maxPriority, timesGoal, timesFailed;
  boost::tuple<float, float> pointA, pointB;

  bool isValid, isGoal;
  OcclusionState_t state, initialState;

private:

};

class OcclusionManager {
public:
  OcclusionManager();
  ~OcclusionManager();

  void updateMap(const nav_msgs::OccupancyGrid &msg);
  void updateLidar(const sensor_msgs::LaserScan::ConstPtr &msg);

  bool setGoal(const geometry_msgs::Pose &jackalPos, bool goalDeleted = false);
  bool updateBuffer(const std::vector<Occlusion> &occlusions,
                    const geometry_msgs::Pose &jackalPos);
  void publishOcclusions(bool debug = false);
  void publishLines(const visualization_msgs::Marker &line_list);
  void generateDeleteMsg(const std::vector<Occlusion> &occlusions);
  void plotOcclusionPoints(const std::vector<Occlusion> &occlusions);
  void generatePublishMsg(const int &action, boost::optional<Occlusion &> occ =
                                                 boost::optional<Occlusion &>(),
                          const std::string& ns="");

private:
};

void isIntersecting(const Occlusion &occ1, const Occlusion &occ2);
bool overlappingObstacle(const Occlusion &occlusion,
                         const nav_msgs::OccupancyGrid &occupancyGrid);
bool isInFreeSpace(const float &mx, const float &my, const float &scale,
                   const nav_msgs::OccupancyGrid &occupancyGrid);
boost::tuple<float, float>
getGoalCoords(const Occlusion &occ,
              const nav_msgs::OccupancyGrid &occupancyGrid,
              const geometry_msgs::Pose &jackalPos);
// bool isSegIntersecting()
