#pragma once

#include <boost/assert.hpp>

#include <vector>
#include <cstring>

#include <Eigen/Core>

namespace library {
namespace kitti {

struct ObjectLabel;

typedef std::vector<ObjectLabel> ObjectLabels;

struct ObjectLabel {
  enum Type {
    CAR,
    VAN,
    TRUCK,
    PEDESTRIAN,
    PERSON_SITTING,
    CYCLIST,
    TRAM,
    MISC,
    DONT_CARE
  };

  static ObjectLabels Load(const char *fn);
  static void Save(const ObjectLabels &labels, const char *fn);

  static Eigen::Matrix4d LoadVelToCam(const char *fn);

  Type type = DONT_CARE;            // Describes the type of object
  float truncated = 0;              // float from 0 (non-truncated) to 1 (truncated), how much of object left image boundaries
  float occluded = 3;               // 0 = fully visible, 1 = partly occluded, 2 = largely occluded, 3 = unknown
  float alpha = 0;                  // Observation angle of object, -pi to pi
  float bbox[4] = {0, 0, 0, 0};     // 0-based index of left, top, right, bottom pixel coordinates of bounding box in image plane
  float dimensions[3] = {0, 0, 0};  // 3d height width length in meter
  float location[3] = {0, 0, 0};    // 3d location x y z in camera coordinates in meters
  float rotation_y = 0;             // rotation around y-axis in camera coordinates, -pi to pi
  float score = 0;                  // for results only, float indicated confidence (higher is better)

 private:
  static Type GetType(const char *type);
  static const char* GetString(const Type &type);
};

} // namespace kitti
} // namespace library
