#pragma once

#include <pcl/point_types.h>
#include <pcl/common/projection_matrix.h>
#include <pcl/io/pcd_io.h>

#include <sophus/se3.hpp>

#include "library/osg_nodes/point_cloud.h"

namespace osgn = library::osg_nodes;

namespace app {
namespace realsense {

class Node {
 public:
  Node(const pcl::PointCloud<pcl::PointXYZRGB> &cloud, const Sophus::SE3d &se3);

  void SetPose(const Sophus::SE3d &se3);

  osgn::PointCloud GetPointCloud() const;

 private:
  pcl::PointCloud<pcl::PointXYZRGB> cloud_;
  Sophus::SE3d se3_;
};

} // namespace realsense
} // namespace app
