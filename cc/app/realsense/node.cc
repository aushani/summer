#include "app/realsense/node.h"

namespace app {
namespace realsense {

Node::Node(const pcl::PointCloud<pcl::PointXYZRGB> &cloud, const Sophus::SE3d &se3) :
 cloud_(cloud),
 se3_(se3) {

}

void Node::SetPose(const Sophus::SE3d &se3) {
  se3_ = se3;
}

osgn::PointCloud Node::GetPointCloud() const {
  return osgn::PointCloud(cloud_, se3_, 0.1);
}

} // namespace realsense
} // namespace app
