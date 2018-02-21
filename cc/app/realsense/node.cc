#include "app/realsense/node.h"

#include <osg/ShapeDrawable>

namespace app {
namespace realsense {

Node::Node(const pcl::PointCloud<pcl::PointXYZRGB> &cloud, const Sophus::SE3d &se3) :
 cloud_(cloud),
 se3_(se3) {

}

void Node::SetPose(const Sophus::SE3d &se3) {
  se3_ = se3;
}

osg::ref_ptr<osg::Node> Node::GetOsg() const {
  osg::ref_ptr<osg::MatrixTransform> res = new osg::MatrixTransform();

  osg::ref_ptr<osgn::PointCloud> pc(new osgn::PointCloud(cloud_));
  res->addChild(pc);

  osg::Vec3 pos(0, 0, 0);
  osg::ref_ptr<osg::Box> box = new osg::Box(pos, 0.1, 0.1, 0.1);
  osg::ref_ptr<osg::ShapeDrawable> shape = new osg::ShapeDrawable(box);

  shape->setColor(osg::Vec4(1, 0, 0, 0));

  res->addChild(shape);

  auto mat = se3_.matrix().transpose();
  //auto trans = se3_.translation();
  //auto quat = se3_.quaternion();

  //std::cout << mat << std::endl;

  osg::Matrixd H(mat(0, 0), mat(0, 1), mat(0, 2), mat(0, 3),
                 mat(1, 0), mat(1, 1), mat(1, 2), mat(1, 3),
                 mat(2, 0), mat(2, 1), mat(2, 2), mat(2, 3),
                 mat(3, 0), mat(3, 1), mat(3, 2), mat(3, 3));
  res->setMatrix(H);

  return res;
}

} // namespace realsense
} // namespace app
