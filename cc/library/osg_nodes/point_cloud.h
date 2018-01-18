// Adapted from dascar
#pragma once

#include <osg/Geometry>
#include <osg/Drawable>
#include <osg/MatrixTransform>
#include <Eigen/Core>

#include "library/kitti/velodyne_scan.h"

namespace kt = library::kitti;

namespace library {
namespace osg_nodes {

class PointCloud : public osg::Geometry {
 public:
  PointCloud(const kt::VelodyneScan &scan);
  PointCloud(const std::vector<Eigen::Vector3d> &hits);

 private:
  static constexpr double kColorMapZMin = -2.5;
  static constexpr double kColorMapZMax = 2.5;

  osg::ref_ptr<osg::Vec3Array> vertices_;
  osg::ref_ptr<osg::Vec4Array> colors_;
  osg::ref_ptr<osg::DrawArrays> draw_arrays_;

  //osg::ColorMap::Type cmap_ = osg::ColorMap::Type::JET;
};

} // osg_nodes
} // library
