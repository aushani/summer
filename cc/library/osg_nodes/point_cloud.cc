// Adpated from dascar
#include "library/osg_nodes/point_cloud.h"

#include <random>

#include <osg/Point>

namespace kt = library::kitti;

namespace library {
namespace osg_nodes {

PointCloud::PointCloud(const kt::VelodyneScan &scan) :
 osg::Geometry(),
 vertices_(new osg::Vec3Array),
 colors_(new osg::Vec4Array) {

  for (const auto &hit : scan.GetHits()) {
    vertices_->push_back(osg::Vec3(hit.x(), hit.y(), hit.z()));
    double z = hit.z();
    double c = 0;
    if (z < kColorMapZMin) {
      c = 0.0;
    } else if (z > kColorMapZMax) {
      c = 1.0;
    } else {
      c = (z - kColorMapZMin)/(kColorMapZMax - kColorMapZMin);
    }

    colors_->push_back(osg::Vec4(1-c, 0, c, 0));
  }

  setVertexArray(vertices_);
  setColorArray(colors_);
  setColorBinding(osg::Geometry::BIND_PER_VERTEX);

  draw_arrays_ = new osg::DrawArrays(osg::PrimitiveSet::POINTS, 0, vertices_->size());
  addPrimitiveSet(draw_arrays_);

  //_geode->addDrawable(this);
  osg::ref_ptr<osg::StateSet> state = getOrCreateStateSet();
  state->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
  state->setAttribute(new osg::Point(3), osg::StateAttribute::ON);
}

PointCloud::PointCloud(const pcl::PointCloud<pcl::PointXYZRGB> &cloud) :
 osg::Geometry(),
 vertices_(new osg::Vec3Array),
 colors_(new osg::Vec4Array) {
  for (const auto &point : cloud) {
    vertices_->push_back(osg::Vec3(point.x, point.y, point.z));
    colors_->push_back(osg::Vec4(point.r/255.0, point.g/255.0, point.b/255.0, 0));
  }

  setVertexArray(vertices_);
  setColorArray(colors_);
  setColorBinding(osg::Geometry::BIND_PER_VERTEX);

  draw_arrays_ = new osg::DrawArrays(osg::PrimitiveSet::POINTS, 0, vertices_->size());
  addPrimitiveSet(draw_arrays_);

  //_geode->addDrawable(this);
  osg::ref_ptr<osg::StateSet> state = getOrCreateStateSet();
  state->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
  state->setAttribute(new osg::Point(1), osg::StateAttribute::ON);
}

PointCloud::PointCloud(const pcl::PointCloud<pcl::PointXYZRGB> &cloud, const Sophus::SE3d &t, float decimate) :
 osg::Geometry(),
 vertices_(new osg::Vec3Array),
 colors_(new osg::Vec4Array) {
  for (const auto &point : cloud) {
    double f = (double) rand() / RAND_MAX;
    if (f > decimate) {
      continue;
    }
    Eigen::Vector3d p(point.x, point.y, point.z);
    Eigen::Vector3d p_w = t*p;

    vertices_->push_back(osg::Vec3(p_w.x(), p_w.y(), p_w.z()));
    colors_->push_back(osg::Vec4(point.r/255.0, point.g/255.0, point.b/255.0, 0));
  }

  setVertexArray(vertices_);
  setColorArray(colors_);
  setColorBinding(osg::Geometry::BIND_PER_VERTEX);

  draw_arrays_ = new osg::DrawArrays(osg::PrimitiveSet::POINTS, 0, vertices_->size());
  addPrimitiveSet(draw_arrays_);

  //_geode->addDrawable(this);
  osg::ref_ptr<osg::StateSet> state = getOrCreateStateSet();
  state->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
  state->setAttribute(new osg::Point(1), osg::StateAttribute::ON);
}

} // osg_nodes
} // library
