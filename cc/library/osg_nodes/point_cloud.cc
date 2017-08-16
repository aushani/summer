// Adpated from dascar
#include "library/osg_nodes/point_cloud.h"

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
}

} // osg_nodes
} // library

