#pragma once

#include <osg/Geometry>
#include <osg/Drawable>
#include <osg/MatrixTransform>
#include <osg/Texture2D>
#include <osg/Geode>

#include "library/detector/detector.h"

namespace dt = library::detector;

namespace app {
namespace kitti_occ_grids {

class MapNode : public osg::Group {
 public:
  MapNode(const dt::Detector &detector);

 private:
  osg::ref_ptr<osg::Image> GetImage(const dt::Detector &detector) const;

  void SetUpTexture(osg::Texture2D *texture, osg::Geode *geode, double x0, double y0, double width, double height, int bin_num) const;
};

} // namespace kitti_occ_grids
} // namespace app
