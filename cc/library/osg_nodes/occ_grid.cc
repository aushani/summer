// Adapted from dascar
#include "library/osg_nodes/occ_grid.h"

#include "library/osg_nodes/colorful_box.h"

namespace library {
namespace osg_nodes {

OccGrid::OccGrid(const rt::OccGrid &og, double thresh_lo) : osg::Group() {
  double scale = og.GetResolution() * 0.75;

  // Iterate over occ grid and add occupied cells
  for (size_t i = 0; i < og.GetLocations().size(); i++) {
    rt::Location loc = og.GetLocations()[i];
    float val = og.GetLogOdds()[i];

    if (val <= thresh_lo) {
      continue;
    }

    double x = loc.i * og.GetResolution();
    double y = loc.j * og.GetResolution();
    double z = loc.k * og.GetResolution();

    double alpha = val*2;
    if (alpha < 0) {
      alpha = 0;
    }

    if (alpha > 0.8) {
      alpha = 0.8;
    }

    osg::Vec4 color(0.1, 0.9, 0.1, alpha);
    osg::Vec3 pos(x, y, z);

    osg::ref_ptr<ColorfulBox> box = new ColorfulBox(color, pos, scale);
    addChild(box);
  }
}

OccGrid::OccGrid(const rt::FeatureOccGrid &fog, double thresh_lo) : osg::Group() {
  double scale = fog.GetResolution() * 0.75;

  // Iterate over occ grid and add occupied cells
  for (size_t i = 0; i < fog.GetLocations().size(); i++) {
    const rt::Location &loc = fog.GetLocations()[i];
    float val = fog.GetLogOdds()[i];

    if (val <= thresh_lo) {
      continue;
    }

    double x = loc.i * fog.GetResolution();
    double y = loc.j * fog.GetResolution();
    double z = loc.k * fog.GetResolution();

    double alpha = val*2;
    if (alpha < 0) {
      alpha = 0;
    }

    if (alpha > 0.8) {
      alpha = 0.8;
    }

    double g = 0.9;
    if (fog.HasStats(loc)) {
      const auto &stats = fog.GetStats(loc);
      printf("intensity: %5.3f, %d\n", stats.intensity, stats.count);
      float intens = stats.intensity;
      if (intens < 0) {
        intens = 0.0;
      }

      if (intens > 1.0) {
        intens = 1.0;
      }

      g = intens;
    }

    osg::Vec4 color(0.1, g, 0.1, alpha);
    osg::Vec3 pos(x, y, z);

    osg::ref_ptr<ColorfulBox> box = new ColorfulBox(color, pos, scale);
    addChild(box);
  }
}

}  // namespace osg_nodes
}  // namespace library
