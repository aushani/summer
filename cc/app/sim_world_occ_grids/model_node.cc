#include "app/sim_world_occ_grids/model_node.h"

#include "library/osg_nodes/colorful_box.h"

namespace osgn = library::osg_nodes;

namespace app {
namespace sim_world_occ_grids {

ModelNode::ModelNode(const JointModel &jm) : osg::Group() {
  int min_ij = -jm.GetNXY() / 2;
  int max_ij = min_ij + jm.GetNXY();

  int min_k = -jm.GetNZ() / 2;
  int max_k = min_k + jm.GetNZ();

  double res = jm.GetResolution();

  for (int i1=min_ij; i1 < max_ij; i1++) {
    for (int j1=min_ij; j1 < max_ij; j1++) {
      for (int k1=min_k; k1 < max_k; k1++) {
        rt::Location loc(i1, j1, k1);

        double x = loc.i * res;
        double y = loc.j * res;
        double z = loc.k * res;

        double c_t = jm.GetCount(loc, true);
        double c_f = jm.GetCount(loc, false);

        double p = c_t / (c_t + c_f);
        if (p < 0.1) {
          continue;
        }

        double alpha = p;
        if (alpha < 0) {
          alpha = 0;
        }

        if (alpha > 1) {
          alpha = 1;
        }

        osg::Vec4 color(0.1, 0.9, 0.1, alpha);
        osg::Vec3 pos(x, y, z);

        osg::ref_ptr<osgn::ColorfulBox> box = new osgn::ColorfulBox(color, pos, 0.9 * res);
        addChild(box);
      }
    }
  }
}

ModelNode::ModelNode(const Model &model) : osg::Group() {
  auto counts = model.GetCounts();

  double res = model.GetResolution();

  for (auto it = counts.cbegin(); it != counts.cend(); it++) {
    auto loc = it->first;
    auto counter = it->second;

    double x = loc.i * res;
    double y = loc.j * res;
    double z = loc.k * res;

    double p = counter.GetProbability(true); // Get probability that it is occupied
    if (p < 0.1) {
      continue;
    }

    double alpha = p;
    if (alpha < 0) {
      alpha = 0;
    }

    if (alpha > 1) {
      alpha = 1;
    }

    osg::Vec4 color(0.1, 0.9, 0.1, alpha);
    osg::Vec3 pos(x, y, z);

    osg::ref_ptr<osgn::ColorfulBox> box = new osgn::ColorfulBox(color, pos, 0.9 * res);
    addChild(box);
  }
}

} // namespace sim_world_occ_grids
} // namespace app
