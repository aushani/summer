#include "library/osg_nodes/occ_model.h"

#include "library/osg_nodes/colorful_box.h"

namespace clt = library::chow_liu_tree;

namespace library {
namespace osg_nodes {

OccModel::OccModel(const clt::JointModel &jm) : osg::Group() {
  double scale = jm.GetResolution() * 0.75;

  // Get all nodes
  int min_ij = - (jm.GetNXY() / 2);
  int max_ij = min_ij + jm.GetNXY();

  int min_k = - (jm.GetNZ() / 2);
  int max_k = min_k + jm.GetNZ();

  for (int i=min_ij; i < max_ij; i++) {
    for (int j=min_ij; j < max_ij; j++) {
      for (int k=min_k; k < max_k; k++) {
        rt::Location loc(i, j, k);

        if (jm.GetNumObservations(loc) < 10) {
          continue;
        }

        int c_t = jm.GetCount(loc, true);
        int c_f = jm.GetCount(loc, false);

        double p_occ = c_t / (static_cast<double>(c_t + c_f));

        if (p_occ < 0.10) {
          continue;
        }

        double x = loc.i * jm.GetResolution();
        double y = loc.j * jm.GetResolution();
        double z = loc.k * jm.GetResolution();

        double alpha = p_occ;

        osg::Vec4 color(0.1, 0.9, 0.1, alpha);
        osg::Vec3 pos(x, y, z);

        osg::ref_ptr<ColorfulBox> box = new ColorfulBox(color, pos, scale);
        addChild(box);
      }
    }
  }
}

OccModel::OccModel(const clt::MarginalModel &mm) : osg::Group() {
  double scale = mm.GetResolution() * 0.75;

  // Get all nodes
  int min_ij = - (mm.GetNXY() / 2);
  int max_ij = min_ij + mm.GetNXY();

  int min_k = - (mm.GetNZ() / 2);
  int max_k = min_k + mm.GetNZ();

  for (int i=min_ij; i < max_ij; i++) {
    for (int j=min_ij; j < max_ij; j++) {
      for (int k=min_k; k < max_k; k++) {
        rt::Location loc(i, j, k);

        if (mm.GetNumObservations(loc) < 10) {
          continue;
        }

        int c_t = mm.GetCount(loc, true);
        int c_f = mm.GetCount(loc, false);

        double p_occ = c_t / (static_cast<double>(c_t + c_f));

        if (p_occ < 0.20) {
          continue;
        }

        double x = loc.i * mm.GetResolution();
        double y = loc.j * mm.GetResolution();
        double z = loc.k * mm.GetResolution();

        double alpha = p_occ;

        double r = (z + 2) / 1.25;
        if (r < 0) r = 0;
        if (r > 1) r = 1;

        double b = 1 - r;

        osg::Vec4 color(r, 0.9, b, alpha);
        osg::Vec3 pos(x, y, z);

        osg::ref_ptr<ColorfulBox> box = new ColorfulBox(color, pos, scale);
        addChild(box);
      }
    }
  }
}

}  // namespace osg_nodes
}  // namespace library
