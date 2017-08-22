#include "app/kitti_occ_grids/chow_lui_tree_node.h"

#include <algorithm>

#include <osg/Geometry>
#include <osg/LineWidth>

namespace app {
namespace kitti_occ_grids {

ChowLuiTreeNode::ChowLuiTreeNode(const ChowLuiTree &clt, const JointModel &jm) : osg::Group() {
  const auto edges = clt.GetEdges();
  double res = clt.GetResolution();

  double min_mi = 0;
  double max_mi = log(2);

  for (const auto &e : edges) {
    const auto &loc1 = e.loc1;
    const auto &loc2 = e.loc2;
    double mi = e.weight;

    double c_t1 = jm.GetCount(loc1, true);
    double c_f1 = jm.GetCount(loc1, false);
    double p1 = c_t1 / (c_t1 + c_f1);

    double c_t2 = jm.GetCount(loc2, true);
    double c_f2 = jm.GetCount(loc2, false);
    double p2 = c_t2 / (c_t2 + c_f2);

    if ( (p1 < 0.1) || (p2 < 0.1)) {
      continue;
    }

    printf("%d %d %d <-> %d %d %d, mutual information %5.3f\n", loc1.i, loc1.j, loc1.k, loc2.i, loc2.j, loc2.k, mi);
    printf("\t       F      T\n");
    printf("\t   ------------\n");
    printf("\t F | %04d  %04d\n", jm.GetCount(loc1, loc2, false, false), jm.GetCount(loc1, loc2, false, true));
    printf("\t T | %04d  %04d\n", jm.GetCount(loc1, loc2, true, false), jm.GetCount(loc1, loc2, true, true));
    printf("\n");

    osg::Vec3 sp(loc1.i*res, loc1.j*res, loc1.k*res);
    osg::Vec3 ep(loc2.i*res, loc2.j*res, loc2.k*res);

    osg::ref_ptr<osg::Vec3Array> vertices = new osg::Vec3Array();

    // set vertices
    vertices->push_back(sp);
    vertices->push_back(ep);

    osg::ref_ptr<osg::Geometry> geometry = new osg::Geometry();
    osg::ref_ptr<osg::DrawElementsUInt> line =
            new osg::DrawElementsUInt(osg::PrimitiveSet::LINES, 0);
    line->push_back(0);
    line->push_back(1);
    geometry->addPrimitiveSet(line);

    osg::ref_ptr<osg::LineWidth> linewidth = new osg::LineWidth(1.0);
    geometry->getOrCreateStateSet()->setAttribute(linewidth);

    // turn off lighting
    geometry->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF | osg::StateAttribute::OVERRIDE);

    double s = mi/max_mi;
    osg::ref_ptr<osg::Vec4Array> colors = new osg::Vec4Array();
    colors->push_back(osg::Vec4(1-sqrt(s), 0, sqrt(s), 1));
    geometry->setColorArray(colors);
    geometry->setColorBinding(osg::Geometry::BIND_PER_PRIMITIVE_SET);

    geometry->setVertexArray(vertices);

    addChild(geometry);
  }
}

} // namespace kitti_occ_grids
} // namespace app
