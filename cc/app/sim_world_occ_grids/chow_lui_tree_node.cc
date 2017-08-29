#include "app/sim_world_occ_grids/chow_lui_tree_node.h"

#include <algorithm>

#include <osg/Geometry>
#include <osg/LineWidth>

namespace app {
namespace sim_world_occ_grids {

ChowLuiTreeNode::ChowLuiTreeNode(const ChowLuiTree &clt) : osg::Group() {
  const auto root_locs = clt.GetRootLocs();

  for (const auto &root : root_locs) {
    Render(root, clt);
  }
}

void ChowLuiTreeNode::Render(const rt::Location &loc, const ChowLuiTree &clt) {
  const auto node = clt.GetNode(loc);
  double res = clt.GetResolution();

  if (node.HasParent()) {
    const auto &loc1 = node.GetParentLocation();
    const auto &loc2 = loc;

    double mi = node.GetMutualInformation();

    if (node.GetMarginalProbability(true) < 0.1) {
      // Don't render
    } else {

      printf("%d %d %d <-> %d %d %d, mutual information %5.3f\n", loc1.i, loc1.j, loc1.k, loc2.i, loc2.j, loc2.k, mi);
      printf("\t Marginal: %5.3f%%\n", node.GetMarginalProbability(true) * 100);
      printf("\t P   L-> F          T\n");
      printf("\t v ------------------\n");
      printf("\t F | %5.3f%%  %5.3f%%\n", 100*node.GetConditionalProbability(false, false), 100*node.GetConditionalProbability(false, true));
      printf("\t T | %5.3f%%  %5.3f%%\n", 100*node.GetConditionalProbability(true, false),  100*node.GetConditionalProbability(true, true));
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

      double s = mi/log(2);
      osg::ref_ptr<osg::Vec4Array> colors = new osg::Vec4Array();
      colors->push_back(osg::Vec4(1-sqrt(s), 0, sqrt(s), 1));
      geometry->setColorArray(colors);
      geometry->setColorBinding(osg::Geometry::BIND_PER_PRIMITIVE_SET);

      geometry->setVertexArray(vertices);

      addChild(geometry);
    }
  }

  // Do same for children
  for (const auto &child : node.GetChildren()) {
    Render(child, clt);
  }
}

} // namespace sim_world_occ_grids
} // namespace app
