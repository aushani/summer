#include "app/kitti_occ_grids/chow_lui_tree_osg_node.h"

#include <algorithm>

#include <osg/Geometry>
#include <osg/LineWidth>

namespace app {
namespace kitti_occ_grids {

ChowLuiTreeOSGNode::ChowLuiTreeOSGNode(const ChowLuiTree &clt) : osg::Group() {
  const auto parent_locs = clt.GetParentLocs();

  for (const auto &parent : parent_locs) {
    Render(parent, clt);
  }
}

void ChowLuiTreeOSGNode::Render(const rt::Location &loc, const ChowLuiTree &clt) {
  const auto node = clt.GetNode(loc);
  double res = clt.GetResolution();

  if (node.NumAncestors() > 0) {
    const auto &loc1 = node.GetAncestorLocation(0);
    const auto &loc2 = loc;

    //double mi = node.GetMutualInformation(0);
    double mi = log(2);

    if (node.GetMarginalProbability(true) < 0.1) {
      // Don't render
    } else {

      printf("%d %d %d <-> %d %d %d, mutual information %5.3f\n", loc1.i, loc1.j, loc1.k, loc2.i, loc2.j, loc2.k, mi);
      printf("\t Marginal: %5.3f%%\n", node.GetMarginalProbability(true) * 100);
      printf("\t P   L-> F          T\n");
      printf("\t v ------------------\n");
      printf("\t F | %5.3f%%  %5.3f%%\n", 100*node.GetConditionalProbability(false, 0, false), 100*node.GetConditionalProbability(true, 0, false));
      printf("\t T | %5.3f%%  %5.3f%%\n", 100*node.GetConditionalProbability(false, 0, true),  100*node.GetConditionalProbability(true, 0, true ));
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
  for (const auto &child : node.GetChildrenLocations()) {
    Render(child, clt);
  }
}

} // namespace kitti_occ_grids
} // namespace app
