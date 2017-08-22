#include "app/kitti_occ_grids/chow_lui_tree_node.h"

#include <algorithm>

#include <osg/Geometry>
#include <osg/LineWidth>

namespace app {
namespace kitti_occ_grids {

ChowLuiTreeNode::ChowLuiTreeNode(const ChowLuiTree &clt) : osg::Group() {
  const auto edges = clt.GetEdges();
  double res = clt.GetResolution();

  std::vector<double> weights;
  for (const auto &e : edges) {
    weights.push_back(e.weight);
  }
  std::sort(weights.begin(), weights.begin() + weights.size());

  //for (int i=0; i<100; i++) {
  //  printf("weight[%d] = %f\n", i, weights[i]);
  //}

  for (const auto &e : edges) {
    const auto &loc1 = e.loc1;
    const auto &loc2 = e.loc2;
    double w = e.weight;

    //if (w > weights[1000]) {
    //  continue;
    //}

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

    osg::ref_ptr<osg::Vec4Array> colors = new osg::Vec4Array();
    colors->push_back(osg::Vec4(1, 0, 0, 0.1));
    geometry->setColorArray(colors);
    geometry->setColorBinding(osg::Geometry::BIND_PER_PRIMITIVE_SET);

    geometry->setVertexArray(vertices);

    addChild(geometry);

  }
}

} // namespace kitti_occ_grids
} // namespace app
