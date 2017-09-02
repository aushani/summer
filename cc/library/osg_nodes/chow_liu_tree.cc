#include "library/osg_nodes/chow_liu_tree.h"

#include <osg/Geometry>
#include <osg/LineWidth>

namespace clt = library::chow_liu_tree;

namespace library {
namespace osg_nodes {

ChowLiuTree::ChowLiuTree(const clt::DynamicCLT &clt) : osg::Group() {
  const auto &tree = clt.GetFullTree();

  for (const auto &root_node : tree) {
    Render(root_node);
  }
}

void ChowLiuTree::Render(const std::shared_ptr<clt::CLTNode> &node) {
  double thresh = log(0.1);

  // Render this node
  if (node->HasParent()) {
    double lp1 = node->GetMarginalLogProb(true);
    double lp2 = node->GetParent()->GetMarginalLogProb(true);

    if (lp1 > thresh && lp2 > thresh) {
      const auto &child_loc = node->GetLocation();
      const auto &parent_loc = node->GetParent()->GetLocation();

      osg::Vec3 sp(child_loc.i, child_loc.j, child_loc.k);
      osg::Vec3 ep(parent_loc.i, parent_loc.j, parent_loc.k);

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
      colors->push_back(osg::Vec4(1, 0, 0, 1));
      geometry->setColorArray(colors);
      geometry->setColorBinding(osg::Geometry::BIND_PER_PRIMITIVE_SET);

      geometry->setVertexArray(vertices);

      addChild(geometry);
    }
  }

  // Render children
  for (const auto &child : node->GetChildren()) {
    Render(child);
  }
}

}  // namespace osg_nodes
}  // namespace library
