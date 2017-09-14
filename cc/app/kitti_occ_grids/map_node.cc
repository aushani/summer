#include "app/kitti_occ_grids/map_node.h"

#include "library/osg_nodes/colorful_box.h"

namespace osgn = library::osg_nodes;

namespace app {
namespace kitti_occ_grids {

MapNode::MapNode(const dt::Detector &detector, const kt::KittiChallengeData &kcd) : osg::Group() {
  // Get image
  osg::ref_ptr<osg::Image> im = GetImage(detector, kcd);

  // Now set up render
  osg::ref_ptr<osg::Texture2D> texture = new osg::Texture2D();
  texture->setResizeNonPowerOfTwoHint(false);
  texture->setImage(im);

  texture->setFilter(osg::Texture::MIN_FILTER, osg::Texture::NEAREST);
  texture->setFilter(osg::Texture::MAG_FILTER, osg::Texture::NEAREST);
  //terrain->getOrCreateStateSet()->setTextureAttribute(0, tex.get(), osg::StateAttribute::ON | osg::StateAttribute::OVERRIDE);

  osg::ref_ptr<osg::Geode> geode = new osg::Geode();

  const int width = detector.GetDim().n_x;
  const int height = detector.GetDim().n_y;
  SetUpTexture(texture, geode, 0, 0, width, height, 12);

  // Origin
  double scale = detector.GetDim().res;
  double x0 = detector.GetDim().min_x;
  double y0 = detector.GetDim().min_y;

  osg::Matrix m = osg::Matrix::identity();
  m.makeScale(scale, scale, scale);
  m.postMultTranslate(osg::Vec3d(x0, y0, -1.7)); // ground plane

  osg::ref_ptr<osg::MatrixTransform> map_image = new osg::MatrixTransform();
  map_image->setMatrix(m);

  // Ready to add
  map_image->addChild(geode);
  addChild(map_image);
}

osg::ref_ptr<osg::Image> MapNode::GetImage(const dt::Detector &detector, const kt::KittiChallengeData &kcd) const {
  const double min = -5;
  const double max = 10;
  const double range = max - min;

  const int width = detector.GetDim().n_x;
  const int height = detector.GetDim().n_y;
  const int depth = 1;

  osg::ref_ptr<osg::Image> im = new osg::Image();
  im->allocateImage(width, height, depth, GL_RGBA, GL_FLOAT);

  for (int i = 0; i < width; i++) {
    for (int j = 0; j < height; j++) {
      dt::ObjectState os = detector.GetDim().GetState(i, j);
      double x = os.x;
      double y = os.y;

      double p_car = detector.GetLogOdds("Car", x, y);
      //double p_cyclist = detector.GetLogOdds("Cyclist", os);
      //double p_pedestrian = detector.GetLogOdds("Pedestrian", os);

      double r = (p_car-min) / range;
      //double g = (p_cyclist-min) / range;
      //double b = (p_pedestrian-min) / range;

      if (r < 0) r = 0;
      if (r > 1) r = 1;

      //if (g < 0) g = 0;
      //if (g > 1) g = 1;

      //if (b < 0) b = 0;
      //if (b > 1) b = 1;

      double g = 0;
      double b = 0;

      osg::Vec4 color(r, g, b, 0.5);

      im->setColor(color, i, j, 0);
    }
  }

  /*
  Eigen::Matrix4d t_cv = kcd.GetTcv().inverse();
  for (const auto &label : kcd.GetLabels()) {
    Eigen::Vector4d center_camera(label.location[0], label.location[1], label.location[2], 1.0);
    Eigen::Vector3d center_vel = (t_cv * center_camera).hnormalized();

    int i = std::round(center_vel.x()/detector.GetResolution() + width/2.0) - 1;
    int j = std::round(center_vel.y()/detector.GetResolution() + height/2.0) - 1;

    if (i < 0 || i >= width || j < 0 || j >= height) {
      continue;
    }

    //im->setColor(osg::Vec4(1, 1, 1, 1), i, j, 0);
  }
  */

  return im;
}

void MapNode::SetUpTexture(osg::Texture2D *texture, osg::Geode *geode, double x0, double y0, double width, double height, int bin_num) const {
  // Adapted from dascar

  osg::ref_ptr<osg::Geometry> geometry = new osg::Geometry();
  osg::ref_ptr<osg::Vec3Array> vertices = new osg::Vec3Array;
  vertices->push_back(osg::Vec3(x0, y0, -1));
  vertices->push_back(osg::Vec3(x0+width, y0, -1));
  vertices->push_back(osg::Vec3(x0+width, y0+height, -1));
  vertices->push_back(osg::Vec3(x0, y0+height, -1));

  osg::ref_ptr<osg::DrawElementsUInt> background_indices = new osg::DrawElementsUInt(osg::PrimitiveSet::POLYGON, 0);
  background_indices->push_back(0);
  background_indices->push_back(1);
  background_indices->push_back(2);
  background_indices->push_back(3);

  osg::ref_ptr<osg::Vec4Array> colors = new osg::Vec4Array;
  colors->push_back(osg::Vec4(1.0f, 1.0f, 1.0f, 0.9f));

  osg::ref_ptr<osg::Vec2Array> texcoords = new osg::Vec2Array(4);
  (*texcoords)[0].set(0.0f,0.0f);
  (*texcoords)[1].set(1.0f,0.0f);
  (*texcoords)[2].set(1.0f,1.0f);
  (*texcoords)[3].set(0.0f,1.0f);

  geometry->setTexCoordArray(0,texcoords);
  texture->setDataVariance(osg::Object::DYNAMIC);

  osg::ref_ptr<osg::Vec3Array> normals = new osg::Vec3Array;
  normals->push_back(osg::Vec3(0.0f,0.0f,1.0f));
  geometry->setNormalArray(normals);
  geometry->setNormalBinding(osg::Geometry::BIND_OVERALL);
  geometry->addPrimitiveSet(background_indices);
  geometry->setVertexArray(vertices);
  geometry->setColorArray(colors);
  geometry->setColorBinding(osg::Geometry::BIND_OVERALL);

  geode->addDrawable(geometry);

  // Create and set up a state set using the texture from above:
  osg::ref_ptr<osg::StateSet> state_set = new osg::StateSet();
  geode->setStateSet(state_set);
  state_set->setTextureAttributeAndModes(0, texture, osg::StateAttribute::ON);

  // For this state set, turn blending on (so alpha texture looks right)
  state_set->setMode(GL_BLEND,osg::StateAttribute::ON);

  // Disable depth testing so geometry is draw regardless of depth values
  // of geometry already draw.
  state_set->setMode(GL_DEPTH_TEST,osg::StateAttribute::OFF);
  state_set->setRenderingHint( osg::StateSet::TRANSPARENT_BIN );

  // Need to make sure this geometry is draw last. RenderBins are handled
  // in numerical order so set bin number to 11 by default
  state_set->setRenderBinDetails( bin_num, "RenderBin");
}

} // namespace kitti_occ_grids
} // namespace app
