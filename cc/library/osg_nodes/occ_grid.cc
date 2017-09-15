// Adapted from dascar
#include "library/osg_nodes/occ_grid.h"

#include <iostream>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <osg/Geometry>
#include <osg/LineWidth>

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

    double alpha = 1;

    double r = 0.1;
    double g = 0.9;
    double b = 0.1;
    if (fog.HasStats(loc)) {
      rt::Stats stats;

      for (int di=-1; di<=1; di++) {
        for (int dj=-1; dj<=1; dj++) {
          for (int dk=-1; dk<=1; dk++) {
            rt::Location loc_i(loc.i + di, loc.j + dj, loc.k + dk);

            if (fog.HasStats(loc_i)) {
              stats = stats + fog.GetStats(loc_i);
            }
          }
        }
      }
      //printf("count: %d\n", stats.count);
      //printf("var: %f, %f, %f\n", stats.GetCovX(), stats.GetCovY(), stats.GetCovZ());
      //printf("int: %f, %f\n", stats.intensity, stats.GetIntensityCov());
      //printf("\n");
      //r = stats.var_x;
      //g = stats.var_y;
      //b = stats.var_z;

      Eigen::Matrix3d cov;
      cov(0, 0) = stats.GetCovX();
      cov(1, 1) = stats.GetCovY();
      cov(2, 2) = stats.GetCovZ();

      cov(0, 1) = stats.GetCovXY();
      cov(1, 0) = stats.GetCovXY();

      cov(0, 2) = stats.GetCovXZ();
      cov(2, 0) = stats.GetCovXZ();

      cov(1, 2) = stats.GetCovYZ();
      cov(2, 1) = stats.GetCovYZ();

      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(cov);

      // Get the normal
      Eigen::Vector3d evals = eig.eigenvalues();
      int idx = 0;
      if (std::abs(evals(1)) < std::abs(evals(idx))) idx = 1;
      if (std::abs(evals(2)) < std::abs(evals(idx))) idx = 2;

      Eigen::Matrix3d evecs = eig.eigenvectors();
      Eigen::Vector3d normal = evecs.col(idx);

      //std::cout << "normal vector: \n" << normal << std::endl;

      Eigen::Vector3d pos(x, y, z);
      if (pos.dot(normal) > 0) {
        normal *= -1;
      }

      // Get angles
      double dx = 0.3*normal.x();
      double dy = 0.3*normal.y();
      double dz = 0.3*normal.z();

      DrawNormal(x, y, z, dx, dy, dz);

      //double d = sqrt(x*x + y*y + z*z);
      //double t = atan(y/x);
      //double p = acos(z/d);

      //r = (t + M_PI) / (2 * M_PI);
      //g = (p + M_PI) / (2 * M_PI);
      //b = 0;

      //if (r < 0) r = 0;
      //if (r > 1) r = 1;

      //if (g < 0) g = 0;
      //if (g > 1) g = 1;

      //if (b < 0) b = 0;
      //if (b > 1) b = 1;
    }

    osg::Vec4 color(r, g, b, alpha);
    osg::Vec3 pos(x, y, z);

    osg::ref_ptr<ColorfulBox> box = new ColorfulBox(color, pos, scale);
    addChild(box);
  }
}

void OccGrid::DrawNormal(double x, double y, double z, double dx, double dy, double dz) {
  osg::ref_ptr<osg::Vec3Array> vertices = new osg::Vec3Array();

  vertices->push_back(osg::Vec3(x, y, z));
  vertices->push_back(osg::Vec3(x+dx, y+dy, z+dz));

  osg::ref_ptr<osg::Geometry> geometry = new osg::Geometry();
  osg::ref_ptr<osg::DrawElementsUInt> line =
          new osg::DrawElementsUInt(osg::PrimitiveSet::LINES, 0);
  line->push_back(0);
  line->push_back(1);
  geometry->addPrimitiveSet(line);

  osg::ref_ptr<osg::LineWidth> linewidth = new osg::LineWidth(4.0);
  geometry->getOrCreateStateSet()->setAttribute(linewidth);

  // turn off lighting
  geometry->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF | osg::StateAttribute::OVERRIDE);

  osg::ref_ptr<osg::Vec4Array> colors = new osg::Vec4Array();
  colors->push_back(osg::Vec4(dx+0.5, dy+0.5, dz+0.5, 1.0));
  geometry->setColorArray(colors);
  geometry->setColorBinding(osg::Geometry::BIND_PER_PRIMITIVE_SET);

  geometry->setVertexArray(vertices);

  addChild(geometry);
}

}  // namespace osg_nodes
}  // namespace library
