#pragma once

#include <vector>

#include "library/geometry/point.h"
#include "library/sim_world/shape.h"

namespace ge = library::geometry;

namespace library {
namespace sim_world {

class SimWorld {
 public:
  SimWorld(int n_shapes);

  void AddShape(const Shape &obj);

  void GenerateSimData(std::vector<ge::Point> *hits, std::vector<ge::Point> *origins);
  void GenerateSimData(std::vector<ge::Point> *points, std::vector<float> *labels);
  void GenerateGrid(double size, std::vector<ge::Point> *points, std::vector<float> *labels, double res = 0.1);
  void GenerateAllSamples(size_t trials, std::vector<ge::Point> *points, std::vector<float> *labels);
  void GenerateVisibleSamples(size_t trials, std::vector<ge::Point> *points, std::vector<float> *labels);
  void GenerateOccludedSamples(size_t trials, std::vector<ge::Point> *points, std::vector<float> *labels);

  const std::vector<Shape>& GetShapes();

  bool IsOccupied(float x, float y);

  bool IsVisible(float x, float y);
  bool IsOccluded(float x, float y);

  double GetMinX() const;
  double GetMaxX() const;
  double GetMinY() const;
  double GetMaxY() const;

  std::vector<ge::Point> GetObjectLocations();

 private:
  std::vector<Shape> shapes_;
  Shape bounding_box_;

  Eigen::Vector2d origin_;

  double GetHit(const Eigen::Vector2d &ray, Eigen::Vector2d *hit);

  void GenerateSamples(size_t trials, std::vector<ge::Point> *points, std::vector<float> *labels, bool visible, bool occluded);
};

}
}
