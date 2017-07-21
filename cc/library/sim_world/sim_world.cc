#include "library/sim_world/sim_world.h"

#include <random>
#include <chrono>

namespace ge = library::geometry;

namespace library {
namespace sim_world {

SimWorld::SimWorld(size_t n_shapes) :
  bounding_box_(Shape::CreateBox(0, 0, 1000, 1000)),
  origin_(0.0, 0.0) {

  std::uniform_real_distribution<double> pos(-20.0, 20.0);
  std::uniform_real_distribution<double> width(2.0, 4.0);
  std::uniform_real_distribution<double> length(4.0, 8.0);
  std::uniform_real_distribution<double> rand_size(1.0, 2.0);
  std::uniform_real_distribution<double> rand_angle(-M_PI, M_PI);
  std::uniform_real_distribution<double> rand_shape(0.0, 1.0);
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine rand_engine(seed);

  // Make shapes in the world
  int attempts = 0;
  while (shapes_.size() < n_shapes) {
    if (attempts++ > 1000)
      break;

    double x = pos(rand_engine);
    double y = pos(rand_engine);
    double angle = rand_angle(rand_engine);

    bool make_box = rand_shape(rand_engine) < 0.5;
    bool make_star = !make_box;

    // Not too close to origin
    if (std::abs(x) < 3 && std::abs(y) < 3)
      continue;

    if (make_box) {
      Shape obj = Shape::CreateBox(x, y, width(rand_engine), length(rand_engine));
      //obj.Rotate(angle);

      // Check for origin inside
      if (!obj.IsInside(0, 0))
        shapes_.push_back(obj);
    } else if (make_star) {
      double size = rand_size(rand_engine);

      Shape obj = Shape::CreateStar(x, y, size);
      //obj.Rotate(angle);

      // Check for origin inside
      if (!obj.IsInside(0, 0))
        shapes_.push_back(obj);
    }
  }

}

void SimWorld::AddShape(const Shape &obj) {
  shapes_.push_back(obj);
}

double SimWorld::GetHit(const Eigen::Vector2d &ray, Eigen::Vector2d *hit) const {
  Eigen::Vector2d ray_hat = ray.normalized();
  double best_distance = bounding_box_.GetHit(origin_, ray_hat, hit);

  Eigen::Vector2d b_hit;
  for (const Shape &b : shapes_) {
    double dist = b.GetHit(origin_, ray_hat, &b_hit);
    if (dist > 0 && dist < best_distance) {
      *hit = b_hit;
      best_distance = dist;
    }
  }

  return best_distance;
}

void SimWorld::GenerateSimData(std::vector<ge::Point> *hits, std::vector<ge::Point> *origins) const {
  Eigen::Vector2d hit;

  for (double angle = -M_PI; angle < M_PI; angle += 0.01) {
    Eigen::Vector2d ray(cos(angle), sin(angle));
    double distance = GetHit(ray, &hit);

    if (distance > 0) {
      hits->push_back(ge::Point(hit(0), hit(1)));
      origins->push_back(ge::Point(origin_(0), origin_(1)));
    }
  }
}

void SimWorld::GenerateSimData(std::vector<ge::Point> *points, std::vector<float> *labels) const {
  Eigen::Vector2d hit;

  std::uniform_real_distribution<double> unif(0.0, 1.0);
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine re(seed);

  for (double angle = -M_PI; angle < M_PI; angle += 0.01) {
    Eigen::Vector2d ray(cos(angle), sin(angle));
    double distance = GetHit(ray, &hit);

    if (distance > 0) {
      points->push_back(ge::Point(hit(0), hit(1)));
      labels->push_back(1.0);

      for (int i=0; i<distance; i++) {
        double random_range = unif(re)*distance;
        Eigen::Vector2d free = origin_ + random_range * ray;

        points->push_back(ge::Point(free(0), free(1)));
        labels->push_back(-1.0);
      }
    }
  }
}

void SimWorld::GenerateGrid(double size, std::vector<ge::Point> *points, std::vector<float> *labels, double res) const {
  // Find extent of sim
  //double x_min = GetMinX();
  //double x_max = GetMaxX();
  //double y_min = GetMinY();
  //double y_max = GetMaxY();

  //// Expand by a bit
  //double x_range = x_max - x_min;
  //x_min -= x_range*0.10;
  //x_max += x_range*0.10;

  //double y_range = y_max - y_min;
  //y_min -= y_range*0.10;
  //y_max += y_range*0.10;

  for (double x = -size; x<size; x+=res) {
    for (double y = -size; y<size; y+=res) {
      points->emplace_back(x, y);
      labels->push_back(IsOccupied(x, y) ? 1.0:-1.0);
    }
  }
}

void SimWorld::GenerateSamples(size_t trials, std::vector<ge::Point> *points, std::vector<float> *labels, bool visible, bool occluded) const {
  // Find extent of sim
  double x_min = GetMinX();
  double x_max = GetMaxX();
  double y_min = GetMinY();
  double y_max = GetMaxY();

  std::uniform_real_distribution<double> random_x(x_min, x_max);
  std::uniform_real_distribution<double> random_y(y_min, y_max);

  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine re(seed);

  std::vector<ge::Point> occu_points;
  std::vector<ge::Point> free_points;

  for (size_t i=0; i<trials; i++) {
    double x = random_x(re);
    double y = random_y(re);

    // Check if it matches the flags we have
    // If we don't have visible points and it is visible, or we don't want occluded points and it is occluded
    if ( (!visible && IsVisible(x, y)) || (!occluded && IsOccluded(x, y)) ) {
      continue;
    }

    if (IsOccupied(x, y)) {
      occu_points.push_back(ge::Point(x, y));
    } else {
      free_points.push_back(ge::Point(x, y));
    }
  }

  size_t min = occu_points.size();
  if (free_points.size() < min)
    min = free_points.size();

  for (size_t i = 0; i<min; i++) {
    points->push_back(occu_points[i]);
    labels->push_back(1.0);

    points->push_back(free_points[i]);
    labels->push_back(-1.0);
  }
}

void SimWorld::GenerateAllSamples(size_t trials, std::vector<ge::Point> *points, std::vector<float> *labels) const {
  bool visible = true;
  bool occluded = true;
  GenerateSamples(trials, points, labels, visible, occluded);
}

void SimWorld::GenerateVisibleSamples(size_t trials, std::vector<ge::Point> *points, std::vector<float> *labels) const {
  bool visible = true;
  bool occluded = false;
  GenerateSamples(trials, points, labels, visible, occluded);
}

void SimWorld::GenerateOccludedSamples(size_t trials, std::vector<ge::Point> *points, std::vector<float> *labels) const {
  bool visible = false;
  bool occluded = true;
  GenerateSamples(trials, points, labels, visible, occluded);
}

bool SimWorld::IsOccupied(float x, float y) const {
  for (const Shape &b : shapes_) {
    if (b.IsInside(x, y))
      return true;
  }

  return false;
}

double SimWorld::GetMinX() const {
  return -10.0;
  //double x_min = 0.0;
  //bool first = false;
  //for (const Shape &s: shapes_) {
  //  double s_x_min = s.GetMinX();

  //  if (s_x_min < x_min || first)
  //    x_min = s_x_min;
  //}

  //return x_min;
}

double SimWorld::GetMaxX() const {
  return 10.0;
  //double x_max = 0.0;
  //bool first = false;
  //for (const Shape &s: shapes_) {
  //  double s_x_max = s.GetMaxX();

  //  if (s_x_max > x_max || first)
  //    x_max = s_x_max;
  //}

  //return x_max;
}

double SimWorld::GetMinY() const {
  return -10.0;
  //double y_min = 0.0;
  //bool first = false;
  //for (const Shape &s: shapes_) {
  //  double s_y_min = s.GetMinY();

  //  if (s_y_min < y_min || first)
  //    y_min = s_y_min;
  //}

  //return y_min;
}

double SimWorld::GetMaxY() const {
  return 10.0;
  //double y_max = 0.0;
  //bool first = false;
  //for (const Shape &s: shapes_) {
  //  double s_y_max = s.GetMaxY();

  //  if (s_y_max > y_max || first)
  //    y_max = s_y_max;
  //}

  //return y_max;
}

const std::vector<Shape>& SimWorld::GetShapes() const {
  return shapes_;
}

bool SimWorld::IsVisible(float x, float y) const {
  Eigen::Vector2d point(x, y);
  Eigen::Vector2d ray = (point - origin_);

  Eigen::Vector2d hit;
  double distance = GetHit(ray, &hit);

  return distance >= ray.norm();
}

bool SimWorld::IsOccluded(float x, float y) const {
  return !IsVisible(x, y);
}

std::vector<ge::Point> SimWorld::GetObjectLocations() const {
  std::vector<ge::Point> locs;

  for(auto shape : shapes_) {
    auto center = shape.GetCenter();
    locs.emplace_back(center(0), center(1));
  }

  return locs;
}

}
}
