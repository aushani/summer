#include "library/sim_world/sim_world.h"

#include <random>
#include <chrono>

namespace ge = library::geometry;

namespace library {
namespace sim_world {

SimWorld::SimWorld(int n_shapes) :
  bounding_box_(Shape::CreateBox(0, 0, 50, 50)),
  origin_(0.0, 0.0) {

  double lower_bound = -8;
  double upper_bound = 8;
  std::uniform_real_distribution<double> unif(lower_bound, upper_bound);
  std::uniform_real_distribution<double> rand_size(1.0, 2.0);
  std::uniform_real_distribution<double> rand_angle(-M_PI, M_PI);
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine re(seed);

  // Make boxes in the world
  int attempts = 0;
  while (shapes_.size() < n_shapes) {
    if (attempts++ > 1000)
      break;

    double x = unif(re);
    double y = unif(re);
    double size = rand_size(re);
    double angle = rand_angle(re);

    if (std::abs(x) < 3 && std::abs(y) < 3)
      continue;

    // Check to see if this object's center is within 3m of any other object
    //bool too_close = false;
    //for (const auto &s : GetShapes()) {
    //  const auto &c_s = s.GetCenter();

    //  if (std::abs(c_s(0) - x) < 6 && std::abs(c_s(1) - y) < 6) {
    //    too_close = true;
    //    continue;
    //  }
    //}

    //if (too_close)
    //  continue;

    Shape obj = Shape::CreateStar(x, y, size);
    //Shape obj = Shape::CreateBox(0, 3, 2, 2);
    obj.Rotate(angle);

    // Check for origin inside
    if (!obj.IsInside(0, 0))
      shapes_.push_back(obj);
  }

}

void SimWorld::AddShape(const Shape &obj) {
  shapes_.push_back(obj);
}

double SimWorld::GetHit(const Eigen::Vector2d &ray, Eigen::Vector2d *hit) {
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

void SimWorld::GenerateSimData(std::vector<ge::Point> *hits, std::vector<ge::Point> *origins) {
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

void SimWorld::GenerateSimData(std::vector<ge::Point> *points, std::vector<float> *labels) {
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

void SimWorld::GenerateGrid(double size, std::vector<ge::Point> *points, std::vector<float> *labels, double res) {
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

void SimWorld::GenerateSamples(size_t trials, std::vector<ge::Point> *points, std::vector<float> *labels, bool visible, bool occluded) {
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

void SimWorld::GenerateAllSamples(size_t trials, std::vector<ge::Point> *points, std::vector<float> *labels) {
  bool visible = true;
  bool occluded = true;
  GenerateSamples(trials, points, labels, visible, occluded);
}

void SimWorld::GenerateVisibleSamples(size_t trials, std::vector<ge::Point> *points, std::vector<float> *labels) {
  bool visible = true;
  bool occluded = false;
  GenerateSamples(trials, points, labels, visible, occluded);
}

void SimWorld::GenerateOccludedSamples(size_t trials, std::vector<ge::Point> *points, std::vector<float> *labels) {
  bool visible = false;
  bool occluded = true;
  GenerateSamples(trials, points, labels, visible, occluded);
}

bool SimWorld::IsOccupied(float x, float y) {
  for (Shape &b : shapes_) {
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

const std::vector<Shape>& SimWorld::GetShapes() {
  return shapes_;
}

bool SimWorld::IsVisible(float x, float y) {
  Eigen::Vector2d point(x, y);
  Eigen::Vector2d ray = (point - origin_);

  Eigen::Vector2d hit;
  double distance = GetHit(ray, &hit);

  return distance >= ray.norm();
}

bool SimWorld::IsOccluded(float x, float y) {
  return !IsVisible(x, y);
}

std::vector<ge::Point> SimWorld::GetObjectLocations() {
  std::vector<ge::Point> locs;

  for(auto shape : shapes_) {
    auto center = shape.GetCenter();
    locs.emplace_back(center(0), center(1));
  }

  return locs;
}

}
}
