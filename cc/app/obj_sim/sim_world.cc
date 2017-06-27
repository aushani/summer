#include "app/obj_sim/sim_world.h"

#include <random>
#include <chrono>

namespace hm = library::hilbert_map;

SimWorld::SimWorld() :
  bounding_box_(0, 0, 50, 50) {

  //objects_.push_back(Box(-3.0, 1.0, 1.9, 1.9));
  //objects_.push_back(Box(1.0, 5.0, 1.9, 1.9));
  //objects_.push_back(Box(3.0, 1.0, 1.9, 1.9));

  double lower_bound = -8;
  double upper_bound = 8;
  std::uniform_real_distribution<double> unif(lower_bound, upper_bound);
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine re(seed);

  // Make boxes in the world
  for (int i=0; i<5; i++) {
    double x = unif(re);
    double y = unif(re);
    //printf("Putting box at %5.3f, %5.3f\n", x, y);
    objects_.push_back(Box(x, y, 2.0, 2.0));
  }

}

void SimWorld::GenerateSimData(std::vector<hm::Point> *hits, std::vector<hm::Point> *origins) {
  Eigen::Vector2d origin(0.0, 0.0);
  Eigen::Vector2d hit;

  for (double angle = -M_PI; angle < M_PI; angle += 0.01) {
    double best_distance = bounding_box_.GetHit(origin, angle, &hit);

    Eigen::Vector2d b_hit;
    for ( Box &b : objects_) {
      double dist = b.GetHit(origin, angle, &b_hit);
      if (dist > 0 && dist < best_distance) {
        hit = b_hit;
        best_distance = dist;
      }
    }

    if (best_distance > 0) {
      hits->push_back(hm::Point(hit(0), hit(1)));
      origins->push_back(hm::Point(origin(0), origin(1)));
    }
  }
}

bool SimWorld::IsOccupied(float x, float y) {
  for (Box &b : objects_) {
    if (b.IsInside(x, y))
      return true;
  }

  return false;
}

const std::vector<Box>& SimWorld::GetObjects() {
  return objects_;
}
