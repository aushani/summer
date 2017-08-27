#pragma once

#include <boost/graph/adjacency_list.hpp>

#include "library/ray_tracing/dense_occ_grid.h"

#include "app/kitti_occ_grids/joint_model.h"

namespace rt = library::ray_tracing;

namespace app {
namespace kitti_occ_grids {

class DynamicCLT {
 public:
  DynamicCLT(const JointModel &jm);

  double BuildAndEvaluate(const rt::DenseOccGrid &dog) const;
  double OldStyle(const rt::DenseOccGrid &dog) const;

 private:
  static constexpr double kMaxDistanceBetweenNodes_ = 1.0;
  static constexpr double kMinNumObservations_ = 100;
  static constexpr double kMinMutualInformation_ = 0.01;

  // Typedef's for convience
  typedef boost::adjacency_list<boost::vecS, boost::vecS,
          boost::undirectedS, boost::no_property,
          boost::property<boost::edge_weight_t, double> > Graph;
  typedef boost::graph_traits<Graph>::edge_descriptor EdgeDescriptor;

  typedef std::pair<int, int> BoostEdge;
  typedef std::pair<rt::Location, rt::Location> LocEdge;

  // Dense graph
  std::vector<LocEdge> possible_edges_;
  std::vector<double> possible_weights_;

  const JointModel jm_;
};

} // namespace kitti_occ_grids
} // namespace app
