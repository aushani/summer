#include "library/chow_liu_tree/dynamic_clt.h"

#include <boost/graph/prim_minimum_spanning_tree.hpp>

#include "library/timer/timer.h"

namespace rt = library::ray_tracing;

namespace library {
namespace chow_liu_tree {

DynamicCLT::DynamicCLT(const JointModel &jm) {
  // Make and store all possible edges
  library::timer::Timer t;
  t.Start();

  // Get all nodes
  int min_ij = - (jm.GetNXY() / 2);
  int max_ij = min_ij + jm.GetNXY();

  int min_k = - (jm.GetNZ() / 2);
  int max_k = min_k + jm.GetNZ();

  for (int i=min_ij; i < max_ij; i++) {
    for (int j=min_ij; j < max_ij; j++) {
      for (int k=min_k; k < max_k; k++) {
        rt::Location loc(i, j, k);

        all_locs_.push_back(loc);

        // Marginals
        marginals_.insert({loc, MarginalDistribution(loc, jm)});
      }
    }
  }

  // Get all edges
  for (size_t idx1 = 0; idx1 < all_locs_.size(); idx1++) {
    const auto &loc1 = all_locs_[idx1];

    for (size_t idx2 = idx1+1; idx2 < all_locs_.size(); idx2++) {
      const auto &loc2 = all_locs_[idx2];

      if (jm.GetNumObservations(loc1, loc2) < kMinObservations_) {
        continue;
      }

      double mi = jm.GetMutualInformation(loc1, loc2);
      if (mi < kMinMutualInformation_) {
        continue;
      }

      all_edges_.emplace_back(loc1, loc2, mi);

      // Conditionals
      std::pair<rt::Location, rt::Location> key_1(loc1, loc2);
      conditionals_.insert({key_1, ConditionalDistribution(loc1, loc2, jm)});

      std::pair<rt::Location, rt::Location> key_flipped(loc2, loc1);
      conditionals_.insert({key_flipped, ConditionalDistribution(loc2, loc1, jm)});
    }
  }
  size_t num_edges = all_edges_.size();
  size_t num_nodes = all_locs_.size();
  printf("Took %5.3f seconds to get %ld edges, %ld nodes\n", t.GetSeconds(), num_edges, num_nodes);
}

double DynamicCLT::BuildAndEvaluate(const rt::DenseOccGrid &dog) const {
  library::timer::Timer t;
  t.Start();

  std::vector<BoostEdge> edges;
  std::vector<double> weights;

  // Get edges and nodes that are known
  std::map<rt::Location, int> loc_to_int;
  std::vector<rt::Location> int_to_loc;
  double max_mi = log(2); // max mutual information
  for (const auto &e : all_edges_) {
    if (dog.IsKnown(e.loc1) && dog.IsKnown(e.loc2)) {

      if (loc_to_int.count(e.loc1) == 0) {
        loc_to_int[e.loc1] = int_to_loc.size();
        int_to_loc.push_back(e.loc1);
      }

      if (loc_to_int.count(e.loc2) == 0) {
        loc_to_int[e.loc2] = int_to_loc.size();
        int_to_loc.push_back(e.loc2);
      }

      edges.emplace_back(loc_to_int[e.loc1], loc_to_int[e.loc2]);
      weights.push_back(max_mi - e.mutual_information); // minimum spanning tree vs maximum spanning tree
      printf("\tedge %d - %d with weight %f\n", edges[edges.size()-1].first, edges[edges.size()-1].second, weights[weights.size()-1]);
    }
  }

  size_t num_edges = edges.size();
  size_t num_nodes = int_to_loc.size();
  printf("have %ld edges and %ld nodes\n", num_edges, num_nodes);

  if (num_nodes == 0) {
    return 0.0;
  }

  // Now get MST
  Graph g(edges.begin(), edges.begin() + num_edges, weights.begin(), num_nodes);

  std::vector<VertexDescriptor> p(num_nodes);

  t.Start();
  boost::prim_minimum_spanning_tree(g, &p[0]);
  printf("prim done in %5.3f ms\n", t.GetMs());

  // Evaluate MST
  double log_p = 0;
  size_t roots = 0;

  for (size_t i = 0; i < p.size(); i++) {
    int int_my_loc = i;
    int int_parent_loc = p[i];

    const auto &my_loc = int_to_loc[int_my_loc];
    bool occu = dog.GetProbability(my_loc) > 0.5;

    if (int_my_loc == int_parent_loc) {
      // Is root node
      const auto &it = marginals_.find(my_loc);
      BOOST_ASSERT(it != marginals_.end());

      log_p += it->second.GetLogProb(occu);
      roots++;
    } else {
      // Is child node
      const auto &parent_loc = int_to_loc[int_parent_loc];
      bool occu_parent = dog.GetProbability(my_loc) > 0.5;

      const auto &it = conditionals_.find({my_loc, parent_loc});
      BOOST_ASSERT(it != conditionals_.end());

      log_p += it->second.GetLogProb(occu, occu_parent);
    }
  }
  printf("have %ld / %ld root nodes\n", roots, num_nodes);

  return log_p;
}

double DynamicCLT::EvaluateMarginal(const rt::DenseOccGrid &dog) const {
  return 0.0;
}

} // namespace chow_liu_tree
} // namespace library
