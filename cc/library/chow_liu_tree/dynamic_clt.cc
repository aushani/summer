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

      //all_edges_.emplace_back(loc1, loc2, mi);
      all_edges_[loc1].emplace_back(loc1, loc2, mi);
      num_total_edges_++;

      // Conditionals
      std::pair<rt::Location, rt::Location> key_1(loc1, loc2);
      conditionals_.insert({key_1, ConditionalDistribution(loc1, loc2, jm)});

      std::pair<rt::Location, rt::Location> key_flipped(loc2, loc1);
      conditionals_.insert({key_flipped, ConditionalDistribution(loc2, loc1, jm)});
    }
  }
  size_t num_nodes = all_locs_.size();
  //printf("Took %5.3f seconds to get %ld edges, %ld nodes\n", t.GetSeconds(), num_total_edges_, num_nodes);

  BuildFullTree();
}

void DynamicCLT::BuildFullTree() {
  library::timer::Timer t;

  std::vector<BoostEdge> edges;
  std::vector<double> weights;

  LocIntMapper mapper;

  // Get edges and nodes that are known
  double max_mi = log(2); // max mutual information
  for (const auto &kv : all_edges_) {
    for (const auto &e : kv.second) {
      edges.emplace_back(mapper.GetInt(e.loc1), mapper.GetInt(e.loc2));
      weights.push_back(max_mi - e.mutual_information); // minimum spanning tree vs maximum spanning tree
    }
  }

  size_t num_edges = edges.size();
  size_t num_nodes = mapper.size();
  //printf("have %ld edges and %ld nodes\n", num_edges, num_nodes);

  // Now get MST
  Graph g(edges.begin(), edges.begin() + num_edges, weights.begin(), num_nodes);

  std::vector<VertexDescriptor> p(num_nodes);

  t.Start();
  boost::prim_minimum_spanning_tree(g, &p[0]);
  //printf("prim done in %5.3f ms\n", t.GetMs());

  // Build tree
  for (size_t i = 0; i < p.size(); i++) {
    int int_my_loc = i;
    int int_parent_loc = p[i];

    if (int_my_loc == int_parent_loc) {
      continue;
    }

    const auto &my_loc = all_locs_[int_my_loc];
    const auto &parent_loc = all_locs_[int_parent_loc];

    full_tree_[my_loc] = parent_loc;
  }
}

double DynamicCLT::BuildAndEvaluate(const rt::DenseOccGrid &dog) const {
  library::timer::Timer t;
  t.Start();

  std::vector<BoostEdge> edges;
  std::vector<double> weights;

  LocIntMapper mapper;

  // Get edges and nodes that are known
  double max_mi = log(2); // max mutual information
  for (const auto &kv : all_edges_) {
    if (!dog.IsKnown(kv.first)) {
      continue;
    }

    for (const auto &e : kv.second) {
      if (dog.IsKnown(e.loc2)) {
        edges.emplace_back(mapper.GetInt(e.loc1), mapper.GetInt(e.loc2));
        weights.push_back(max_mi - e.mutual_information); // minimum spanning tree vs maximum spanning tree
      }
    }
  }

  size_t num_edges = edges.size();
  size_t num_nodes = mapper.size();
  //printf("have %ld / %ld edges and %ld nodes in %7.3f ms\n", num_edges, num_total_edges_, num_nodes, t.GetMs());

  if (num_nodes == 0) {
    return 0.0;
  }

  // Now get MST
  Graph g(edges.begin(), edges.begin() + num_edges, weights.begin(), num_nodes);
  std::vector<VertexDescriptor> p(num_nodes);

  t.Start();
  boost::prim_minimum_spanning_tree(g, &p[0]);
  //printf("prim done in %7.3f ms\n", t.GetMs());

  // Evaluate MST
  double log_p = 0;

  t.Start();
  for (size_t i = 0; i < p.size(); i++) {
    BOOST_ASSERT(p[i] < num_nodes);

    const auto &my_loc = mapper.GetLocation(i);
    BOOST_ASSERT(dog.IsKnown(my_loc));

    bool occu = dog.GetProbability(my_loc) > 0.5;

    if (i == p[i]) {
      // Is root node
      const auto &it = marginals_.find(my_loc);
      BOOST_ASSERT(it != marginals_.end());

      log_p += it->second.GetLogProb(occu);
    } else {
      // Is child node
      const auto &parent_loc = mapper.GetLocation(p[i]);
      BOOST_ASSERT(dog.IsKnown(parent_loc));

      bool occu_parent = dog.GetProbability(parent_loc) > 0.5;

      std::pair<rt::Location, rt::Location> key(my_loc, parent_loc);
      const auto &it = conditionals_.find(key);
      BOOST_ASSERT(it != conditionals_.end());

      log_p += it->second.GetLogProb(occu, occu_parent);
    }
  }
  //printf("Traversed tree in %7.3f ms\n", t.GetMs());

  return log_p;
}

double DynamicCLT::EvaluateMarginal(const rt::DenseOccGrid &dog) const {
  double log_p = 0.0;

  for (const auto &kv : marginals_) {
    const auto &loc = kv.first;

    if (dog.IsKnown(loc)) {
      log_p += kv.second.GetLogProb(dog.GetProbability(loc) > 0.5);
    }
  }

  return log_p;
}

const std::map<rt::Location, rt::Location>& DynamicCLT::GetFullTree() const {
  return full_tree_;
}

double DynamicCLT::GetMarginal(const rt::Location &loc, bool occu) const {
  const auto &it = marginals_.find(loc);
  BOOST_ASSERT(it != marginals_.end());

  double log_p = it->second.GetLogProb(occu);

  return exp(log_p);
}

} // namespace chow_liu_tree
} // namespace library
