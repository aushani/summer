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

      mutual_information_[{loc1, loc2}] = mi;
      mutual_information_[{loc2, loc1}] = mi;
    }
  }
  size_t num_nodes = all_locs_.size();
  printf("Took %5.3f seconds to get %ld edges, %ld nodes\n", t.GetSeconds(), num_total_edges_, num_nodes);

  BuildFullTree(jm);
}

void DynamicCLT::BuildFullTree(const JointModel &jm) {
  library::timer::Timer t;

  std::vector<BoostEdge> edges;
  std::vector<double> weights;

  LocIntMapper mapper;

  // Get edges and nodes that are known
  double max_mi = log(2); // max mutual information
  for (const auto &kv : all_edges_) {
    for (const auto &e : kv.second) {
      edges.emplace_back(mapper.GetInt(e.loc1), mapper.GetInt(e.loc2));

      double edge_weight = max_mi - e.mutual_information;
      BOOST_ASSERT(edge_weight > 0);
      weights.push_back(edge_weight); // minimum spanning tree vs maximum spanning tree
    }
  }

  size_t num_edges = edges.size();
  size_t num_nodes = mapper.size();
  //printf("have %ld edges and %ld nodes\n", num_edges, num_nodes);

  // Now get MST
  Graph g(edges.begin(), edges.begin() + num_edges, weights.begin(), num_nodes);

  std::vector<VertexDescriptor> p(num_nodes);

  //t.Start();
  boost::prim_minimum_spanning_tree(g, &p[0]);
  //printf("prim done in %5.3f ms\n", t.GetMs());

  // Build tree

  // Build nodes
  std::vector<std::shared_ptr<CLTNode> > nodes;
  for (size_t i = 0; i < p.size(); i++) {
    const auto &my_loc = mapper.GetLocation(i);
    //const auto &my_loc = all_locs_[i];
    nodes.emplace_back(new CLTNode(my_loc, jm));
  }

  // Assign children and get root(s)
  for (size_t i = 0; i < p.size(); i++) {
    int int_my_loc = i;
    int int_parent_loc = p[i];

    if (int_my_loc == int_parent_loc) {
      // Root node
      full_tree_.push_back(nodes[int_my_loc]);
    } else {
      // Child node
      auto &parent = nodes[int_parent_loc];
      nodes[i]->SetParent(parent, jm);
    }
  }

  // How much mi?
  //double total_mi = 0;
  //for (const auto &node : nodes) {
  //  double mi = node->GetNodeMutualInformation();
  //  //printf("\tmi %f\n", mi);
  //  total_mi += mi;
  //}
  //printf("Total MI: %f\n", total_mi);
  //printf("Total MI: %f\n", full_tree_[0]->GetTreeMutualInformation());

  printf("Took %7.3f ms to build full tree\n", t.GetMs());
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

double DynamicCLT::BuildAndEvaluateGreedy(const rt::OccGrid &og) const {
  const auto &locs = og.GetLocations();
  const auto &los = og.GetLogOdds();

  double log_p = 0;
  size_t sz = locs.size();

  for (size_t i=0; i<sz; i++) {
    const auto &loc1 = locs[i];
    const bool occ1 = los[i] > 0;

    int best_idx = -1;
    double best_mi = 0;

    int start = 0;
    if (i > 1) {
      start = i - 1;
    }
    start = i;

    for (size_t j=start; j<i; j++) {
      const auto &loc2 = locs[j];

      std::pair<rt::Location, rt::Location> key(loc1, loc2);
      const auto &it = mutual_information_.find(key);
      if (it == mutual_information_.end()) {
        continue;
      }

      double mi = it->second;

      if (mi > best_mi || best_idx < 0) {
        best_idx = j;
        best_mi = mi;
      }
    }

    if (best_idx < 0) {
      // Marginal
      const auto &it = marginals_.find(loc1);
      BOOST_ASSERT(it != marginals_.end());

      log_p += it->second.GetLogProb(occ1);
    } else {
      // Conditional
      const auto &loc2 = locs[best_idx];
      const bool occ2 = los[best_idx] > 0;

      std::pair<rt::Location, rt::Location> key(loc1, loc2);
      const auto &it = conditionals_.find(key);
      BOOST_ASSERT(it != conditionals_.end());

      log_p += it->second.GetLogProb(occ1, occ2);
    }
  }

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

const Tree& DynamicCLT::GetFullTree() const {
  return full_tree_;
}

Tree DynamicCLT::GetGreedyTree(const JointModel &jm) const {
  library::timer::Timer t;
  library::timer::Timer t_extra;
  double ms_extra = 0;

  Tree tree;

  // Greedily build tree
  std::map<rt::Location, std::shared_ptr<CLTNode> > nodes;
  for (const auto &loc : all_locs_) {
    std::shared_ptr<CLTNode> node(new CLTNode(loc, jm));

    // Find best edge to nodes in tree
    std::shared_ptr<CLTNode> best_node;
    double best_mi = 0;
    for (const auto &kv : nodes) {
      t_extra.Start();
      if (jm.GetNumObservations(loc, kv.second->GetLocation()) < kMinObservations_) {
        continue;
      }

      double mi = jm.GetMutualInformation(loc, kv.second->GetLocation());
      ms_extra += t_extra.GetMs();

      if (mi > best_mi) {
        best_mi = mi;
        best_node = kv.second;
      }
    }

    if (best_node) {
      //printf("got %f mi\n", best_mi);
      node->SetParent(best_node, jm);
    } else {
      //printf("no best node, root?\n");
      tree.push_back(node);
    }

    nodes[loc] = node;
  }

  printf("Took %7.3f ms to build greedy tree\n", t.GetMs() - ms_extra);

  return tree;
}

double DynamicCLT::GetMarginal(const rt::Location &loc, bool occu) const {
  const auto &it = marginals_.find(loc);
  BOOST_ASSERT(it != marginals_.end());

  double log_p = it->second.GetLogProb(occu);

  return exp(log_p);
}

} // namespace chow_liu_tree
} // namespace library
