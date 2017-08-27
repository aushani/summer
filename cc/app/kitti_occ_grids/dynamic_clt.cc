#include "dynamic_clt.h"

#include <boost/graph/kruskal_min_spanning_tree.hpp>

#include "library/timer/timer.h"

namespace rt = library::ray_tracing;

namespace app {
namespace kitti_occ_grids {

DynamicCLT::DynamicCLT(const JointModel &jm) :
 jm_(jm) {
  library::timer::Timer t;

  // Get all possible edges
  int min_ij = -jm.GetNXY() / 2;
  int max_ij = min_ij + jm.GetNXY();

  int min_k = -jm.GetNZ() / 2;
  int max_k = min_k + jm.GetNZ();

  int max_dijk = std::ceil(kMaxDistanceBetweenNodes_ / jm.GetResolution());
  double max_d2 = kMaxDistanceBetweenNodes_*kMaxDistanceBetweenNodes_;

  for (int i1=min_ij; i1 < max_ij; i1++) {
    for (int j1=min_ij; j1 < max_ij; j1++) {
      for (int k1=min_k; k1 < max_k; k1++) {
        rt::Location loc1(i1, j1, k1);
        if (!jm.InRange(loc1)) {
          continue;
        }

        int i_bound = std::min(i1 + max_dijk, max_ij);
        int j_bound = std::min(j1 + max_dijk, max_ij);
        int k_bound = std::min(k1 + max_dijk, max_k);

        for (int i2 = i1; i2 < i_bound; i2++) {
          for (int j2 = j1; j2 < j_bound; j2++) {
            for (int k2 = k1; k2 < k_bound; k2++) {
              double di = i1 - i2;
              double dj = j1 - j2;
              double dk = k1 - k2;

              double d2 = di*di + dj*dj + dk*dk;
              if (d2 > max_d2) {
                continue;
              }

              rt::Location loc2(i2, j2, k2);
              if (loc1 == loc2) {
                continue;
              }

              if (jm.GetNumObservations(loc1, loc2) < kMinNumObservations_) {
                continue;
              }

              double mi = jm.ComputeMutualInformation(loc1, loc2);
              if (mi < kMinMutualInformation_) {
                continue;
              }
              double weight = -mi; // because we have minimum spanning tree from boost but want max

              possible_edges_.push_back(LocEdge(loc1, loc2));
              possible_weights_.push_back(weight);
            }
          }
        }
      }
    }
  }
  size_t num_edges = possible_edges_.size();
  printf("Took %5.3f ms to get %ld possible edges\n", t.GetMs(), num_edges);
}

double DynamicCLT::BuildAndEvaluate(const rt::DenseOccGrid &dog) const {
  library::timer::Timer t;

  // Go through list of possible edges and see which ones we can evaluate
  std::vector<BoostEdge> edges;
  std::vector<double> weights;

  std::vector<rt::Location> int_to_loc;
  std::map<rt::Location, int> loc_to_int;

  for (size_t i=0; i<possible_edges_.size(); i++) {
    const LocEdge &le = possible_edges_[i];

    const rt::Location &loc1 = le.first;
    const rt::Location &loc2 = le.second;

    if (dog.IsKnown(loc1) && dog.IsKnown(loc2)) {
      // Maintain mapping from location to int
      if (loc_to_int.count(loc1) == 0) {
        loc_to_int[loc1] = int_to_loc.size();
        int_to_loc.push_back(loc1);
      }

      if (loc_to_int.count(loc2) == 0) {
        loc_to_int[loc2] = int_to_loc.size();
        int_to_loc.push_back(loc2);
      }

      int i_loc1 = loc_to_int[loc1];
      int i_loc2 = loc_to_int[loc2];

      BoostEdge be(i_loc1, i_loc2);
      edges.push_back(be);
      weights.push_back(possible_weights_[i]);
    }
  }
  size_t num_edges = edges.size();
  size_t num_nodes = int_to_loc.size();
  //printf("Took %5.3f ms to get %ld / %ld edges\n", t.GetMs(), num_edges, possible_edges_.size());

  // Now make graph
  Graph g(edges.begin(), edges.begin() + num_edges, weights.begin(), num_nodes);

  boost::property_map<Graph, boost::edge_weight_t>::type weight = get(boost::edge_weight, g);
  std::vector<EdgeDescriptor> spanning_tree;

  // Now Make Tree
  t.Start();
  boost::kruskal_minimum_spanning_tree(g, std::back_inserter(spanning_tree));
  //printf("kruskal done in %5.3f ms, have %ld edges\n", t.GetMs(), spanning_tree.size());

  // Make list of list of edges
  t.Start();
  std::vector<std::vector<int> > clt_edges(num_nodes, std::vector<int>());
  for (std::vector < EdgeDescriptor >::iterator ei = spanning_tree.begin(); ei != spanning_tree.end(); ++ei) {
    int i_loc1 = source(*ei, g);
    int i_loc2 = target(*ei, g);

    clt_edges[i_loc1].push_back(i_loc2);
    clt_edges[i_loc2].push_back(i_loc1);
  }
  //printf("Took %5.3f ms to make list of list of edges\n", t.GetMs());

  // Now traverse tree and evaluate edges
  double update = 0;
  std::vector<bool> visited(num_nodes, false);

  for (size_t i_loc = 0; i_loc < num_nodes; i_loc++) {
    if (visited[i_loc]) {
      continue;
    }

    std::deque<int> visit_queue;
    visit_queue.push_back(i_loc);

    // Evaluate as root to tree
    const rt::Location &loc_root = int_to_loc[i_loc];
    update += log(jm_.GetMarginalProbability(loc_root, dog.GetProbability(loc_root) > 0.5));

    // Visit and process all children
    while (visit_queue.size() > 0) {
      int i_visit = visit_queue.front();
      visit_queue.pop_front();

      if (visited[i_visit]) {
        continue;
      }
      visited[i_visit] = true;

      const rt::Location &parent = int_to_loc[i_visit];

      // Keep visiting children
      for (int i_child : clt_edges[i_visit]) {
        visit_queue.push_back(i_child);

        const rt::Location &child = int_to_loc[i_child];

        // Evaluate conditional
        update += log(jm_.GetConditionalProbability(child, dog.GetProbability(child) > 0.5,
                                                    parent, dog.GetProbability(parent) > 0.5));
      }
    }
  }
  //printf("Took %5.3f ms to traverse\n", t.GetMs());

  return update;
}

} // namespace kitti_occ_grids
} // namespace app

