#include "app/kitti_occ_grids/dynamic_clt.h"

#include <boost/graph/kruskal_min_spanning_tree.hpp>

#include "library/timer/timer.h"

#include "app/kitti_occ_grids/chow_lui_tree.h"

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

        for (int di = -max_dijk; di <= max_dijk; di++) {
          for (int dj = -max_dijk; dj <= max_dijk; dj++) {
            for (int dk = -max_dijk; dk <= max_dijk; dk++) {
              // Make location
              int i2 = i1 + di;
              int j2 = j1 + dj;
              int k2 = k1 + dk;
              rt::Location loc2(i2, j2, k2);

              // Checks
              if (!jm.InRange(loc2) || loc1 == loc2 || loc2 < loc1) {
                continue;
              }

              // Check distance
              double dx = di*jm.GetResolution();
              double dy = dj*jm.GetResolution();
              double dz = dk*jm.GetResolution();

              double d2 = dx*dx + dy*dy + dz*dz;
              if (d2 > max_d2) {
                continue;
              }

              // Check num observations
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
  //printf("dclt kruskal done in %5.3f ms, have %ld edges, %ld nodes\n", t.GetMs(), spanning_tree.size(), num_nodes);

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
  int num_updates = 0;
  int edges_evaled = 0;
  std::vector<bool> visited(num_nodes, false);

  for (size_t i_loc = 0; i_loc < num_nodes; i_loc++) {
    // Have we hit this node before?
    if (visited[i_loc]) {
      continue;
    }

    std::deque<int> visit_queue;
    visit_queue.push_back(i_loc);

    // Evaluate as root to tree
    const rt::Location &loc_root = int_to_loc[i_loc];
    update += log(jm_.GetMarginalProbability(loc_root, dog.GetProbability(loc_root) > 0.5));
    num_updates++;

    // Visit and process all connected nodes
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

        // Did we already eval this edge?
        if (visited[i_child]) {
          continue;
        }

        // Evaluate conditional
        update += log(jm_.GetConditionalProbability(child, dog.GetProbability(child) > 0.5,
                                                    parent, dog.GetProbability(parent) > 0.5));
        num_updates++;
        edges_evaled++;
      }
    }
  }
  //printf("Took %5.3f ms to traverse\n", t.GetMs());
  //printf("dclt evaled %d updates, %d edges, had %d nodes\n", num_updates, edges_evaled, num_nodes);

  return update;
}

double DynamicCLT::OldStyle(const rt::DenseOccGrid &dog) const {
  ChowLuiTree clt(jm_, dog);

  return clt.EvaluateLogProbability(dog, ChowLuiTree::EvalType::DENSE);
}

} // namespace kitti_occ_grids
} // namespace app

