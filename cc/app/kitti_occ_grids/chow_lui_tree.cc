#include "app/kitti_occ_grids/chow_lui_tree.h"

#include <iostream>
#include <fstream>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/kruskal_min_spanning_tree.hpp>

#include "library/timer/timer.h"

namespace app {
namespace kitti_occ_grids {

ChowLuiTree::ChowLuiTree() {
}

ChowLuiTree::ChowLuiTree(const JointModel &jm) : resolution_(jm.GetResolution()) {
  library::timer::Timer t;
  t.Start();

  // Adapted from boost example

  // Typedef's for convience
  typedef boost::adjacency_list<boost::vecS, boost::vecS,
          boost::undirectedS, boost::no_property,
          boost::property<boost::edge_weight_t, double> > Graph;
  typedef boost::graph_traits<Graph>::edge_descriptor EdgeDescriptor;
  typedef std::pair<int, int> BoostEdge;

  // Making graph for boost;
  std::vector<BoostEdge> edges;
  std::vector<double> weights;

  // Mapping from node id (int) to location
  std::map<rt::Location, int> loc_int_mapping;
  std::vector<rt::Location> int_loc_mapping;

  // Get all edges
  int min_ij = -jm.GetNXY() / 2;
  int max_ij = min_ij + jm.GetNXY();

  int min_k = -jm.GetNZ() / 2;
  int max_k = min_k + jm.GetNZ();

  size_t n_at = 0;

  printf("%d-%d, %d-%d\n", min_ij, max_ij, min_k, max_k);

  for (int i1=min_ij; i1 < max_ij; i1++) {
    for (int j1=min_ij; j1 < max_ij; j1++) {
      for (int k1=min_k; k1 < max_k; k1++) {
        rt::Location loc1(i1, j1, k1);

        if (loc_int_mapping.count(loc1) == 0) {
          loc_int_mapping[loc1] = int_loc_mapping.size();
          int_loc_mapping.push_back(loc1);
        }

        int i_loc1 = loc_int_mapping[loc1];

        if (!jm.InRange(loc1)) {
          printf("Out of range! %d, %d, %d\n",
              loc1.i, loc1.j, loc1.k);
          continue;
        }

        for (int i2 = i1; i2 < max_ij; i2++) {
          for (int j2 = j1; j2 < max_ij; j2++) {
            for (int k2 = k1; k2 < max_k; k2++) {
              n_at++;
              if (n_at % (1000 * 1000) == 0) {
                printf("At %ld M\n", n_at/(1000*1000));
                printf("\tHave %ld edges, %ld nodes so far...\n",
                    edges.size(), int_loc_mapping.size());
              }

              rt::Location loc2(i2, j2, k2);

              if (loc1 == loc2) {
                continue;
              }

              if (loc_int_mapping.count(loc2) == 0) {
                loc_int_mapping[loc2] = int_loc_mapping.size();
                int_loc_mapping.push_back(loc2);
              }

              if (jm.GetNumObservations(loc1, loc2) < 0) {
                printf("Out of range! %d, %d, %d, \t%d, %d, %d\n",
                    loc1.i, loc1.j, loc1.k, loc2.i, loc2.j, loc2.k);
              }

              if (jm.GetNumObservations(loc1, loc2) < 10) {
                continue;
              }

              int i_loc2 = loc_int_mapping[loc2];

              double mi = jm.ComputeMutualInformation(loc1, loc2);
              double weight = -mi; // because we have minimum spanning tree but want max

              edges.push_back(BoostEdge(i_loc1, i_loc2));
              weights.push_back(weight);
            }
          }
        }
      }
    }
  }
  size_t num_edges = edges.size();
  size_t num_nodes = int_loc_mapping.size();
  printf("Took %5.3f seconds to get %ld edges, %ld nodes\n", t.GetSeconds(), num_edges, num_nodes);

  Graph g(edges.begin(), edges.begin() + num_edges, weights.begin(), num_nodes);

  boost::property_map<Graph, boost::edge_weight_t>::type weight = get(boost::edge_weight, g);
  std::vector<EdgeDescriptor> spanning_tree;

  printf("passing to boost...\n");
  t.Start();
  boost::kruskal_minimum_spanning_tree(g, std::back_inserter(spanning_tree));
  printf("kruskal done in %5.3f sec, have %ld edges\n", t.GetSeconds(), spanning_tree.size());

  // add edges
  for (std::vector < EdgeDescriptor >::iterator ei = spanning_tree.begin(); ei != spanning_tree.end(); ++ei) {
    int i_loc1 = source(*ei, g);
    int i_loc2 = target(*ei, g);

    const rt::Location &loc1 = int_loc_mapping[i_loc1];
    const rt::Location &loc2 = int_loc_mapping[i_loc2];

    double w = weight[*ei];

    AddEdge(Edge(loc1, loc2, w));
  }
}

void ChowLuiTree::AddEdge(const Edge &e) {
  tree_edges_.push_back(e);
}

double ChowLuiTree::GetResolution() const {
  return resolution_;
}

const std::vector<ChowLuiTree::Edge>& ChowLuiTree::GetEdges() const {
  return tree_edges_;
}

void ChowLuiTree::Save(const char *fn) const {
  std::ofstream ofs(fn);
  boost::archive::binary_oarchive oa(ofs);
  oa << (*this);
}

ChowLuiTree ChowLuiTree::Load(const char *fn) {
  ChowLuiTree t;
  std::ifstream ifs(fn);
  boost::archive::binary_iarchive ia(ifs);
  ia >> t;

  return t;
}

} // namespace kitti_occ_grids
} // namespace app
