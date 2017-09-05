#pragma once

#include <algorithm>
#include <vector>
#include <map>

#include <boost/graph/adjacency_list.hpp>

#include "library/chow_liu_tree/clt_node.h"
#include "library/chow_liu_tree/conditional_distribution.h"
#include "library/chow_liu_tree/joint_model.h"
#include "library/chow_liu_tree/marginal_distribution.h"
#include "library/ray_tracing/dense_occ_grid.h"
#include "library/ray_tracing/occ_grid_location.h"

namespace rt = library::ray_tracing;

namespace library {
namespace chow_liu_tree {

typedef std::vector<std::shared_ptr<CLTNode> > Tree;

class DynamicCLT {
 public:
  DynamicCLT(const JointModel &jm);

  double BuildAndEvaluate(const rt::DenseOccGrid &dog) const;
  double EvaluateMarginal(const rt::DenseOccGrid &dog) const;

  const Tree& GetFullTree() const;
  Tree GetGreedyTree(const JointModel &jm) const;

  double GetMarginal(const rt::Location &loc, bool occu) const;

 private:
  //static constexpr int kMinObservations_ = 10;
  //static constexpr double kMinMutualInformation_ = 0.01;
  static constexpr int kMinObservations_ = 100;
  static constexpr double kMinMutualInformation_ = 0.01;

  // Typedef's for convience
  typedef boost::adjacency_list<boost::vecS, boost::vecS,
          boost::undirectedS, boost::no_property,
          boost::property<boost::edge_weight_t, double> > Graph;
  typedef boost::graph_traits<Graph>::vertex_descriptor VertexDescriptor;
  typedef std::pair<int, int> BoostEdge;

  struct Edge {
    const rt::Location loc1;
    const rt::Location loc2;
    const double mutual_information;

    Edge(const rt::Location &l1, const rt::Location &l2, double mi) :
     loc1(l1), loc2(l2), mutual_information(mi) { }

    bool operator<(const Edge &e) const {
      if (loc1 != e.loc1) {
        return loc1 < e.loc1;
      }
       return loc2 < e.loc2;
    }
  };

  // for convience, mapping from loc to int
  struct LocIntMapper {
    std::map<rt::Location, int> loc_to_int;
    std::vector<rt::Location> int_to_loc;

    size_t size() {
      return int_to_loc.size();
    }

    rt::Location GetLocation(int i) {
      BOOST_ASSERT(i < int_to_loc.size());
      return int_to_loc[i];
    }

    int GetInt(const rt::Location &loc) {
      if (loc_to_int.count(loc) == 0) {
        loc_to_int[loc] = int_to_loc.size();

        //BOOST_ASSERT(std::find(int_to_loc.begin(), int_to_loc.end(), loc) == int_to_loc.end());
        int_to_loc.push_back(loc);
      }

      return loc_to_int[loc];
    }
  };

  std::vector<rt::Location> all_locs_;

  //std::vector<Edge> all_edges_;
  std::map<rt::Location, std::vector<Edge>> all_edges_;
  size_t num_total_edges_ = 0;

  std::map<rt::Location, MarginalDistribution> marginals_;

  Tree full_tree_;

  // First is child, second is parent
  std::map<std::pair<rt::Location, rt::Location>, ConditionalDistribution> conditionals_;

  void BuildFullTree(const JointModel &jm);
};

} // namespace chow_liu_tree
} // namespace library
