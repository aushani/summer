#pragma once

#include <vector>
#include <map>

#include <boost/graph/adjacency_list.hpp>

#include "library/chow_liu_tree/joint_model.h"
#include "library/ray_tracing/dense_occ_grid.h"
#include "library/ray_tracing/occ_grid_location.h"

namespace rt = library::ray_tracing;

namespace library {
namespace chow_liu_tree {

class DynamicCLT {
 public:
  DynamicCLT(const JointModel &jm);

  double BuildAndEvaluate(const rt::DenseOccGrid &dog) const;
  double EvaluateMarginal(const rt::DenseOccGrid &dog) const;

  const std::map<rt::Location, rt::Location>& GetFullTree() const;

  double GetMarginal(const rt::Location &loc, bool occu) const;

 private:
  //static constexpr int kMinObservations_ = 10;
  //static constexpr double kMinMutualInformation_ = 0.01;
  static constexpr int kMinObservations_ = 100;
  static constexpr double kMinMutualInformation_ = 0.01;

  struct MarginalDistribution {
    float log_p[2] = {0.0, 0.0};

    MarginalDistribution(const rt::Location &loc, const JointModel &jm) {
      int c_t = jm.GetCount(loc, true);
      int c_f = jm.GetCount(loc, false);
      double denom = jm.GetNumObservations(loc);

      log_p[GetIndex(true)] = log(c_t/denom);
      log_p[GetIndex(false)] = log(c_f/denom);
    }

    double GetLogProb(bool occ) const {
      return log_p[GetIndex(occ)];
    }

    size_t GetIndex(bool occ) const {
      return occ ? 0:1;
    }
  };

  struct ConditionalDistribution {
    float log_p[4] = {0.0, 0.0, 0.0, 0.0};

    ConditionalDistribution(const rt::Location &loc, const rt::Location &loc_parent, const JointModel &jm) {
      for (int i=0; i<2; i++) {
        bool occ = i==0;

        for (int j=0; j<2; j++) {
          bool parent = j==0;

          int count = jm.GetCount(loc, occ, loc_parent, parent);
          int count_other = jm.GetCount(loc, !occ, loc_parent, parent);
          double denom = count + count_other;

          log_p[GetIndex(occ, parent)] = log(count/denom);
        }
      }
    }

    double GetLogProb(bool occ, bool given) const {
      return log_p[GetIndex(occ, given)];
    }

    size_t GetIndex(bool occ, bool given) const {
      size_t idx = 0;
      if (occ) {
        idx += 1;
      }

      if (given) {
        idx += 2;
      }

      return idx;
    }
  };

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

  std::vector<rt::Location> all_locs_;
  std::map<rt::Location, int> loc_to_int_;

  //std::vector<Edge> all_edges_;
  std::map<rt::Location, std::vector<Edge>> all_edges_;
  size_t num_total_edges_ = 0;

  std::map<rt::Location, MarginalDistribution> marginals_;

  std::map<rt::Location, rt::Location> full_tree_;

  // First is child, second is parent
  std::map<std::pair<rt::Location, rt::Location>, ConditionalDistribution> conditionals_;

  void BuildFullTree();
};

} // namespace chow_liu_tree
} // namespace library
