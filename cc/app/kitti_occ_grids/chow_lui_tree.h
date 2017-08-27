#pragma once

#include <map>
#include <vector>

#include <boost/optional.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>

#include "library/ray_tracing/occ_grid_location.h"
#include "library/ray_tracing/dense_occ_grid.h"

#include "app/kitti_occ_grids/clt_node.h"
#include "app/kitti_occ_grids/joint_model.h"

namespace rt = library::ray_tracing;

namespace app {
namespace kitti_occ_grids {

class ChowLuiTree {
 public:
  // Edges used to construct tree
  struct Edge {
    rt::Location loc1;
    rt::Location loc2;

    double weight;

    Edge(const Edge &e) : loc1(e.loc1), loc2(e.loc2), weight(e.weight) { }

    Edge(const rt::Location &l1, const rt::Location &l2, double w) :
      loc1(l1), loc2(l2), weight(w) {}

    bool operator<(const Edge& e) const {
      if (loc1 != e.loc1) {
        return loc1 < e.loc1;
      }

      return loc2 < e.loc2;
    }

    bool operator==(const Edge &e) const {
      return loc1 == e.loc1 && loc2 == e.loc2;
    }

    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int /* file_version */){
      ar & loc1;
      ar & loc2;

      ar & weight;
    }

   private:
    // For convience with boost serialization
    Edge() : loc1(), loc2(), weight(0.0) {}
  };

  ChowLuiTree(const JointModel &jm);
  ChowLuiTree(const JointModel &jm, const rt::DenseOccGrid &dog);

  double GetResolution() const;

  const std::vector<rt::Location>& GetParentLocs() const;
  const CLTNode& GetNode(const rt::Location &loc) const;

  enum class EvalType {
    DENSE,    // Dense evaluation of tree, assumes all nodes are known!
    LOTP,     // Uses Law of Total Probability to approximate result for missing nodes
    SC,       // Short-circuits from children to grandparents (and beyond) if parents missing (approximate, misses correlation between children)
    MARGINAL  // Evaluates using only marginal probabilities
  };

  double EvaluateLogProbability(const rt::DenseOccGrid &dog, const EvalType &type) const;

  rt::OccGrid Sample() const;

  size_t Size() const;
  size_t NumRoots() const;

  void Save(const char *fn) const;
  static ChowLuiTree Load(const char *fn);

 private:
  static constexpr double kMaxDistanceBetweenNodes_ = 1.0;
  static constexpr double kMinNumObservations_ = 100;

  double resolution_;

  std::map<rt::Location, CLTNode> nodes_;
  std::vector<rt::Location> parent_locs_;

  // For convience with boost serialization
  ChowLuiTree();

  void ConstructEdges(const JointModel &jm, const boost::optional<const rt::DenseOccGrid&> &dog);
  void MakeTree(const std::vector<ChowLuiTree::Edge> &e, const JointModel &jm);

  void SampleHelper(const CLTNode &node_at, std::map<rt::Location, bool> *sample_og_pointer, std::default_random_engine *rand_engine) const;

  double EvaluateLogProbabilityHelperLOTP(const CLTNode &node_at, rt::DenseOccGrid *dog) const;
  double EvaluateLogProbabilityHelperSC(const CLTNode &node_at, const rt::DenseOccGrid &dog, int level_at, int last_observed_parent) const;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int /* file_version */){
    ar & resolution_;

    ar & nodes_;
    ar & parent_locs_;
  }
};

} // namespace kitti_occ_grids
} // namespace app
