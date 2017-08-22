#pragma once

#include <map>
#include <vector>

#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>

#include "library/ray_tracing/occ_grid_location.h"

#include "app/kitti_occ_grids/joint_model.h"

namespace rt = library::ray_tracing;

namespace app {
namespace kitti_occ_grids {

class ChowLuiTree {
 public:
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

  // Nodes in the three
  class Node {
   public:
    Node() {}; // for bosst serialization

    Node(const rt::Location &l, const JointModel &jm) : loc_(l), parent_(), parent_valid_(false) {
      int c_t = jm.GetCount(loc_, true);
      int c_f = jm.GetCount(loc_, false);

      p_occu_parent_occu_ = 0.0;
      p_occu_parent_free_ = 0.0;

      p_occu_marginal_ = c_t / static_cast<double>(c_t + c_f);
    }

    Node(const rt::Location &l, const rt::Location &p, const JointModel &jm) : loc_(l), parent_(p), parent_valid_(true) {
      {
        int c_tt = jm.GetCount(parent_, loc_, true, true);
        int c_tf = jm.GetCount(parent_, loc_, true, false);
        p_occu_parent_occu_ = c_tt / (static_cast<double>(c_tt + c_tf));
      }

      {
        int c_ft = jm.GetCount(parent_, loc_, false, true);
        int c_ff = jm.GetCount(parent_, loc_, false, false);
        p_occu_parent_free_ = c_ft / (static_cast<double>(c_ft + c_ff));
      }

      {
        int c_t = jm.GetCount(loc_, true);
        int c_f = jm.GetCount(loc_, false);
        p_occu_marginal_ = c_t / (static_cast<double>(c_t + c_f));
      }

      mutual_information_ = jm.ComputeMutualInformation(parent_, loc_);
    }

    bool HasParent() const { return parent_valid_; }

    void AddChild(const Node &n) {
      children_.push_back(n.loc_);
    }

    const std::vector<rt::Location>& GetChildren() const {
      return children_;
    }

    const rt::Location& GetParentLocation() const {
      BOOST_ASSERT(HasParent());
      return parent_;
    }

    const rt::Location& GetLocation() const {
      return loc_;
    }

    double GetMutualInformation() const {
      return mutual_information_;
    }

    double GetConditionalProbability(bool parent_occu, bool loc_occu) const {
      BOOST_ASSERT(HasParent());

      double p = parent_occu ? p_occu_parent_occu_ : p_occu_parent_free_;
      return loc_occu ? p : (1-p);
    }

    double GetMarginalProbability(bool occu) const {
      return occu ? p_occu_marginal_ : (1 - p_occu_marginal_);
    }

   private:
    // Location this node in the tree corresponds to
    rt::Location loc_;

    // Node chidren
    std::vector<rt::Location> children_;

    // parent, if there is one
    rt::Location parent_;
    bool parent_valid_ = false;
    double mutual_information_ = 0.0;

    // Conditiional occupancy probabilities
    double p_occu_parent_occu_ = 0.0;
    double p_occu_parent_free_ = 0.0;

    // Marginal
    double p_occu_marginal_ = 0.0;

    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int /* file_version */){
      ar & loc_;

      ar & children_;

      ar & parent_;
      ar & parent_valid_;
      ar & mutual_information_;

      ar & p_occu_parent_occu_;
      ar & p_occu_parent_free_;
      ar & p_occu_marginal_;
    }
  };

  ChowLuiTree(const JointModel &jm);

  double GetResolution() const;

  const std::vector<rt::Location>& GetParentLocs() const;
  const Node& GetNode(const rt::Location &loc) const;

  rt::OccGrid Sample() const;

  void Save(const char *fn) const;
  static ChowLuiTree Load(const char *fn);

 private:
  static constexpr double kMaxDistanceBetweenNodes_ = 1.0;

  double resolution_;

  std::map<rt::Location, Node> nodes_;
  std::vector<rt::Location> parent_locs_;

  // For convience with boost serialization
  ChowLuiTree();

  std::vector<Edge> ConstructEdges(const JointModel &jm);
  void MakeTree(const std::vector<ChowLuiTree::Edge> &e, const JointModel &jm);

  void SampleHelper(const Node &node_at, std::map<rt::Location, bool> *sample_og_pointer, std::default_random_engine *rand_engine) const;

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
