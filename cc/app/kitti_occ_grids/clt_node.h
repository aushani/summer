#pragma once

#include <vector>

#include <boost/serialization/vector.hpp>

#include "library/ray_tracing/occ_grid_location.h"

#include "app/kitti_occ_grids/joint_model.h"

namespace rt = library::ray_tracing;

namespace app {
namespace kitti_occ_grids {

// Nodes in the CLT Tree
// TODO templated?
class CLTNode {
 public:
  CLTNode(){};  // for boost serialization

  CLTNode(const rt::Location &loc, const JointModel &jm);
  CLTNode(const rt::Location &loc, const CLTNode &parent_node, const JointModel &jm);

  bool HasParent() const;

  void AddChild(const rt::Location &child_loc);

  const rt::Location& GetLocation() const;
  const std::vector<rt::Location>& GetChildrenLocations() const;

  size_t NumAncestors() const;
  const rt::Location& GetAncestorLocation(size_t levels) const;

  //double GetMutualInformation(size_t ancestor_level) const;

  double GetConditionalProbability(bool node_occu, size_t ancestor_level, bool ancestor_occu) const;
  double GetMarginalProbability(bool occu) const;

 private:
  // Helper struct to capture conditional distribution between this node and parent nodes
  struct ConditionalDistribution {
    double p_occu_given_occu = 0.0;
    double p_free_given_occu = 0.0;

    double p_occu_given_free = 0.0;
    double p_free_given_free = 0.0;

    // For boost serialization
    ConditionalDistribution() {};

    ConditionalDistribution(const rt::Location &my_loc, const rt::Location &ancestor_loc, const JointModel &jm) {
      {
        int c_tt = jm.GetCount(ancestor_loc, my_loc, true, true);
        int c_tf = jm.GetCount(ancestor_loc, my_loc, true, false);
        p_occu_given_occu = c_tt / (static_cast<double>(c_tt + c_tf));
        p_free_given_occu = c_tf / (static_cast<double>(c_tt + c_tf));
      }

      {
        int c_ft = jm.GetCount(ancestor_loc, my_loc, false, true);
        int c_ff = jm.GetCount(ancestor_loc, my_loc, false, false);
        p_occu_given_free = c_ft / (static_cast<double>(c_ft + c_ff));
        p_free_given_free = c_ff / (static_cast<double>(c_ft + c_ff));
      }
    }

    double GetProbability(bool occu, bool given_occu) const {
      if (occu) {
        if (given_occu) {
          return p_occu_given_occu;
        } else {
          return p_occu_given_free;
        }
      } else {
        if (given_occu) {
          return p_free_given_occu;
        } else {
          return p_free_given_free;
        }
      }
    }

    friend class boost::serialization::access;
    template <class Archive>
    void serialize(Archive &ar, const unsigned int /* file_version */) {
      ar & p_occu_given_occu;
      ar & p_free_given_occu;

      ar & p_occu_given_free;
      ar & p_free_given_free;
    }
  };

  // Helper struct to capture marginal distribution
  struct MarginalDistribution {
    double p_occu = 0.0;
    double p_free = 0.0;

    // For boost serializtion
    MarginalDistribution() {};

    MarginalDistribution(const rt::Location &my_loc, const JointModel &jm) {
      int c_t = jm.GetCount(my_loc, true);
      int c_f = jm.GetCount(my_loc, false);
      p_occu = c_t / (static_cast<double>(c_t + c_f));
      p_free = c_f / (static_cast<double>(c_t + c_f));
    }

    double GetProbability(bool occu) const {
      if (occu) {
        return p_occu;
      } else {
        return p_free;
      }
    }

    friend class boost::serialization::access;
    template <class Archive>
    void serialize(Archive &ar, const unsigned int /* file_version */) {
      ar & p_occu;
      ar & p_free;
    }
  };

  // Location this node in the tree corresponds to
  rt::Location loc_;

  // Node chidren
  std::vector<rt::Location> children_locs_;

  // Node ancestors and distributions, starting from parent going all the way up the tree so we can short circuit if
  // necessary because we're missing observations
  // These are parallel data structures
  std::vector<rt::Location>             ancestors_locs_;
  std::vector<ConditionalDistribution>  ancestors_dists_;

  // Marginal
  MarginalDistribution marginal_;

  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive &ar, const unsigned int /* file_version */) {
    ar & loc_;

    ar & children_locs_;

    ar & ancestors_locs_;
    ar & ancestors_dists_;

    ar & marginal_;
  }
};

} // namespace kitti_occ_grids
} // namespace app
