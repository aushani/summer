#pragma once

#include <vector>

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

  ChowLuiTree(const JointModel &jm);

  void Save(const char *fn) const;
  static ChowLuiTree Load(const char *fn);

 private:
  std::vector<Edge> tree_edges_;

  // For convience with boost serialization
  ChowLuiTree();

  void AddEdge(const Edge &e);

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int /* file_version */){
    ar & tree_edges_;
  }

};

} // namespace kitti_occ_grids
} // namespace app
