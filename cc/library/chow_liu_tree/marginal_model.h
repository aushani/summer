#pragma once

#include <vector>

#include <boost/serialization/vector.hpp>

#include "library/ray_tracing/occ_grid.h"
#include "library/chow_liu_tree/joint_model.h"

namespace rt = library::ray_tracing;

namespace library {
namespace chow_liu_tree {

class MarginalModel {
 public:
  MarginalModel(double range_xy, double range_z, double res);
  MarginalModel(const JointModel &jm);

  void MarkObservations(const rt::OccGrid &og);

  int GetCount(const rt::Location &loc, bool occ) const;
  void SetCount(const rt::Location &loc, bool occ, int count);

  int GetNumObservations(const rt::Location &loc) const;

  double Evaluate(const rt::OccGrid &og) const;

  double GetResolution() const;
  size_t GetNXY() const;
  size_t GetNZ() const;
  bool InRange(const rt::Location &loc) const;

  void Save(const char *fn) const;
  static MarginalModel Load(const char *fn);

 private:
  struct Counter {
    //int counts[4] = {0, 0, 0, 0};
    int counts[2] = {0, 0};

    void Count(bool occ) {
      counts[GetIndex(occ)]++;
    }

    void SetCount(bool occ, int count) {
      counts[GetIndex(occ)] = count;
    }

    int GetCount(bool occ) const {
      return counts[GetIndex(occ)];
    }

    int GetTotalCount() const {
      return counts[0] + counts[1];
    }

    int GetIndex(bool occ) const {
      size_t idx = 0;
      if (occ) {
        idx += 1;
      }

      return idx;
    }

    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int /* file_version */) {
      ar & counts;
    }
  };

  double resolution_;
  double range_xy_;
  double range_z_;

  size_t n_xy_;
  size_t n_z_;
  size_t n_loc_;

  std::vector<Counter> counts_;

  // for easier boost serialization
  MarginalModel();

  void MarkObservationsWorker(const rt::OccGrid &og, size_t idx_start, size_t idx_end);

  size_t GetIndex(const rt::Location &loc) const;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int /* file_version */) {
    ar & resolution_;
    ar & range_xy_;
    ar & range_z_;

    ar & n_xy_;
    ar & n_z_;

    ar & counts_;
  }
};

} // namespace chow_liu_tree
} // namespace library
