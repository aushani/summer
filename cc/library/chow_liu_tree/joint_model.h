#pragma once

#include <vector>

#include <boost/serialization/vector.hpp>

#include "library/ray_tracing/occ_grid.h"

namespace rt = library::ray_tracing;

namespace library {
namespace chow_liu_tree {

class JointModel {
 public:
  JointModel(double range_xy, double range_z, double res);

  void MarkObservations(const rt::OccGrid &og);

  int GetCount(const rt::Location &loc, bool occ) const;
  int GetCount(const rt::Location &loc1, bool occ1, const rt::Location &loc2, bool occ2) const;

  int GetNumObservations(const rt::Location &loc1) const;
  int GetNumObservations(const rt::Location &loc1, const rt::Location &loc2) const;

  double GetMutualInformation(const rt::Location &loc1, const rt::Location &loc2) const;

  double GetResolution() const;
  size_t GetNXY() const;
  size_t GetNZ() const;
  bool InRange(const rt::Location &loc) const;

  void Save(const char *fn) const;
  static JointModel Load(const char *fn);

 private:
  struct Counter {
    //int counts[4] = {0, 0, 0, 0};
    int counts[4] = {1, 1, 1, 1};

    void Count(bool occ1, bool occ2) {
      counts[GetIndex(occ1, occ2)]++;
    }

    int GetCount(bool occ1, bool occ2) const {
      return counts[GetIndex(occ1, occ2)];
    }

    int GetTotalCount() const {
      return counts[0] + counts[1] + counts[2] + counts[3];
    }

    double GetMutualInformation() const {
      double mi = 0;

      double c_total = 0;
      for (int i=0; i<4; i++) {
        c_total += counts[i];
      }

      for (int i=0; i<2; i++) {
        bool occ1 = (i==0);
        double p_x = (GetCount(occ1, true) + GetCount(occ1, false)) / c_total;

        for (int j=0; j<2; j++) {
          bool occ2 = (j==0);

          double p_xy = GetCount(occ1, occ2) / c_total;
          double p_y = (GetCount(true, occ2) + GetCount(false, occ2)) / c_total;

          if (p_xy < 1e-10) {
            continue;
          }

          mi += p_xy * log(p_xy / (p_x*p_y) );
        }
      }

      return mi;
    }

    int GetIndex(bool occ1, bool occ2) const {
      size_t idx = 0;
      if (occ1) {
        idx += 1;
      }

      if (occ2) {
        idx += 2;
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
  JointModel();

  void MarkObservationsWorker(const rt::OccGrid &og, size_t idx1_start, size_t idx1_end);

  size_t GetIndex(const rt::Location &loc1, const rt::Location &loc2) const;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int /* file_version */) {
    ar & resolution_;
    ar & range_xy_;
    ar & range_z_;

    ar & n_xy_;
    ar & n_z_;
    ar & n_loc_;

    ar & counts_;
  }
};

} // namespace chow_liu_tree
} // namespace library
