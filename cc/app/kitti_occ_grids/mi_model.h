#pragma once

#include <vector>

#include <boost/serialization/vector.hpp>

#include "library/ray_tracing/occ_grid.h"
#include "library/ray_tracing/occ_grid_location.h"

namespace rt = library::ray_tracing;

namespace app {
namespace kitti_occ_grids {

class MiModel {
 public:
  struct Counter {
    //size_t counts[4] = {0, 0, 0, 0};
    int counts[4] = {0, 0, 0, 0};

    void Count(bool occ1, bool occ2) {
      counts[GetIndex(occ1, occ2)]++;
    }

    void Count(float lo1, float lo2) {
      Count(lo1 > 0, lo2 > 0);
    }

    size_t GetIndex(float lo1, float lo2) {
      return GetIndex(lo1 > 0, lo2 > 0);
    }

    size_t GetIndex(bool occ1, bool occ2) {
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

  MiModel(double range_xy, double range_z, double res);

  void MarkObservations(const rt::OccGrid &og);

  double GetResolution() const;

  void Save(const char *fn) const;
  static MiModel Load(const char *fn);

 private:
  // Just to make serialization easier
  MiModel();

  //std::map<std::pair<rt::Location, rt::Location>, Counter> counts_;
  double resolution_;
  double range_xy_;
  double range_z_;

  size_t n_xy_;
  size_t n_z_;

  std::vector<Counter> counts_;

  void MarkObservatonsWorker(const rt::OccGrid &og, size_t idx1_start, size_t idx1_end);

  size_t GetIndex(const rt::Location &loc1, const rt::Location &loc2) const;
  bool InRange(const rt::Location &loc) const;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int /* file_version */){
    ar & counts_;

    ar & resolution_;
    ar & range_xy_;
    ar & range_z_;

    ar & n_xy_;
    ar & n_z_;
  }

};

} // namespace kitti_occ_grids
} // namespace app
