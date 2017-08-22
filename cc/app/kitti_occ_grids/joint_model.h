#pragma once

#include <vector>

#include <boost/serialization/vector.hpp>

#include "library/ray_tracing/occ_grid.h"
#include "library/ray_tracing/occ_grid_location.h"

namespace rt = library::ray_tracing;

namespace app {
namespace kitti_occ_grids {

class JointModel {
 public:
  struct Counter {
    //size_t counts[4] = {0, 0, 0, 0};
    int counts[4] = {0, 0, 0, 0};

    // Counting
    void Count(bool occ1, bool occ2) {
      counts[GetIndex(occ1, occ2)]++;
    }

    void Count(float lo1, float lo2) {
      Count(lo1 > 0, lo2 > 0);
    }

    // Accessors
    int GetCount(bool occ1, bool occ2) const {
      return counts[GetIndex(occ1, occ2)];
    }

    double GetProb(bool occ1, bool occ2) const {
      return GetCount(occ1, occ2) / static_cast<double>(GetTotalCount());
    }

    int GetCount1(bool occ1) const {
      return GetCount(occ1, true) + GetCount(occ1, false);
    }

    double GetProb1(bool occ1) const {
      return GetCount1(occ1) / static_cast<double>(GetTotalCount());
    }

    int GetCount2(bool occ2) const {
      return GetCount(true, occ2) + GetCount(false, occ2);
    }

    double GetProb2(bool occ2) const {
      return GetCount2(occ2) / static_cast<double>(GetTotalCount());
    }

    // float accessors, for convinence
    int GetCount(float lo1, bool lo2) const {
      return counts[GetIndex(lo1 > 0, lo2 > 0)];
    }

    int GetCount1(float lo1) const {
      return GetCount1(lo1 > 0);
    }

    int GetCount2(float lo2) const {
      return GetCount2(lo2 > 0);
    }


    int GetTotalCount() const {
      return counts[0] + counts[1] + counts[2] + counts[3];
    }

    // Helpers
    size_t GetIndex(float lo1, float lo2) const {
      return GetIndex(lo1 > 0, lo2 > 0);
    }

    size_t GetIndex(bool occ1, bool occ2) const {
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

  JointModel(double range_xy, double range_z, double res);

  void MarkObservations(const rt::OccGrid &og);

  double ComputeMutualInformation(const rt::Location &loc1, const rt::Location &loc2) const;
  int GetNumObservations(const rt::Location &loc1, const rt::Location &loc2) const;

  int GetCount(const rt::Location &loc, bool occu) const;

  double GetResolution() const;

  int GetNXY() const;
  int GetNZ() const;
  bool InRange(const rt::Location &loc) const;

  void Save(const char *fn) const;
  static JointModel Load(const char *fn);

 private:
  // Just to make serialization easier
  JointModel();

  //std::map<std::pair<rt::Location, rt::Location>, Counter> counts_;
  double resolution_;
  double range_xy_;
  double range_z_;

  size_t n_xy_;
  size_t n_z_;

  std::vector<Counter> counts_;

  void MarkObservatonsWorker(const rt::OccGrid &og, size_t idx1_start, size_t idx1_end);

  size_t GetIndex(const rt::Location &loc1, const rt::Location &loc2) const;

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
