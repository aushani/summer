#pragma once

#include "library/ray_tracing/occ_grid_location.h"

#include <boost/serialization/map.hpp>

namespace rt = library::ray_tracing;

namespace app {
namespace sim_world_occ_grids {

class Model {
 public:
  struct Counter {
    size_t occu_count = 0;
    size_t free_count = 0;

    void Count(bool occu) {
      if (occu) {
        occu_count++;
      } else {
        free_count++;
      }
    }

    size_t GetCount(bool occu) const {
      if (occu) {
        return occu_count;
      } else {
        return free_count;
      }
    }

    size_t GetTotalCount() const {
      return occu_count + free_count;
    }

    double GetProbability(bool occu) const {
      return static_cast<double>(GetCount(occu)) / static_cast<double>(GetTotalCount());
    }

    double GetProbability(float lo) const {
      return GetProbability(lo > 0);
    }

    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int /* file_version */){
      ar & occu_count;
      ar & free_count;
    }
  };

  Model(double res);

  void MarkObservation(const rt::Location &loc, bool occu);
  double GetProbability(const rt::Location &loc, bool occu) const;

  void MarkObservation(const rt::Location &loc, float lo);
  double GetProbability(const rt::Location &loc, float lo) const;

  size_t GetSupport(const rt::Location &loc) const;

  const std::map<rt::Location, Counter>& GetCounts() const;
  double GetResolution() const;

  void Save(const char *fn) const;
  static Model Load(const char *fn);


 private:
  Model();

  std::map<rt::Location, Counter> counts_;
  double resolution_;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int /* file_version */){
    ar & counts_;
    ar & resolution_;
  }

};

} // namespace sim_world_occ_grids
} // namespace app
