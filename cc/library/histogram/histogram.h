#pragma once

#include <map>
#include <cstddef>

#include <boost/assert.hpp>
#include <boost/serialization/map.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

namespace library {
namespace histogram {

class Histogram {
 public:
  Histogram();
  Histogram(double min, double max, double res);

  bool InRange(double val) const;
  double GetMin() const;
  double GetMax() const;
  double GetRes() const;

  void Mark(double val, double weight=1.0);
  void Clear();

  double GetCount(double val) const;
  double GetCountsTotal() const;

  double Sample() const;

  double GetProbability(double val) const;
  double GetLikelihood(double val) const;
  double GetCumulativeProbability(double val) const;

  double GetMedian() const;
  double GetPercentile(double percentile) const;

  bool IsCompatibleWith(const Histogram &hist) const;
  void Add(const Histogram &hist, double weight);

  void Blur(double std);

 private:
  double min_ = 0.0;
  double max_ = 0.0;
  double res_ = 1.0;

  double observed_min_ = 0.0;
  double observed_max_ = 0.0;

  std::map<size_t, double> counts_;
  size_t counts_index_max_ = 0;
  double counts_total_ = 0.0f;

  size_t GetIndex(double val) const;
  double GetValue(size_t idx) const;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int /* file_version */){
    ar & min_;
    ar & max_;
    ar & res_;

    ar & observed_min_;
    ar & observed_max_;

    ar & counts_;
    ar & counts_index_max_;
    ar & counts_total_;
  }
};

} // namespace histogram
} // namespace library
