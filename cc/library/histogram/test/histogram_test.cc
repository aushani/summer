#include <boost/test/auto_unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include "library/histogram/histogram.h"

namespace ht = library::histogram;
namespace tt = boost::test_tools;

BOOST_AUTO_TEST_CASE(BoundsTest) {
  double min = -1.231;
  double max =  91.231;
  double res = 0.231;

  ht::Histogram h(min, max, res);
  BOOST_TEST(h.GetMin() == min, tt::tolerance(0.001));
  BOOST_TEST(h.GetMax() == max, tt::tolerance(0.001));
  BOOST_TEST(h.GetRes() == res, tt::tolerance(0.001));
}

BOOST_AUTO_TEST_CASE(IndexTest) {
  double min = -1.231;
  double max =  91.231;
  double res = 0.231;

  ht::Histogram h(min, max, res);

  for (double x = min - res*10; x < max + res*10; x += res/10) {
    double count_before = h.GetCount(x);
    h.Mark(x);
    double count_after = h.GetCount(x);

    BOOST_TEST(count_before + 1.0 == count_after);
  }
}

BOOST_AUTO_TEST_CASE(PercentileTest) {
  double min = 0;
  double max = 100;
  double res = 1.0;

  ht::Histogram h(min, max, res);

  for (double x = min; x <= max; x += res/100.0) {
    h.Mark(x);
  }

  for (double x = min; x <= max; x += res/100.0) {
    double cdf = h.GetCumulativeProbability(x);
    BOOST_CHECK_SMALL(x/(max - min) - cdf, res/(max - min) + 1e-3);
  }

  for (int percentile = 0; percentile <= 100; percentile++) {
    double p = percentile/100.0;
    double val = h.GetPercentile(p);
    BOOST_CHECK_SMALL(percentile - val, res/(max - min) + 1e-3);
  }
}

BOOST_AUTO_TEST_CASE(ClearTest) {
  double min = -1.231;
  double max =  91.231;
  double res = 0.231;

  ht::Histogram h(min, max, res);

  for (double x = min - res*10; x < max + res*10; x += res/10) {
    h.Mark(x);
  }

  BOOST_TEST(h.GetCountsTotal() > 0);

  h.Clear();

  BOOST_CHECK_SMALL(h.GetCountsTotal(), 1e-5);
}
