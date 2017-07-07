#include "object_model.h"

#include <math.h>

namespace ge = library::geometry;

ObjectModel::ObjectModel(double size, double res) :
 size_(size), res_(res), dim_size_(2*size/res + 1),
  counts_(dim_size_*dim_size_*2, 0L) {
}

bool ObjectModel::InBounds(const ge::Point &x) const {
  return GetTableIndex(x, 1.0) != -1;
}

double ObjectModel::EvaluateLikelihood(const ge::Point &x, double label) const {
  double count_free = LookupCount(x, -1.0);
  double count_occu = LookupCount(x, 1.0);
  double count_pos = count_free + count_occu;

  if (count_pos == 0)
    return 0.5;

  //double p_x = count_pos / sum_counts_;
  double p_y = (label > 0 ? count_occu:count_free) / (count_pos);
  return p_y;
  //double p = p_y * p_x;

  //double p = ((double) LookupCount(x, label)) / sum_counts_;
  //double l = p / (res_*res_);

  //return l;
}

void ObjectModel::Build(const ge::Point &x, double label) {
  int idx = GetTableIndex(x, label);
  if (idx < 0)
    return;

  counts_[idx]++;
  sum_counts_++;
}

int ObjectModel::LookupCount(const ge::Point &x, double label) const {
  int idx = GetTableIndex(x, label);
  if (idx < 0)
    return 0;

  return counts_[idx];
}

int ObjectModel::GetTableIndex(const ge::Point &x, double label) const {
  int idx_x = round((x.x + size_)/res_);
  if (idx_x < 0 || idx_x >= dim_size_)
    return -1;

  int idx_y = round((x.y + size_)/res_);
  if (idx_y < 0 || idx_y >= dim_size_)
    return -1;

  int idx_label = label > 0 ? 1:0;

  return (idx_x * dim_size_ + idx_y)*2 + idx_label;
}
