#pragma once

namespace library {
namespace chow_lui_tree {

class JointModel {
 public:
  JointModel(double dim_range);

 private:
  struct Counter {
    int counts[4] = {0, 0, 0, 0};

    friend class boost:serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int /* file_version */) {
      ar & counts;
    }
  }:
};

}
}
