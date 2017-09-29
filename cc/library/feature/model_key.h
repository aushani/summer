#pragma once

namespace library {
namespace feature {

struct ModelKey {
  std::string classname;
  int angle_bin;

  // For boost serialization
  ModelKey() {};

  ModelKey(const std::string &cn, int b) :
    classname(cn), angle_bin(b) {}

  bool operator<(const ModelKey &k) const {
    if (classname != k.classname) {
      return classname < k.classname;
    }

    return angle_bin < k.angle_bin;
  }

  bool operator==(const ModelKey &k) const {
    return classname == k.classname && angle_bin == k.angle_bin;
  }

  bool operator!=(const ModelKey &k) const {
    return !( (*this)==k );
  }

  template<class Archive>
  void serialize(Archive & ar, const unsigned int /* file_version */) {
    ar & classname;
    ar & angle_bin;
  }
};

} // namespace feature
} // namespace library
