#pragma once

#include <vector>
#include <memory>

#include <boost/optional.hpp>

#include "library/ray_tracing/occ_grid_location.h"

namespace rt = library::ray_tracing;

namespace library {
namespace chow_liu_tree {

class CLTNode : public std::enable_shared_from_this<CLTNode> {
 public:
  CLTNode(const rt::Location &loc);
  CLTNode(const rt::Location &loc, CLTNode *parent);

  bool HasParent() const;
  void SetParent(const std::shared_ptr<CLTNode> &parent);
  const std::shared_ptr<CLTNode>& GetParent() const;

  const std::vector<std::shared_ptr<CLTNode> >& GetChildren() const;

  const rt::Location& GetLocation() const;

 private:
  rt::Location loc_;
  std::vector<std::shared_ptr<CLTNode> > children_;

  std::shared_ptr<CLTNode> parent_;
};

} // namespace chow_liu_tree
} // namespace library
