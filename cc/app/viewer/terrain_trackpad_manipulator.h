// Adapted from dascar
#pragma once

#include <osgGA/TerrainManipulator>

namespace app {
namespace viewer {

class TerrainTrackpadManipulator : public osgGA::TerrainManipulator {
 public:
  TerrainTrackpadManipulator(int flags = DEFAULT_SETTINGS);
  bool performMovement() override;

 protected:
  virtual ~TerrainTrackpadManipulator() = default;
};

} // namespace viewer
} // namespace app
