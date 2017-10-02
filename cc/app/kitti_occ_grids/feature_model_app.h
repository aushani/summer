#pragma once

#include <thread>

#include <osgGA/GUIActionAdapter>
#include <osgGA/GUIEventAdapter>
#include <osgGA/GUIEventHandler>
#include <osgViewer/CompositeViewer>

#include "library/feature/model_bank.h"
#include "library/viewer/viewer.h"

namespace ft = library::feature;
namespace vw = library::viewer;

namespace app {
namespace kitti_occ_grids {

class FeatureModelApp : public osgGA::GUIEventHandler {
 public:
  FeatureModelApp(osg::ArgumentParser *args);

  void Run();

  bool handle(const osgGA::GUIEventAdapter& ea, osgGA::GUIActionAdapter& aa);

 private:
  std::vector<std::string> classnames_;
  size_t classname_at_ = 0;
  int angle_bin_at_ = 0;

  std::shared_ptr<vw::Viewer> viewer_;

  std::shared_ptr<ft::ModelBank> model_bank_;

  std::shared_ptr<std::thread> render_thread_;

  void Render(const std::string &classname, int angle_bin);
  void RenderBackground(const std::string &classname, int angle_bin);
};

} // namespace viewer
} // namespace app
