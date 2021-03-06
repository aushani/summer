#pragma once

#include <thread>

#include <osg/ArgumentParser>
#include <boost/filesystem.hpp>

#include <osgGA/GUIActionAdapter>
#include <osgGA/GUIEventAdapter>
#include <osgGA/GUIEventHandler>
#include <osgViewer/CompositeViewer>

#include "library/feature/feature_model.h"
#include "library/osg_nodes/point_cloud.h"
#include "library/osg_nodes/object_labels.h"
#include "library/osg_nodes/tracklets.h"
#include "library/osg_nodes/occ_grid.h"
#include "library/kitti/kitti_challenge_data.h"
#include "library/ray_tracing/occ_grid_builder.h"
#include "library/timer/timer.h"
#include "library/viewer/viewer.h"

#include "app/kitti_occ_grids/detector_handler.h"
#include "app/kitti_occ_grids/map_node.h"

namespace dt = library::detector;
namespace ft = library::feature;
namespace kt = library::kitti;
namespace rt = library::ray_tracing;
namespace osgn = library::osg_nodes;
namespace fs = boost::filesystem;
namespace vw = library::viewer;

namespace app {
namespace kitti_occ_grids {

class DetectorApp : public osgGA::GUIEventHandler {
 public:
  DetectorApp(osg::ArgumentParser *args, bool viewer = true);

  void Run();

  bool SetFrame(int f);
  kt::KittiChallengeData Process();

  bool handle(const osgGA::GUIEventAdapter& ea, osgGA::GUIActionAdapter& aa);

  const dt::Detector& GetDetector() const;

 private:
  static constexpr char* my_results_dir = "/home/aushani/summer/cc/results/";
  static constexpr int kFirstFrame_ = 0;
  static constexpr int kLastFrame_ = 7481;
  static constexpr float kRes_ = 0.2;
  int frame_at_ = 0;
  std::string dirname_;

  dt::Detector detector_;
  std::shared_ptr<vw::Viewer> viewer_;
  rt::OccGridBuilder og_builder_;

  bool render_og_ = false;

  std::shared_ptr<std::thread> process_thread_;

  void ProcessBackground();
};

} // namespace viewer
} // namespace app
