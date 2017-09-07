#include <iostream>

#include <osg/ArgumentParser>
#include <boost/filesystem.hpp>

#include "library/kitti/velodyne_scan.h"
#include "library/osg_nodes/point_cloud.h"
#include "library/osg_nodes/occ_grid.h"
#include "library/osg_nodes/tracklets.h"
#include "library/ray_tracing/occ_grid_builder.h"
#include "library/ray_tracing/dense_occ_grid.h"
#include "library/timer/timer.h"
#include "library/viewer/viewer.h"

//#include "app/kitti_occ_grids/detector.h"
#include "app/kitti_occ_grids/map_node.h"

namespace dt = library::detector;
namespace kt = library::kitti;
namespace rt = library::ray_tracing;
namespace osgn = library::osg_nodes;
namespace vw = library::viewer;
namespace kog = app::kitti_occ_grids;
namespace fs = boost::filesystem;

kt::VelodyneScan LoadVelodyneScan(const std::string &kitti_log_dir,
                                  const std::string &kitti_log_date,
                                  int log_num,
                                  int frame_num) {
  char fn[1000];
  sprintf(fn, "%s/%s/%s_drive_%04d_sync/velodyne_points/data/%010d.bin",
      kitti_log_dir.c_str(), kitti_log_date.c_str(), kitti_log_date.c_str(), log_num, frame_num);

  return kt::VelodyneScan(fn);
}

kt::Tracklets LoadTracklets(const std::string &kitti_log_dir,
                            const std::string &kitti_log_date,
                            int log_num) {
  // Load Tracklets
  char fn[1000];
  sprintf(fn, "%s/%s/%s_drive_%04d_sync/tracklet_labels.xml",
      kitti_log_dir.c_str(), kitti_log_date.c_str(), kitti_log_date.c_str(), log_num);
  kt::Tracklets tracklets;
  if (!tracklets.loadFromFile(fn)) {
    printf("Could not load tracklets from %s\n", fn);
  }

  return tracklets;
}

int main(int argc, char** argv) {
  osg::ArgumentParser args(&argc, argv);
  osg::ApplicationUsage* au = args.getApplicationUsage();

  // report any errors if they have occurred when parsing the program arguments.
  if (args.errors()) {
    args.writeErrorMessages(std::cout);
    return EXIT_FAILURE;
  }

  au->setApplicationName(args.getApplicationName());
  au->setCommandLineUsage( args.getApplicationName() + " [options]");
  au->setDescription(args.getApplicationName() + " viewer");

  au->addCommandLineOption("--kitti-log-dir <dirname>", "KITTI data directory", "~/data/kittidata/extracted/");
  au->addCommandLineOption("--kitti-log-date <dirname>", "KITTI date", "2011_09_26");
  au->addCommandLineOption("--log-num <num>", "KITTI log number", "18");
  au->addCommandLineOption("--frame-num <num>", "KITTI frame number", "0");
  au->addCommandLineOption("--models <dir>", "Models to evaluate", "");

  // handle help text
  // call AFTER init viewer so key bindings have been set
  unsigned int helpType = 0;
  if ((helpType = args.readHelpType())) {
    au->write(std::cout, helpType);
    return EXIT_SUCCESS;
  }

  // Read params
  std::string home_dir = getenv("HOME");
  std::string kitti_log_dir = home_dir + "/data/kittidata/extracted/";
  if (!args.read("--kitti-log-dir", kitti_log_dir)) {
    printf("Using default KITTI log dir: %s\n", kitti_log_dir.c_str());
  }

  std::string kitti_log_date = "2011_09_26";
  if (!args.read("--kitti-log-date", kitti_log_date)) {
    printf("Using default KITTI date: %s\n", kitti_log_date.c_str());
  }

  int log_num = 18;
  if (!args.read("--log-num", log_num)) {
    printf("Using default KITTI log number: %d\n", log_num);
  }

  int frame_num = 0;
  if (!args.read("--frame-num", frame_num)) {
    printf("Using default KITTI frame number: %d\n", frame_num);
  }

  std::string model_dir = "/home/aushani/data/trainer/0009/";
  if (!args.read("--models", model_dir)) {
    printf("no model given, using default dir %s\n", model_dir.c_str());
  }

  // Load velodyne scan
  printf("Loading vel\n");
  kt::VelodyneScan scan = LoadVelodyneScan(kitti_log_dir, kitti_log_date, log_num, frame_num);
  printf("Loading tracklets\n");
  kt::Tracklets tracklets = LoadTracklets(kitti_log_dir, kitti_log_date, log_num);

  // Detector
  dt::Detector detector(0.5, 50, 50);

  // Load models
  //kog::Detector detector(og.GetResolution(), 50, 50);
  printf("loading models from %s\n", model_dir.c_str());

  fs::directory_iterator end_it;
  for (fs::directory_iterator it(model_dir); it != end_it; it++) {
    // Make sure it's not a directory
    if (!fs::is_regular_file(it->path())) {
      continue;
    }

    // Make sure it's a joint model
    if (fs::extension(it->path()) != ".jm") {
      continue;
    }

    std::string classname = it->path().stem().string();

    if (! (classname == "Car" || classname == "Cyclist" || classname == "Pedestrian" || classname == "Background")) {
      continue;
    }

    printf("Found %s\n", classname.c_str());

    clt::JointModel jm = clt::JointModel::Load(it->path().string().c_str());

    float lp = 0;
    //if (classname == "Background") {
    //  lp = 10;
    //}

    //if (classname == "Car") {
    //  lp = -20;
    //}

    //if (classname == "Cyclist") {
    //  lp = -50;
    //}

    //if (classname == "Pedestrian") {
    //  lp = -70;
    //}

    detector.AddModel(classname, jm, lp);
  }
  printf("Loaded all models\n");

  library::timer::Timer t;
  t.Start();
  detector.Run(scan.GetHits());
  printf("Took %5.3f ms to run detector\n", t.GetMs());

  vw::Viewer v(&args);

  osg::ref_ptr<osgn::PointCloud> pc = new osgn::PointCloud(scan);
  //osg::ref_ptr<osgn::OccGrid> ogn = new osgn::OccGrid(og);
  osg::ref_ptr<osgn::Tracklets> tn = new osgn::Tracklets(&tracklets, frame_num);
  osg::ref_ptr<kog::MapNode> map_node = new kog::MapNode(detector);

  v.AddChild(pc);
  //v.AddChild(ogn);
  v.AddChild(tn);
  v.AddChild(map_node);

  v.Start();

  return 0;
}
