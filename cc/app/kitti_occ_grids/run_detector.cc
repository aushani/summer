#include <iostream>
#include <osg/ArgumentParser>

#include "library/kitti/velodyne_scan.h"
#include "library/osg_nodes/point_cloud.h"
#include "library/osg_nodes/occ_grid.h"
#include "library/osg_nodes/tracklets.h"
#include "library/ray_tracing/occ_grid_builder.h"
#include "library/ray_tracing/dense_occ_grid.h"
#include "library/timer/timer.h"
#include "library/viewer/viewer.h"

#include "app/kitti_occ_grids/detector.h"
#include "app/kitti_occ_grids/map_node.h"
#include "app/kitti_occ_grids/model.h"

namespace kt = library::kitti;
namespace rt = library::ray_tracing;
namespace osgn = library::osg_nodes;
namespace vw = library::viewer;
namespace kog = app::kitti_occ_grids;

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
  au->addCommandLineOption("--model <dir>", "Models to evaluate", "");
  au->addCommandLineOption("--bg-model <dir>", "BG Models to evaluate", "");

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

  std::string model_fn;
  if (!args.read("--model", model_fn)) {
    printf("no model given!\n");
    return EXIT_FAILURE;
  }

  std::string bg_model_fn;
  if (!args.read("--bg-model", bg_model_fn)) {
    printf("no model given!\n");
    return EXIT_FAILURE;
  }

  // Load velodyne scan
  printf("Loading vel\n");
  kt::VelodyneScan scan = LoadVelodyneScan(kitti_log_dir, kitti_log_date, log_num, frame_num);
  printf("Loading tracklets\n");
  kt::Tracklets tracklets = LoadTracklets(kitti_log_dir, kitti_log_date, log_num);

  // Load model
  printf("loading model %s\n", model_fn.c_str());
  kog::Model model    = kog::Model::Load(model_fn.c_str());
  kog::Model bg_model = kog::Model::Load(bg_model_fn.c_str());

  // Build occ grid
  rt::OccGridBuilder builder(200000, 0.3, 100.0);

  library::timer::Timer t;
  rt::OccGrid og = builder.GenerateOccGrid(scan.GetHits());
  printf("Took %5.3f ms to build occ grid\n", t.GetMs());

  t.Start();
  rt::DenseOccGrid dog(og, 50.0, 50.0, 10.0);
  printf("Took %5.3f ms to make dense occ grid\n", t.GetMs());

  kog::Detector detector(og.GetResolution(), 75, 1);

  t.Start();
  detector.Evaluate(dog, model, bg_model);
  printf("Took %5.3f ms to run detector\n", t.GetMs());

  vw::Viewer v(&args);

  //osg::ref_ptr<osgn::PointCloud> pc = new osgn::PointCloud(scan);
  osg::ref_ptr<osgn::OccGrid> ogn = new osgn::OccGrid(og);
  osg::ref_ptr<osgn::Tracklets> tn = new osgn::Tracklets(&tracklets, frame_num);
  osg::ref_ptr<kog::MapNode> map_node = new kog::MapNode(detector);

  //v.AddChild(pc);
  v.AddChild(ogn);
  v.AddChild(tn);
  v.AddChild(map_node);

  v.Start();

  return 0;
}
