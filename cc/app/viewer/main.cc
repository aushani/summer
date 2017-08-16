#include <iostream>
#include <osg/ArgumentParser>

#include "library/kitti/velodyne_scan.h"
#include "library/ray_tracing/occ_grid_builder.h"
#include "library/timer/timer.h"

#include "app/viewer/viewer.h"

namespace kt = library::kitti;
namespace rt = library::ray_tracing;

kt::VelodyneScan LoadVelodyneScan(osg::ArgumentParser *args) {
  std::string home_dir = getenv("HOME");
  std::string kitti_log_dir = home_dir + "/data/kittidata/extracted/";
  if (!args->read(std::string("--kitti-log-dir"), kitti_log_dir)) {
    printf("Using default KITTI log dir: %s\n", kitti_log_dir.c_str());
  }

  std::string kitti_log_date = "2011_09_26";
  if (!args->read(std::string("--kitti-log-date"), kitti_log_date)) {
    printf("Using default KITTI date: %s\n", kitti_log_date.c_str());
  }

  int log_num = 18;
  if (!args->read(std::string("--log-num"), log_num)) {
    printf("Using default KITTI log number: %d\n", log_num);
  }

  int frame_num = 0;
  if (!args->read(std::string("--frame-num"), frame_num)) {
    printf("Using default KITTI frame number: %d\n", frame_num);
  }

  char fn[1000];
  sprintf(fn, "%s/%s/%s_drive_%04d_sync/velodyne_points/data/%010d.bin",
      kitti_log_dir.c_str(), kitti_log_date.c_str(), kitti_log_date.c_str(), log_num, frame_num);

  return kt::VelodyneScan(fn);
}

kt::Tracklets LoadTracklets(osg::ArgumentParser *args) {
  std::string home_dir = getenv("HOME");
  std::string kitti_log_dir = home_dir + "/data/kittidata/extracted/";
  if (!args->read(std::string("--kitti-log-dir"), kitti_log_dir)) {
    printf("Using default KITTI log dir: %s\n", kitti_log_dir.c_str());
  }

  std::string kitti_log_date = "2011_09_26";
  if (!args->read(std::string("--kitti-log-date"), kitti_log_date)) {
    printf("Using default KITTI date: %s\n", kitti_log_date.c_str());
  }

  int log_num = 18;
  if (!args->read(std::string("--log-num"), log_num)) {
    printf("Using default KITTI log number: %d\n", log_num);
  }

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

  // handle help text
  // call AFTER init viewer so key bindings have been set
  unsigned int helpType = 0;
  if ((helpType = args.readHelpType())) {
    au->write(std::cout, helpType);
    return EXIT_SUCCESS;
  }

  // Load velodyne scan
  kt::VelodyneScan scan = LoadVelodyneScan(&args);

  int frame_num = 0;
  if (!args.read(std::string("--frame-num"), frame_num)) {
    printf("Using default KITTI frame number: %d\n", frame_num);
  }
  kt::Tracklets tracklets = LoadTracklets(&args);

  // Build occ grid
  rt::OccGridBuilder builder(200000, 0.3, 100.0);
  for (int t_id=0; t_id<tracklets.numberOfTracklets(); t_id++) {
    if (!tracklets.isActive(t_id, frame_num)) {
      continue;
    }

    auto *tt = tracklets.getTracklet(t_id);
    kt::Tracklets::tPose* pose;
    tracklets.getPose(t_id, frame_num, pose);

    printf("Tracklet %d (%s) with size %5.3f x %5.3f x %5.3f at %5.3f, %5.3f, %5.3f, angle %5.3f\n",
           t_id, tt->objectType.c_str(), tt->w, tt->l, tt->h,
           pose->tx, pose->ty, pose->tz + kt::Tracklets::kZOffset, pose->rz);

    builder.ConfigureSize(2.5, 2.5, 2.0);
    builder.SetPose(Eigen::Vector3d(pose->tx, pose->ty, 0), pose->rz);
    break;
  }

  library::timer::Timer t;
  rt::OccGrid og = builder.GenerateOccGrid(scan.GetHits());
  printf("Took %5.3f ms to build occ grid\n", t.GetMs());

  app::viewer::Viewer v(&args);

  v.AddVelodyneScan(scan);
  v.AddOccGrid(og);
  v.AddTracklets(&tracklets, frame_num);

  v.Start();

  return 0;
}
