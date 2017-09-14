#include <iostream>

#include <osg/ArgumentParser>

#include "library/kitti/object_label.h"
#include "library/timer/timer.h"
#include "library/util/angle.h"

#include "app/kitti_occ_grids/detector_app.h"

namespace ut = library::util;
namespace dt = library::detector;
namespace kt = library::kitti;
namespace kog = app::kitti_occ_grids;

void WriteDetections(const std::vector<dt::Detection> &detections, const kt::KittiChallengeData &kcd, const char *fn) {
  kt::ObjectLabels labels;

  for (const dt::Detection &d : detections) {
    // This is ugly, but check a few times to make sure we're not on the boundary
    if (!kcd.InCameraView(d.os.x - 1.0, d.os.y + 1.0, 0.0)) {
      continue;
    }

    if (!kcd.InCameraView(d.os.x - 1.0, d.os.y - 1.0, 0.0)) {
      continue;
    }

    if (!kcd.InCameraView(d.os.x + 1.0, d.os.y + 1.0, 0.0)) {
      continue;
    }

    if (!kcd.InCameraView(d.os.x + 1.0, d.os.y - 1.0, 0.0)) {
      continue;
    }

    // Score threshold
    //if (d.log_odds_score < 1) {
    //  continue;
    //}

//ty  tr   o al   bbox 1    2     3      4    dim1  2    3    loc1   2    3    roty
//Car 0.00 0 1.85 387.63 181.54 423.81 203.12 1.67 1.87 3.69 -16.53 2.39 58.49 1.57

    // Project to camera coordinates
    Eigen::Vector3d x_vel(d.os.x, d.os.y, 0.0);
    Eigen::Vector4d x_camera = kcd.GetTcv() * x_vel.homogeneous();

    kt::ObjectLabel label;

    label.type = kt::ObjectLabel::GetType(d.classname.c_str());
    label.alpha = ut::MinimizeAngle(1.57);

    Eigen::Vector2d p_c = kcd.ToCameraPixels(d.os.x, d.os.y, 0);

    label.bbox[0] = p_c.x() - 20;
    label.bbox[1] = p_c.y() - 20;
    label.bbox[2] = p_c.x() + 20;
    label.bbox[3] = p_c.y() + 20;

    label.dimensions[0] = 0;
    label.dimensions[1] = 1.7;
    label.dimensions[2] = 4.5;

    label.location[0] = x_camera.x();
    label.location[1] = 2.3;//x_camera.y();
    label.location[2] = x_camera.z();

    label.rotation_y = ut::MinimizeAngle(1.57);


    label.score = d.log_odds_score;

    labels.push_back(label);
  }

  kt::ObjectLabel::Save(labels, fn);
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

  au->addCommandLineOption("--kitti-challenge-dir <dirname>", "KITTI challenge data directory", "~/data/kitti_challenge/");
  au->addCommandLineOption("--models <dir>", "Models to evaluate", "");
  au->addCommandLineOption("--results <dir>", "Where to save results", "");
  au->addCommandLineOption("--num <num>", "KITTI scan number", "0");

  // handle help text
  // call AFTER init viewer so key bindings have been set
  unsigned int helpType = 0;
  if ((helpType = args.readHelpType())) {
    au->write(std::cout, helpType);
    return EXIT_SUCCESS;
  }

  std::string results_dir = "";
  args.read("--results", results_dir);

  kog::DetectorApp app(&args);

  int frame_at = 0;

  while (app.SetFrame(frame_at)) {
    printf("\n\n---- Frame %6d ----\n\n", frame_at);

    // Process frame
    kt::KittiChallengeData kcd = app.Process();

    // Get detections
    library::timer::Timer t;
    std::vector<dt::Detection> detections = app.GetDetector().GetDetections();
    printf("Took %5.3f ms to get list of detections\n", t.GetMs());

    // Write to file
    char fn[100] = {0};
    sprintf(fn, "%s/%06d.txt", results_dir.c_str(), frame_at);
    WriteDetections(detections, kcd, fn);

    // Next frame
    frame_at++;
  }

  return 0;
}
