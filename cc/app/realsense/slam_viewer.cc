#include <iostream>

#include <osg/ArgumentParser>
#include <boost/filesystem.hpp>
#include <pcl/point_types.h>
#include <pcl/common/projection_matrix.h>
#include <pcl/io/pcd_io.h>

#include "library/osg_nodes/point_cloud.h"
#include "library/timer/timer.h"
#include "library/viewer/viewer.h"

#include "app/realsense/node.h"

namespace osgn = library::osg_nodes;
namespace vw = library::viewer;
namespace fs = boost::filesystem;
namespace rs = app::realsense;

typedef struct {
  int idx1;
  int idx2;
  int time;
  Sophus::SE3d pose;
} relpose_t;

relpose_t LoadFactor(char* line) {
  float M[16];
  int idx1, idx2, time, iter;

  sscanf(line, "%d, %d, %d, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %d\n",
     &idx1, &idx2, &time,
     &M[0],
     &M[1],
     &M[2],
     &M[3],
     &M[4],
     &M[5],
     &M[6],
     &M[7],
     &M[8],
     &M[9],
     &M[10],
     &M[11],
     &M[12],
     &M[13],
     &M[14],
     &M[15],
     &iter);

  Eigen::Matrix4d e_t;
  e_t = Eigen::Matrix4d::Zero();
  for (int i=0; i<16; i++) {
    //e_t(i%4, i/4) = M[i];
    e_t(i/4, i%4) = M[i];
  }
  auto s_t = Sophus::SE3d::fitToSE3(e_t);

  relpose_t res;

  res.idx1 = idx1;
  res.idx2 = idx2;
  res.pose = s_t;

  return res;
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

  au->addCommandLineOption("--data-dir <dir>", "PCD Data Dir", "~/qut_data/lab/");
  au->addCommandLineOption("--csv <filename>", "SM Result", "~/qut_data/sparki.csv");

  // handle help text
  // call AFTER init viewer so key bindings have been set
  unsigned int helpType = 0;
  if ((helpType = args.readHelpType())) {
    au->write(std::cout, helpType);
    return EXIT_SUCCESS;
  }

  // Read params
  std::string home_dir = getenv("HOME");
  std::string pcd_dir = home_dir + "/qut_data/lab/";
  if (!args.read("--data-dir", pcd_dir)) {
    printf("Using default PCD file: %s\n", pcd_dir.c_str());
  }

  std::string csv_fn = home_dir + "/qut_data/lab/sparki.csv";
  if (!args.read("--csv", csv_fn)) {
    printf("Using default CSV file: %s\n", csv_fn.c_str());
  }

  // Load transformation
  FILE *f_t = fopen(csv_fn.c_str(), "r");

  char *line = NULL;
  size_t len = 0;
  float M[16];
  int idx1, idx2, time, iter;

  std::vector<Sophus::SE3d> transformations;

  for (int i=0; i<200; i++) {
    transformations.push_back(Sophus::SE3<double>::trans(0, 0, 0));
  }

  while (getline(&line, &len, f_t) != -1) {
    relpose_t res = LoadFactor(line);

    int this_idx = res.idx2;
    int prev_idx = res.idx1;

    auto last = transformations[prev_idx];
    auto next = last * res.pose;

    transformations[this_idx] = next;
  }

  fclose(f_t);

  fs::path dir(pcd_dir);

  vw::Viewer v(&args);
  for (int pcd=85; pcd<96; pcd++) {
    char fn[1000];
    sprintf(fn, "pcd_%06d.pcd", pcd);

    fs::path path = dir / fs::path(fn);
    printf("Loading scan %i\n", pcd);

    pcl::PointCloud<pcl::PointXYZRGB> cloud;
    pcl::io::loadPCDFile<pcl::PointXYZRGB>(path.c_str(), cloud);

    auto t0 = Sophus::SE3<double>::rotX(M_PI/4);

    rs::Node node(cloud, t0 * transformations[pcd]);

    v.AddChild(node.GetOsg());

  }

  v.Start();

  return 0;
}
