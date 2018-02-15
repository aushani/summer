#include <iostream>
#include <osg/ArgumentParser>

#include <pcl/point_types.h>
#include <pcl/common/projection_matrix.h>
#include <pcl/io/pcd_io.h>

#include "library/osg_nodes/point_cloud.h"
#include "library/timer/timer.h"
#include "library/viewer/viewer.h"

namespace osgn = library::osg_nodes;
namespace vw = library::viewer;

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

  au->addCommandLineOption("--pcd <filename>", "PCD Filename", "~/qut_data/lab/pcd_000001.pcd");

  // handle help text
  // call AFTER init viewer so key bindings have been set
  unsigned int helpType = 0;
  if ((helpType = args.readHelpType())) {
    au->write(std::cout, helpType);
    return EXIT_SUCCESS;
  }

  // Read params
  std::string home_dir = getenv("HOME");
  std::string pcd_fn = home_dir + "/qut_data/lab/pcd_000001.pcd";
  if (!args.read("--pcd-log-dir", pcd_fn)) {
    printf("Using default PCD file: %s\n", pcd_fn.c_str());
  }

  // Load PCD scan
  printf("Loading pcd\n");
  pcl::PointCloud<pcl::PointXYZRGB> cloud;
  pcl::io::loadPCDFile<pcl::PointXYZRGB>(pcd_fn.c_str(), cloud);

  vw::Viewer v(&args);

  osg::ref_ptr<osgn::PointCloud> pc = new osgn::PointCloud(cloud);

  v.AddChild(pc);

  v.Start();

  return 0;
}
