#include "app/kitti_occ_grids/trainer.h"

#include "library/viewer/viewer.h"

namespace vw = library::viewer;
namespace kog = app::kitti_occ_grids;

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
  au->setDescription(args.getApplicationName());

  au->addCommandLineOption("--save-dir <dir>", "Save dir", "");
  au->addCommandLineOption("--load-dir <dir>", "Load dir", "");
  au->addCommandLineOption("--epoch <int>", "Starting Epoch", "");
  au->addCommandLineOption("--frame <int>", "Starting Frame", "");

  // Start viewer
  std::shared_ptr<vw::Viewer> viewer = std::make_shared<vw::Viewer>(&args);

  // handle help text
  // call AFTER init viewer so key bindings have been set
  unsigned int helpType = 0;
  if ((helpType = args.readHelpType())) {
    au->write(std::cout, helpType);
    return EXIT_SUCCESS;
  }

  std::string save_dir = "/home/aushani/data/trainer/";
  if (!args.read("--save-dir", save_dir)) {
    printf("Using default save dir: %s\n", save_dir.c_str());
  }

  kog::Trainer trainer(save_dir.c_str());
  trainer.SetViewer(viewer);

  std::string load_dir = "";
  if (args.read("--load-dir", load_dir)) {
    printf("Loading models from %s\n", load_dir.c_str());
    trainer.LoadFrom(load_dir);
  }

  int epoch = 0;
  args.read("--epoch", epoch);

  int frame = 0;
  args.read("--frame", frame);

  printf("Starting from epoch %d and frame %d\n", epoch, frame);

  trainer.RunBackground(epoch, frame);

  viewer->Start();

  return 0;
}
