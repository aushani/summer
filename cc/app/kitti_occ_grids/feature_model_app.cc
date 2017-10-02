#include "app/kitti_occ_grids/feature_model_app.h"

#include <boost/assert.hpp>

#include "library/feature/feature_model.h"
#include "library/osg_nodes/occ_model.h"

namespace osgn = library::osg_nodes;

namespace app {
namespace kitti_occ_grids {

FeatureModelApp::FeatureModelApp(osg::ArgumentParser *args) {
  osg::ApplicationUsage* au = args->getApplicationUsage();

  au->addCommandLineOption("--mb <dirname>", "Model Bank Filename", "");

  // Load model bank
  std::string fn;
  if (!args->read("--mb", fn)) {
    printf("Error! Need file to render! (--mb) \n");
    BOOST_ASSERT(false);
  }
  model_bank_ = std::make_shared<ft::ModelBank>(ft::ModelBank::Load(fn.c_str()));

  classnames_ = model_bank_->GetClasses();

  // Make viewer
  viewer_ = std::make_shared<vw::Viewer>(args);
  viewer_->AddHandler(this);

  // Render
  RenderBackground(classnames_[classname_at_], angle_bin_at_);
}

void FeatureModelApp::Run() {
  viewer_->Start();
}

bool FeatureModelApp::handle(const osgGA::GUIEventAdapter& ea, osgGA::GUIActionAdapter& aa) {
  if (ea.getEventType() == osgGA::GUIEventAdapter::KEYDOWN) {
    if (ea.getKey() == 'c') {
      classname_at_++;
      classname_at_ %= classnames_.size();
      const std::string &classname = classnames_[classname_at_];
      angle_bin_at_ %= model_bank_->GetNumAngleBins(classname);

      RenderBackground(classname, angle_bin_at_);

      return true;
    }

    if (ea.getKey() == 'a') {
      const std::string &classname = classnames_[classname_at_];
      angle_bin_at_++;
      angle_bin_at_ %= model_bank_->GetNumAngleBins(classname);

      RenderBackground(classname, angle_bin_at_);

      return true;
    }
  }

  return false;
}

void FeatureModelApp::RenderBackground(const std::string &classname, int angle_bin) {
  if (render_thread_) {
    render_thread_->join();
  }

  render_thread_ = std::make_shared<std::thread>(&FeatureModelApp::Render, this, classname, angle_bin);

}

void FeatureModelApp::Render(const std::string &classname, int angle_bin) {
  printf("Rendering %s (angle bin %d)\n", classname.c_str(), angle_bin);

  ft::FeatureModel fm = model_bank_->GetFeatureModel(classname, angle_bin);

  osg::ref_ptr<osgn::OccModel> node = new osgn::OccModel(fm);

  printf("remove\n");
  viewer_->RemoveAllChildren();

  printf("add\n");
  viewer_->AddChild(node);

  printf("done\n");
}

} // namespace viewer
} // namespace app
