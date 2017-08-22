#include <algorithm>
#include <iostream>

#include <osg/ArgumentParser>

#include "library/osg_nodes/occ_grid.h"
#include "library/timer/timer.h"
#include "library/viewer/viewer.h"

#include "app/kitti_occ_grids/chow_lui_tree.h"
#include "app/kitti_occ_grids/chow_lui_tree_node.h"
#include "app/kitti_occ_grids/model.h"
#include "app/kitti_occ_grids/model_node.h"

namespace osgn = library::osg_nodes;
namespace rt = library::ray_tracing;
namespace vw = library::viewer;

namespace kog = app::kitti_occ_grids;

class MyKeyboardEventHandler : public osgGA::GUIEventHandler {
 protected:
  vw::Viewer *viewer_;

  kog::JointModel jm_;
  kog::ChowLuiTree clt_;

 public:
  MyKeyboardEventHandler(vw::Viewer *v, const std::string &fn_clt, const std::string &fn_jm) :
   osgGA::GUIEventHandler(), viewer_(v), jm_(kog::JointModel::Load(fn_jm.c_str())), clt_(kog::ChowLuiTree::Load(fn_clt.c_str())) {
    MakeSample();
  }

  void MakeSample() {
    library::timer::Timer t;

    // Get tree edges
    std::vector<kog::ChowLuiTree::Edge> edges(clt_.GetEdges());
    std::sort(edges.begin(), edges.begin() + edges.size());

    // Random sampling
    std::uniform_real_distribution<double> rand_unif(0.0, 1.0);
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine rand_engine(seed);

    std::map<rt::Location, bool> sample_og;

    while (!edges.empty()) {
      // Find an edge we can process
      auto it = edges.begin();
      bool found = false;
      bool flip = false;
      for ( ; it != edges.end(); it++) {
        if (sample_og.count(it->loc1) > 0) {
          found = true;
          flip = false;
          break;
        }

        if (sample_og.count(it->loc2) > 0) {
          found = true;
          flip = true;
          break;
        }
      }

      // We need to sample
      if (!found) {
        it = edges.begin();

        // Sample loc1
        int c_t = jm_.GetCount(it->loc1, true);
        int c_f = jm_.GetCount(it->loc1, false);
        double p = c_t / (static_cast<double>(c_t + c_f));

        bool occu = rand_unif(rand_engine) < p;
        sample_og[it->loc1] = occu;
        printf("\t marginal\n");

        found = true;
        flip = false;
      }

      kog::ChowLuiTree::Edge edge = *it;
      auto loc1 = (!flip) ? edge.loc1 : edge.loc2;
      auto loc2 = (!flip) ? edge.loc2 : edge.loc1;

      //printf("Edge %d, %d, %d <--> %d, %d, %d\n",
      //    loc1.i, loc1.j, loc1.k,
      //    loc2.i, loc2.j, loc2.k);

      // Sample loc2 based on loc1
      BOOST_ASSERT(sample_og.count(loc1) > 0);
      BOOST_ASSERT(sample_og.count(loc2) == 0);

      int c_t = jm_.GetCount(loc1, loc2, sample_og[loc1], true);
      int c_f = jm_.GetCount(loc1, loc2, sample_og[loc1], false);

      if (c_t < 0 || c_f < 0) {
        printf("Out of range????\n");
      }

      if (c_t == 0 && c_f == 0) {
        printf("Weird, counts are 0, mi = %f, %f\n", jm_.ComputeMutualInformation(loc1, loc2), edge.weight);
        printf("num obs: %d\n", jm_.GetNumObservations(loc1, loc2));

        printf("%d %d %d <-> %d %d %d, mutual information %5.3f\n", loc1.i, loc1.j, loc1.k, loc2.i, loc2.j, loc2.k, edge.weight);
        printf("\t       F      T\n");
        printf("\t   ------------\n");
        printf("\t F | %04d  %04d\n", jm_.GetCount(loc1, loc2, false, false), jm_.GetCount(loc1, loc2, false, true));
        printf("\t T | %04d  %04d\n", jm_.GetCount(loc1, loc2, true, false), jm_.GetCount(loc1, loc2, true, true));
        printf("\n");
      }

      BOOST_ASSERT(c_t + c_f > 0);
      double p = c_t / (static_cast<double>(c_t + c_f));

      bool occu = rand_unif(rand_engine) < p;
      sample_og[loc2] = occu;

      // Remove edge that we processed
      edges.erase(it);
    }

    // Now we have a sample
    printf("Took %5.3f sec to make sample\n", t.GetSeconds());

    // Spoof an occ grid
    std::vector<rt::Location> locs;
    std::vector<float> los;

    for (auto it : sample_og) {
      locs.push_back(it.first);
      los.push_back(it.second ? 1.0:-1.0);
    }

    rt::OccGrid og(locs, los, clt_.GetResolution());
    osg::ref_ptr<osgn::OccGrid> ogn = new osgn::OccGrid(og);

    viewer_->RemoveAllChildren();
    viewer_->AddChild(ogn);
  }


  /**
      OVERRIDE THE HANDLE METHOD:
      The handle() method should return true if the event has been dealt with
      and we do not wish it to be handled by any other handler we may also have
      defined. Whether you return true or false depends on the behaviour you
      want - here we have no other handlers defined so return true.
  **/
  virtual bool handle(const osgGA::GUIEventAdapter& ea, osgGA::GUIActionAdapter& aa, osg::Object* obj,
                      osg::NodeVisitor* nv) {
    switch (ea.getEventType()) {
      case osgGA::GUIEventAdapter::KEYDOWN: {
        switch (ea.getKey()) {
          case 'r':
            MakeSample();
            return true;
        }

        default:
          return false;
      }
    }
  }
};

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

  au->addCommandLineOption("--clt <dirname>", "CLT Filename", "");
  au->addCommandLineOption("--jm <dirname>", "Model Filename", "");

  // handle help text
  // call AFTER init viewer so key bindings have been set
  unsigned int helpType = 0;
  if ((helpType = args.readHelpType())) {
    au->write(std::cout, helpType);
    return EXIT_SUCCESS;
  }


  std::string fn_clt;
  if (!args.read("--clt", fn_clt)) {
    printf("No CLT file to render!\n");
    return 1;
  }

  std::string fn_jm;
  if (!args.read("--jm", fn_jm)) {
    printf("No model file to render!\n");
    return 1;
  }

  vw::Viewer v(&args);
  v.AddHandler(new MyKeyboardEventHandler(&v, fn_clt, fn_jm));

  v.Start();

  return 0;
}
