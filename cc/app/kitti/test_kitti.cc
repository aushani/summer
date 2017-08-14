#include "library/kitti/tracklets.h"
#include "library/kitti/velodyne_scan.h"
#include "library/timer/timer.h"

#include "app/kitti/model_bank.h"
#include "app/kitti/detection_map.h"

namespace kt = library::kitti;

using namespace app::kitti;

constexpr double kZOffset = 0.8; // ???

void SaveResult(const DetectionMap &detection_map, const std::string &cn, const char *fn) {
  std::ofstream res_file(fn);

  printf("Saving %s map...\n", cn.c_str());

  auto map = detection_map.GetScores();
  for (auto it = map.begin(); it != map.end(); it++) {
    const ObjectState &os = it->first;

    // Check class
    if (os.classname != cn) {
      continue;
    }

    float x = os.pos(0);
    float y = os.pos(1);
    float z = os.pos(2);

    double angle = os.theta;

    double score = detection_map.GetScore(os);
    //double logodds = detection_map.GetLogOdds(os);
    double prob = detection_map.GetProb(os);

    double p = prob;
    if (p < 1e-16)
      p = 1e-16;
    if (p > (1 - 1e-16))
      p = 1 - 1e-16;
    double logodds = -log(1.0/p - 1);

    res_file << x << "," << y << "," << z << "," << angle << "," << score << "," << logodds << "," << prob << std::endl;
  }
  res_file.close();
}

int main(int argc, char** argv) {
  printf("Test KITTI...\n");

  if (argc < 4) {
    printf("Usage: test_kitti model_bank_file log_num frame_num\n");
    return 1;
  }

  const char *model_bank_file = argv[1];
  int log_num = atoi(argv[2]);
  int frame_num = atoi(argv[3]);

  library::timer::Timer t;
  ModelBank mb = ModelBank::LoadModelBank(model_bank_file);
  printf("Took %5.3f sec to load model bank\n", t.GetMs()/1e3);

  // Load Tracklets
  char fn[1000];
  sprintf(fn, "/home/aushani/data/kittidata/extracted/2011_09_26/2011_09_26_drive_%04d_sync/tracklet_labels.xml", log_num);
  kt::Tracklets tracklets;
  if (!tracklets.loadFromFile(fn)) {
    printf("Could not load tracklets from %s\n", fn);
    return 1;
  }

  // Tracklets stats
  for (int i=0; i<tracklets.numberOfTracklets(); i++) {
    auto *tt = tracklets.getTracklet(i);
    printf("Have %s (size %5.3f x %5.3f x %5.3f) for %ld frames starting at %d\n",
        tt->objectType.c_str(), tt->h, tt->w, tt->l, tt->poses.size(), tt->first_frame);
  }

  // Load velodyne scan
  sprintf(fn, "/home/aushani/data/kittidata/extracted/2011_09_26/2011_09_26_drive_%04d_sync/velodyne_points/data/%010d.bin", log_num, frame_num);
  kt::VelodyneScan scan(fn);
  printf("Loaded scan %d, has %ld hits\n", frame_num, scan.GetHits().size());

  t.Start();
  std::vector<Observation> obs;
  for (const auto &x_hit : scan.GetHits()) {
    obs.emplace_back(x_hit);
  }
  printf("Took %5.3f ms to make observations\n", t.GetMs());

  // Go through tracklets and see which ones have hits
  for (int t_id=0; t_id<tracklets.numberOfTracklets(); t_id++) {
    if (!tracklets.isActive(t_id, frame_num)) {
      continue;
    }

    auto *tt = tracklets.getTracklet(t_id);
    kt::Tracklets::tPose* pose;
    tracklets.getPose(t_id, frame_num, pose);

    printf("Tracklet %d (%s) at %5.3f, %5.3f, %5.3f, angle %5.3f\n",
           t_id, tt->objectType.c_str(), pose->tx, pose->ty, pose->tz + kZOffset, pose->rz);

    ObjectState os(Eigen::Vector3d(pose->tx, pose->ty, pose->tz + kZOffset), pose->rz, tt->objectType);

    // Convert to ModelObservation's
    t.Start();
    std::vector<ModelObservation> mos = ModelObservation::MakeModelObservations(os, obs, mb.GetMaxSizeXY(), mb.GetMaxSizeZ());
    printf("Took %5.3f ms to make model observations\n", t.GetMs());

    auto models = mb.GetModels();
    std::string best_class;
    double best_score = -999999;

    for (auto it = models.begin(); it != models.end(); it++) {
      const auto& model = it->second;
      library::timer::Timer t;
      double log_prob = model.EvaluateObservations(mos);

      if (log_prob > best_score && log_prob < 0) {
        best_class = it->first;
        best_score = log_prob;
      }

      printf("\tModel %10s \t --> %+07.5f \t (%5.3f ms)\n", it->first.c_str(), log_prob, t.GetMs());
    }
    printf("  Best %s vs actual %s\n", best_class.c_str(), os.classname.c_str());
  }

  DetectionMap detection_map(25.0, 2.0, mb);

  t.Start();
  detection_map.ProcessScan(scan);
  printf("Took %5.3f sec to process scan\n", t.GetSeconds());

  auto classes = mb.GetClasses();
  for (const std::string &cn : classes) {
    sprintf(fn, "result_%s.csv", cn.c_str());
    SaveResult(detection_map, cn, fn);
  }
}
