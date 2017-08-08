#include <iostream>
#include <sys/stat.h>
#include <unistd.h>

#include <Eigen/Geometry>
#include <Eigen/Dense>

#include "library/kitti/tracklets.h"
#include "library/kitti/velodyne_scan.h"

namespace kt = library::kitti;

inline bool FileExists(const char* fn) {
  struct stat buffer;
  return stat(fn, &buffer) == 0;
}

Eigen::Affine3d CreateRotationMatrix(double ax, double ay, double az) {
  Eigen::Affine3d rx(Eigen::AngleAxisd(ax, Eigen::Vector3d(1, 0, 0)));
  Eigen::Affine3d ry(Eigen::AngleAxisd(ay, Eigen::Vector3d(0, 1, 0)));
  Eigen::Affine3d rz(Eigen::AngleAxisd(az, Eigen::Vector3d(0, 0, 1)));
  return rz * ry * rx;
}

int main() {
  printf("Testing KITTI...\n");

  kt::Tracklets t;

  double z_offset = 0.8; // ???

  bool success = t.loadFromFile("/home/aushani/data/kittidata/extracted/2011_09_26/2011_09_26_drive_0001_sync/tracklet_labels.xml");

  printf("Loaded tracklets: %s\n", success ? "SUCCESS":"FAIL");
  printf("Have %d tracklets\n", t.numberOfTracklets());

  for (int i=0; i<t.numberOfTracklets(); i++) {
    auto *tt = t.getTracklet(i);
    printf("Have %s (size %5.3f x %5.3f x %5.3f) for %ld frames\n",
        tt->objectType.c_str(), tt->h, tt->w, tt->l, tt->poses.size());
  }

  std::vector<kt::VelodyneScan> scans;
  int scan_id = 0;
  while (true) {
    char fn[1000];
    sprintf(fn, "/home/aushani/data/kittidata/extracted/2011_09_26/2011_09_26_drive_0001_sync/velodyne_points/data/%010d.bin", scan_id);
    if (!FileExists(fn)) {
      break;
    }

    scans.emplace_back(fn);
    printf("Loaded scan %d, has %ld hits\n", scan_id, scans[scan_id].GetHits().size());

    for (int t_id=0; t_id<t.numberOfTracklets(); t_id++) {
      if (!t.isActive(t_id, scan_id)) {
        continue;
      }
      auto *tt = t.getTracklet(t_id);
      kt::Tracklets::tPose* pose;
      t.getPose(t_id, scan_id, pose);
      printf("\tTrack %d (%s) at %5.3f, %5.3f, %5.3f, rot %5.3f, %5.3f, %5.3f\n",
          t_id, tt->objectType.c_str(), pose->tx, pose->ty, pose->tz, pose->rx, pose->ry, pose->rz);

      // Find points within this track
      Eigen::Affine3d r = CreateRotationMatrix(pose->rx, pose->ry, pose->rz);
      Eigen::Affine3d t(Eigen::Translation3d(Eigen::Vector3d(pose->tx, pose->ty, pose->tz + z_offset)));
      Eigen::Matrix4d m = (t * r).inverse().matrix();

      auto hits = scans[scan_id].GetHits();

      std::vector<Eigen::Vector3d> hits_t;
      for (const auto &h : hits) {
        Eigen::Vector4d h_h = h.homogeneous();
        Eigen::Vector3d h_t = (m * h_h).hnormalized();

        if (std::abs(h_t.x()) < tt->l/2 &&
            std::abs(h_t.y()) < tt->w/2 &&
            std::abs(h_t.z()) < tt->h/2) {
          hits_t.push_back(h_t);
        }
      }

      if (hits_t.size() > 0) {
        sprintf(fn, "track_%03d_frame_%03d.csv", t_id, scan_id);
        std::ofstream t_hits(fn);

        for (const auto &h_t : hits_t) {
          t_hits << h_t.x() << ", " << h_t.y() << ", " << h_t.z() << std::endl;
        }
      }

    }

    scan_id++;
  }
}
