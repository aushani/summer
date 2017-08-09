#include <iostream>
#include <sys/stat.h>
#include <unistd.h>

#include <Eigen/Geometry>
#include <Eigen/Dense>

#include "library/util/angle.h"
#include "library/kitti/tracklets.h"
#include "library/kitti/velodyne_scan.h"

#include "app/kitti/model_observation.h"
#include "app/kitti/model_bank.h"
#include "app/kitti/observation.h"
#include "app/kitti/object_state.h"

namespace kt = library::kitti;
namespace ut = library::util;

using namespace app::kitti;

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

  ModelBank mb;

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
          t_id, tt->objectType.c_str(), pose->tx, pose->ty, pose->tz + z_offset, pose->rx, pose->ry, pose->rz);

      // Find points within this track
      Eigen::Affine3d r = CreateRotationMatrix(pose->rx, pose->ry, pose->rz);
      Eigen::Affine3d t(Eigen::Translation3d(Eigen::Vector3d(pose->tx, pose->ty, pose->tz + z_offset)));
      Eigen::Matrix4d m = (t * r).inverse().matrix();

      auto hits = scans[scan_id].GetHits();

      std::vector<Eigen::Vector3d> hits_t;
      std::vector<Eigen::Vector3d> hits_world_t;
      for (const auto &h : hits) {
        Eigen::Vector4d h_h = h.homogeneous();
        Eigen::Vector3d h_t = (m * h_h).hnormalized();

        if (std::abs(h_t.x()) < tt->l/2 &&
            std::abs(h_t.y()) < tt->w/2 &&
            std::abs(h_t.z()) < tt->h/2) {
          hits_t.push_back(h_t);
          hits_world_t.push_back(h);
        }
      }

      // Test with model?
      ObjectState os(Eigen::Vector3d(pose->tx, pose->ty, pose->tz + z_offset), pose->rz, tt->objectType);
      std::vector<Observation> obs;
      for (const auto &h_t : hits_world_t) {
        obs.emplace_back(h_t);
      }

      mb.MarkObservations(os, obs);

      //for (const auto &x_hit : obs) {
      //  double range = mb.GetModel(os.classname).SampleRange(os, x_hit.theta, x_hit.phi);
      //  printf("for theta = %5.3f, phi = %5.3f, range is %5.3f vs sampled at %5.3f\n",
      //      ut::RadiansToDegrees(x_hit.theta), ut::RadiansToDegrees(x_hit.phi), x_hit.range, range);
      ////  //printf("\t\t (x, y, z) = %5.3f, %5.3f, %5.3f ", x_hit.pos.x(), x_hit.pos.y(), x_hit.pos.z());
      ////  //printf("\t theta = %5.3f, phi = %5.3f\n", x_hit.theta * 180.0 / M_PI, x_hit.phi * 180.0 / M_PI);
      ////  //ModelObservation mo(os, x_hit);
      ////  //printf("\t\tTheta: %07.1f,\t Phi : %07.1f,\t dist_ray: %07.1f,\t dist_z: %07.1f,\t dist_obs: %07.1f \t(in front %s)\n",
      ////  //        mo.theta * 180.0 / M_PI, mo.phi * 180.0 / M_PI, mo.dist_ray, mo.dist_z, mo.dist_obs, mo.in_front ? "T":"F");
      //}

      if (hits_t.size() > 0) {
        sprintf(fn, "track_%03d_frame_%03d.csv", t_id, scan_id);
        std::ofstream t_hits(fn);

        for (const auto &h_t : hits_t) {
          t_hits << h_t.x() << ", " << h_t.y() << ", " << h_t.z() << std::endl;
        }
      }

      if (obs.size() > 0) {
        sprintf(fn, "track_%03d_frame_%03d_synth.csv", t_id, scan_id);
        std::ofstream model_file(fn);

        for (const auto &x_hit : obs) {
          double range = mb.GetModel(os.classname).SampleRange(os, x_hit.theta, x_hit.phi);

          if (range < 100 && range > 0) {
            double x = range * ( cos(x_hit.phi)  * cos(x_hit.theta));
            double y = range * ( cos(x_hit.phi)  * sin(x_hit.theta));
            double z = range * ( sin(x_hit.phi) );
            Eigen::Vector3d h(x, y, z);

            Eigen::Vector4d h_h = h.homogeneous();
            Eigen::Vector3d h_t = (m * h_h).hnormalized();

            model_file << h_t.x() << ", " << h_t.y() << ", " << h_t.z() << std::endl;
          }
        }
      }
    }

    scan_id++;
  }

  mb.PrintStats();
}
