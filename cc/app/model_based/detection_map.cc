#include "detection_map.h"

#include <thread>

#include <Eigen/Core>

DetectionMap::DetectionMap(double size, double res, const RayModel &model) :
 size_(size), res_(res), model_(model) {
  for (double x = -size; x <= size; x += res) {
    for (double y = -size; y <= size; y += res) {
      for (int angle_step = 0; angle_step < 8; angle_step++) {
        double angle = angle_step * (2*M_PI/5) / 8;
        ObjectState s(x, y, angle);

        scores_[s] = 0.0;
      }
    }
  }
}

void DetectionMap::ProcessObservationsForState(const std::vector<Eigen::Vector2d> &x_hits, const ObjectState &state) {
  Eigen::Vector2d x_sensor_object;
  x_sensor_object(0) = state.pos.x;
  x_sensor_object(1) = state.pos.y;

  double object_angle = state.angle;
  double update = model_.EvaluateObservations(x_sensor_object, object_angle, x_hits);

  scores_[state] += update;
}

void DetectionMap::ProcessObservationsWorker(const std::vector<Eigen::Vector2d> &x_hits, std::deque<ObjectState> *states, std::mutex *mutex) {
  bool done = false;

  while (!done) {
    mutex->lock();
    if (states->empty()) {
      done = true;
      mutex->unlock();
    } else {
      ObjectState state = states->front();
      states->pop_front();
      mutex->unlock();
      ProcessObservationsForState(x_hits, state);
    }
  }
}

void DetectionMap::ProcessObservations(const std::vector<ge::Point> &hits) {
  std::vector<Eigen::Vector2d> x_hits;
  for (auto h : hits) {
    Eigen::Vector2d hit;
    hit << h.x, h.y;
    x_hits.push_back(hit);
  }

  std::deque<ObjectState> states;
  std::mutex mutex;

  for (auto it = scores_.begin(); it != scores_.end(); it++) {
    states.push_back(it->first);
  }

  int num_threads = 64;
  std::vector<std::thread> threads;
  for (int i=0; i<num_threads; i++) {
    threads.push_back(std::thread(&DetectionMap::ProcessObservationsWorker, this, x_hits, &states, &mutex));
  }

  for (auto &thread : threads) {
    thread.join();
  }

  printf("Done\n");
}

double DetectionMap::Lookup(const ge::Point &p, double angle) {
  ObjectState s(p.x, p.y, angle);

  double score = scores_[s];

  if (score < -100)
    return 0.0f;
  if (score > 100)
    return 1.0f;
  return 1/(1+exp(-score));
}

bool DetectionMap::IsMaxDetection(const ObjectState &state) {
  double val = scores_[state];

  int window_size = 5;

  double x = state.pos.x;
  double y = state.pos.y;

  for (double xm = fmax(-size_, x - window_size*res_); xm <= fmin(size_, x + window_size*res_); xm += res_) {
    for (double ym = fmax(-size_, y - window_size*res_); ym <= fmin(size_, y + window_size*res_); ym += res_) {
      for (int angle_step = 0; angle_step < 8; angle_step++) {
        double am = angle_step * (2*M_PI/5) / 8;

        ObjectState sm(xm, ym, am);

        if (scores_.count(sm) == 0) {
          printf("state not in map!\n");
        }

        if (scores_[sm] > val) {
          return false;
        }
      }
    }
  }

  return true;
}

void DetectionMap::GetMaxDetectionsWorker(std::deque<ObjectState> *states, std::map<ObjectState, double> *result, std::mutex *mutex) {
  bool done = false;

  std::map<ObjectState, double> thread_result;

  while (!done) {
    mutex->lock();
    if (states->empty()) {
      done = true;
      mutex->unlock();
    } else {
      ObjectState state = states->front();
      states->pop_front();
      mutex->unlock();

      if (IsMaxDetection(state)) {
        thread_result[state] = scores_[state];
      }
    }
  }

  if (thread_result.size() > 0) {
    mutex->lock();
    for (auto it = thread_result.begin(); it != thread_result.end(); it++) {
      (*result)[it->first] = it->second;
    }
    mutex->unlock();
  }
}

std::map<ObjectState, double> DetectionMap::GetMaxDetections(double thresh_score) {
  std::map<ObjectState, double> result;

  std::deque<ObjectState> states;
  std::mutex mutex;

  for (auto it = scores_.begin(); it != scores_.end(); it++) {
    if (it->second > thresh_score) {
      states.push_back(it->first);
    }
  }

  int num_threads = 4;
  std::vector<std::thread> threads;
  for (int i=0; i<num_threads; i++) {
    threads.push_back(std::thread(&DetectionMap::GetMaxDetectionsWorker, this, &states, &result, &mutex));
  }

  for (auto &thread : threads) {
    thread.join();
  }

  return result;
}

const std::map<ObjectState, double>& DetectionMap::GetScores() const {
  return scores_;
}
