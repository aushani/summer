#include "detection_map.h"

#include <thread>

#include <Eigen/Core>

DetectionMap::DetectionMap(double size, double res, const ModelBank &model_bank) :
 size_(size), res_(res), model_bank_(model_bank) {
  auto classes = GetClasses();

  int num_angles = ceil(2*M_PI / angle_res_);

  for (double x = -size; x <= size; x += res) {
    for (double y = -size; y <= size; y += res) {
      for (double angle = 0; angle < 2*M_PI; angle += angle_res_) {
        for (std::string classname : classes) {
          ObjectState s(x, y, angle, classname);

          double p_obj = model_bank_.GetProbObj(classname);
          double l_p = log(p_obj);

          //printf("p_obj = %f\n", p_obj);
          //printf("l_p = %f\n", l_p);

          scores_.insert( std::pair<ObjectState, double>(s, l_p) );
        }
      }
    }
  }
}

std::vector<std::string> DetectionMap::GetClasses() const {
  std::vector<std::string> classes;
  auto models = model_bank_.GetModels();
  for (auto it = models.begin(); it != models.end(); it++) {
    classes.push_back(it->first);
  }

  return classes;
}

double DetectionMap::EvaluateObservationsForState(const std::vector<Eigen::Vector2d> &x_hits, const ObjectState &state) const {
  Eigen::Vector2d x_sensor_object;
  x_sensor_object(0) = state.pos.x;
  x_sensor_object(1) = state.pos.y;

  double object_angle = state.angle;

  return model_bank_.EvaluateObservations(x_sensor_object, object_angle, x_hits, state.classname);
}

void DetectionMap::ProcessObservationsForState(const std::vector<Eigen::Vector2d> &x_hits, const ObjectState &state) {
  double update = EvaluateObservationsForState(x_hits, state);
  //printf("update: %5.3f\n", update);
  //printf("prev score: %5.3f\n", scores_[state]);
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

bool DetectionMap::IsMaxDetection(const ObjectState &state) {
  double val = scores_[state];

  int window_size = 5;

  double x = state.pos.x;
  double y = state.pos.y;

  for (double xm = fmax(-size_, x - window_size*res_); xm <= fmin(size_, x + window_size*res_); xm += res_) {
    for (double ym = fmax(-size_, y - window_size*res_); ym <= fmin(size_, y + window_size*res_); ym += res_) {
      for (double angle = 0; angle < 2*M_PI; angle += angle_res_) {
        for (std::string c : GetClasses()) {
          ObjectState sm(xm, ym, angle, c);

          auto it = scores_.find(sm);

          if (it == scores_.end()) {
            printf("state not in map!\n");
          }

          if (it->second > val) {
            return false;
          }
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

double DetectionMap::GetProb(const ObjectState &os) const {
  auto it_os = scores_.find(os);
  if (it_os == scores_.end()) {
    printf("not found!\n");
    return 0.0f;
  }

  double my_score = it_os->second;
  double denom = 0.0;

  auto it = it_os;
  while (std::abs(it->first.pos.x - it_os->first.pos.x) < 1e-3 &&
         std::abs(it->first.pos.y - it_os->first.pos.y) < 1e-3) {
    double s = it->second - my_score;
    denom += exp(s);
    it++;
  }

  it = it_os;
  it--; // don't double count!
  while (std::abs(it->first.pos.x - it_os->first.pos.x) < 1e-3 &&
         std::abs(it->first.pos.y - it_os->first.pos.y) < 1e-3) {
    double s = it->second - my_score;
    denom += exp(s);
    it--;
  }

  double p = 1.0/denom;
  //printf("my score = %5.3f, denom = %5.3f\n", my_score, denom);
  return p;
}
