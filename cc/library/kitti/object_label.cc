#include "library/kitti/object_label.h"

#include <iostream>

namespace library {
namespace kitti {

ObjectLabels ObjectLabel::Load(const char *fn) {
  ObjectLabels labels;

  FILE *fp = fopen(fn, "r");

  char type[100];
  char *line = NULL;
  size_t len = 0;
  while (getline(&line, &len, fp) != -1) {
    printf("read line: <%s>\n", line);
    ObjectLabel label;
    sscanf(line, "%s %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n",
        type,
        &label.truncated,
        &label.occluded,
        &label.alpha,
        &label.bbox[0],
        &label.bbox[1],
        &label.bbox[2],
        &label.bbox[3],
        &label.dimensions[0],
        &label.dimensions[1],
        &label.dimensions[2],
        &label.location[0],
        &label.location[1],
        &label.location[2],
        &label.rotation_y);

    label.type = GetType(type);

    labels.push_back(label);
  }

  fclose(fp);

  return labels;
}

void ObjectLabel::Save(const ObjectLabels &labels, const char *fn) {
  FILE *fp = fopen(fn, "w");

  for (const auto &label : labels) {
    fprintf(fp, "%s %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n",
        GetString(label.type),
        label.truncated,
        label.occluded,
        label.alpha,
        label.bbox[0],
        label.bbox[1],
        label.bbox[2],
        label.bbox[3],
        label.dimensions[0],
        label.dimensions[1],
        label.dimensions[2],
        label.location[0],
        label.location[1],
        label.location[2],
        label.rotation_y,
        label.score);

  }

  fclose(fp);
}

ObjectLabel::Type ObjectLabel::GetType(const char *type) {
  size_t count = strlen(type);
  if (strncmp(type, "Car", count) == 0) {
    return CAR;
  } else if (strncmp(type, "Van", count) == 0) {
    return VAN;
  } else if (strncmp(type, "Truck", count) == 0) {
    return TRUCK;
  } else if (strncmp(type, "Pedestrian", count) == 0) {
    return PEDESTRIAN;
  } else if (strncmp(type, "Person_sitting", count) == 0) {
    return PERSON_SITTING;
  } else if (strncmp(type, "Cyclist", count) == 0) {
    return CYCLIST;
  } else if (strncmp(type, "Tram", count) == 0) {
    return TRAM;
  } else if (strncmp(type, "Misc", count) == 0) {
    return MISC;
  } else if (strncmp(type, "DontCare", count) == 0) {
    return DONT_CARE;
  } else {
    BOOST_ASSERT(false);
    return DONT_CARE;
  }
}

const char* ObjectLabel::GetString(const Type &type) {
  switch(type) {
    case CAR:
      return "Car";
    case VAN:
      return "Van";
    case TRUCK:
      return "Truck";
    case PEDESTRIAN:
      return "Pedestrian";
    case PERSON_SITTING:
      return "Person_sitting";
    case CYCLIST:
        return "Cyclist";
    case TRAM:
      return "Tram";
    case MISC:
      return "Misc";
    case DONT_CARE:
      return "DontCare";
  }

  return "DontCare";
}

Eigen::Matrix4d ObjectLabel::LoadVelToCam(const char *fn) {
  FILE *f_cc = fopen(fn, "r");

  const char *header = "Tr_velo_to_cam: ";
  char *line = NULL;
  size_t len = 0;
  double R[9];
  double t[3];
  while (getline(&line, &len, f_cc) != -1) {
    if (strncmp(header, line, strlen(header)) == 0) {
      sscanf(&line[strlen(header)],
          "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n",
          &R[0], &R[1], &R[2], &R[3], &R[4], &R[5], &R[6], &R[7], &R[8],
          &t[9], &t[10], &t[11]);
    }
  }

  fclose(f_cc);

  Eigen::Matrix4d T_cv;

  T_cv.setZero();
  for (int i=0; i<3; i++) {
    for (int j=0; j<3; j++) {
      T_cv(i, j) = R[i*3 + j];
    }
    T_cv(i, 3) = t[i];
  }
  T_cv(3, 3) = 1;

  return T_cv;
}

} // namespace kitti
} // namespace library
