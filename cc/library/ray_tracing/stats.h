#pragma once

// This is gross but lets this play nicely with both cuda and non-cuda compilers
#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif

namespace library {
namespace ray_tracing {

struct Stats {
  float intensity = 0;
  float sum_intensity_var = 0;

  float x = 0;
  float y = 0;
  float z = 0;

  float sum_var_x = 0;
  float sum_var_y = 0;
  float sum_var_z = 0;

  float sum_cov_xy = 0;
  float sum_cov_xz = 0;
  float sum_cov_yz = 0;

  int count = 0;

  CUDA_CALLABLE Stats operator+(const Stats &other) const {
    Stats s;
    s.count = count + other.count;

    s.intensity = CombineMean(intensity, count, other.intensity, other.count);
    s.sum_intensity_var = CombineVariance(intensity, count, sum_intensity_var, other.intensity, other.count, other.sum_intensity_var);

    s.x = CombineMean(x, count, other.x, other.count);
    s.y = CombineMean(y, count, other.y, other.count);
    s.z = CombineMean(z, count, other.z, other.count);

    s.sum_var_x = CombineVariance(x, count, sum_var_x, other.x, other.count, other.sum_var_x);
    s.sum_var_y = CombineVariance(y, count, sum_var_y, other.y, other.count, other.sum_var_y);
    s.sum_var_z = CombineVariance(z, count, sum_var_z, other.z, other.count, other.sum_var_z);

    s.sum_cov_xy = CombineCovariance(sum_cov_xy, x, y, count, other.sum_cov_xy, other.x, other.y, other.count);
    s.sum_cov_xz = CombineCovariance(sum_cov_xz, x, z, count, other.sum_cov_xz, other.x, other.z, other.count);
    s.sum_cov_yz = CombineCovariance(sum_cov_yz, y, z, count, other.sum_cov_yz, other.y, other.z, other.count);

    return s;
  }

  CUDA_CALLABLE static float CombineMean(float avg_a, int count_a, float avg_b, int count_b) {
    return (avg_a * count_a + avg_b * count_b) / (count_a + count_b);
  }

  CUDA_CALLABLE static float CombineVariance(float avg_a, int count_a, float var_a, float avg_b, int count_b, float var_b) {
    // From wikipedia
    // https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    //float delta = avg_b - avg_a;
    //float m_a = var_a * (count_a - 1);
    //float m_b = var_b * (count_b - 1);
    //float M2 = m_a + m_b + (delta * delta) * ((float)count_a * count_b) / (count_a + count_b);
    //return M2 / (count_a + count_b - 1.0);
    float delta = avg_a - avg_b;
    return var_a + var_b + delta*delta*count_a*count_b / (count_a + count_b);
  }

  CUDA_CALLABLE static float CombineCovariance(float cov_a, float x_a, float y_a, int count_a, float cov_b, float x_b, float y_b, int count_b) {
    // From wikipedia
    // https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    float delta_x = x_a - x_b;
    float delta_y = y_a - y_b;
    return cov_a + cov_b + delta_x*delta_y*count_a*count_b / (count_a + count_b);
  }

  CUDA_CALLABLE float GetIntensityCov() const {
    return sum_intensity_var / count;
  }

  CUDA_CALLABLE float GetCovX() const {
    return sum_var_x / count;
  }

  CUDA_CALLABLE float GetCovY() const {
    return sum_var_y / count;
  }

  CUDA_CALLABLE float GetCovZ() const {
    return sum_var_z / count;
  }

  CUDA_CALLABLE float GetCovXY() const {
    return sum_cov_xy / count;
  }

  CUDA_CALLABLE float GetCovXZ() const {
    return sum_cov_xz / count;
  }

  CUDA_CALLABLE float GetCovYZ() const {
    return sum_cov_yz / count;
  }
};

}  // namespace ray_tracing
}  // namespace library
