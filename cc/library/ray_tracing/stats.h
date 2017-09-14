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
  float intensity_var = 0;

  int count = 0;

  CUDA_CALLABLE Stats operator+(const Stats &other) const {
    Stats s;
    s.count = count + other.count;
    s.intensity = (count*intensity + other.count*other.intensity) / s.count;
    s.intensity_var = CombineVariance(intensity, count, intensity_var, other.intensity, other.count, other.intensity_var);

    return s;
  }

  CUDA_CALLABLE static float CombineVariance(float avg_a, int count_a, float var_a, float avg_b, int count_b, float var_b) {
    // From wikipedia
    float delta = avg_b - avg_a;
    float m_a = var_a * (count_a - 1);
    float m_b = var_b * (count_b - 1);
    float M2 = m_a + m_b + (delta * delta) * ((float)count_a * count_b) / (count_a + count_b);
    return M2 / (count_a + count_b - 1.0);
  }
};

}  // namespace ray_tracing
}  // namespace library
