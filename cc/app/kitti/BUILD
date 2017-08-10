cc_library(
    name = "kitti_app_lib",
    srcs = [
            "model_bank.cc", "model_bank.h",
            "model_observation.h",
            "object_state.h",
            "observation.h",
            "ray_model.cc", "ray_model.h",
          ],
    linkopts = ["-lm"],
    deps = [
          "//library/kitti",
          "//library/histogram",
          "//library/util",
          "@eigen//:eigen"
        ]
)

cc_binary(
    name = "build_model_bank",
    srcs = ["build_model_bank.cc"],
    deps = ["kitti_app_lib"],
)

cc_binary(
    name = "test_kitti",
    srcs = ["test_kitti.cc"],
    deps = ["kitti_app_lib", "//library/timer"],
)
