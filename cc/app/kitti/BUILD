cc_library(
    name = "kitti_app_lib",
    srcs = [
            "model_bank.cc", "model_bank.h",
            "model_observation.h",
            "object_state.h",
            "observation.h",
            "ray_model.cc", "ray_model.h",
          ],
    linkopts = ["-lm", "-pthread"],
    deps = [
          "//library/kitti",
          "//library/histogram",
          "//library/util",
          "//library/timer",
          "@eigen//:eigen"
        ]
)

cc_binary(
    name = "build_model_bank",
    srcs = ["build_model_bank.cc"],
    deps = ["kitti_app_lib"],
)

cc_binary(
    name = "blur_model_bank",
    srcs = ["blur_model_bank.cc"],
    deps = ["kitti_app_lib"],
)

cc_binary(
    name = "save_model_bank_coverage",
    srcs = ["save_model_bank_coverage.cc"],
    deps = ["kitti_app_lib"],
)

cc_binary(
    name = "generate_sample_scans",
    srcs = ["generate_sample_scans.cc"],
    deps = ["kitti_app_lib"],
)

cc_binary(
    name = "test_kitti",
    srcs = ["test_kitti.cc"],
    deps = ["kitti_app_lib"],
)
