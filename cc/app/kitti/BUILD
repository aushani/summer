cc_binary(
    name = "test-kitti",
    srcs = glob(["*.cc", "*.h"]),
    linkopts = ["-lm"],
    deps = [
          "//library/kitti",
          "//library/histogram",
          "//library/util",
          "@eigen//:eigen"
        ]
)
