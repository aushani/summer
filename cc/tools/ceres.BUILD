# not portable, see https://stackoverflow.com/questions/34984290/building-opencv-code-using-bazel

cc_library(
    name = "ceres",
    srcs = glob(["lib/libceres*.a"]),
    hdrs = glob(["include/ceres/**"]),
    visibility = ["//visibility:public"],
    #deps = [],
)
