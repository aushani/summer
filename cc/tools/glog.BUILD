# not portable, see https://stackoverflow.com/questions/34984290/building-opencv-code-using-bazel

cc_library(
    name = "glog",
    srcs = glob(["lib/libglog.so"]),
    hdrs = glob(["include/glog/**"]),
    visibility = ["//visibility:public"],
    #deps = [],
)
