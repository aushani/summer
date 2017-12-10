# not portable, see https://stackoverflow.com/questions/34984290/building-opencv-code-using-bazel

cc_library(
    name = "ceres",
    linkstatic = 1,
    srcs = glob(["lib/libceres.a"]),
    hdrs = glob(["include/ceres/**"]),
    visibility = ["//visibility:public"],
    linkopts = [
                "-lpthread",
                "-fopenmp",
                "-llapack",
                "-lblas"
               ],
    deps = [
             "@glog//:glog",
             "@suitesparse//:suitesparse",
           ],
)
