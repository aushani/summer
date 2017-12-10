# not portable, see https://stackoverflow.com/questions/34984290/building-opencv-code-using-bazel

cc_library(
    name = "suitesparse",
    srcs = [
             "lib/x86_64-linux-gnu/libcholmod.so",
             "lib/x86_64-linux-gnu/libcsparse.so",
             "lib/x86_64-linux-gnu/libcxsparse.so",
           ],
    hdrs = glob(["include/suitesparse/**"]),
    visibility = ["//visibility:public"],
    deps = [
             "@glog//:glog",
           ],
)
