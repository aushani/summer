cc_library(
    name = "pcl",
    srcs = glob(["lib/libpcl*.so"]),
    hdrs = glob(["include/pcl-1.8/**"]),
    linkopts = [ ],
    visibility = ["//visibility:public"],
    strip_include_prefix = "include/pcl-1.8",
    deps = [
      "@boost//:boost_system",
    ],
)
