cc_library(
    name = "detector_cuda",
    srcs = glob(['*.cu.cc']),
    hdrs = glob(["*.h"]),
    visibility = ["//visibility:public"],
    copts = ["-x",  "cuda", "-std=c++11"],
    deps = [
            "//tools:cuda",
            "@eigen//:eigen",
            "@boost//:boost_headers",
            "//library/timer",
            "//library/chow_liu_tree",
            "//library/ray_tracing",
           ],
)
