cc_library(
    name = "detection_mapping_cuda",
    srcs = glob(['*.cu.cc']),
    hdrs = glob(["*.h"]),
    visibility = ["//visibility:public"],
    copts = ["-x",  "cuda", "-std=c++11"],
    deps = [
            "//tools:cuda",
            "@eigen//:eigen",
            "@boost//:boost_headers",
            "//library/timer",
           ],
)

cc_library(
    name = "detection_mapping",
    srcs = glob(['*.cc'], exclude=["*.cu.cc"]),
    hdrs = glob(["*.h"]),
    visibility = ["//visibility:public"],
    linkopts = ["-lpthread"],
    deps = [
            ":detection_mapping_cuda",
            "@eigen//:eigen",
            "@boost//:boost_headers",
            "@boost//:boost_serialization",
            "//library/timer",
           ],
)
