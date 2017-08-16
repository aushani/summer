cc_library(
    name = "ray_tracing",
    srcs = glob(['*.cc']),
    hdrs = glob(["*.h"]),
    visibility = ["//visibility:public"],
    copts = ["-x",  "cuda", "-std=c++11"],
    deps = [
            "//tools:cuda",
            "@eigen//:eigen",
            "@boost//:boost_headers",
           ],
)
