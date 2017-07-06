cc_library(
    name = "hilbert_map",
    srcs = glob(["*.cc", "*.h"]),
    copts = ["-x",  "cuda", "-std=c++11"],
    deps = ["//tools:cuda", "//library/geometry"],
    visibility = ["//visibility:public"],
)
