cc_library(
    name = "sim_world",
    srcs = glob(["*.cc", "*.h"]),
    visibility = ["//visibility:public"],
    copts = ["-pthread"],
    linkopts = ["-lm", "-pthread"],
    deps = ["@eigen//:eigen", "//library/geometry"]
)
