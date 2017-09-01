cc_library(
    name = "chow_liu_tree",
    srcs = glob(["*.cc", "*.h"]),
    visibility = ["//visibility:public"],
    linkopts = ["-pthread"],
    deps = [
            "//library/ray_tracing",
            "//library/timer",
            "@boost//:boost_serialization",
           ],
)
