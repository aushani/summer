cc_library(
    name = "feature",
    srcs = glob(["*.cc", "*.h"]),
    visibility = ["//visibility:public"],
    linkopts = ["-pthread"],
    deps = [
            "//library/ray_tracing",
            "@boost//:boost_serialization",
           ],
)

