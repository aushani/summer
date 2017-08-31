cc_library(
    name = "chow_liu_tree",
    srcs = [
            "joint_model.cc", "joint_model.h",
            "dynamic_clt.cc", "dynamic_clt.h",
           ],
    #srcs = glob(["*.cc", "*.h"]),
    visibility = ["//visibility:public"],
    linkopts = ["-pthread"],
    deps = [
            "//library/ray_tracing",
            "//library/timer",
            "@boost//:boost_serialization",
           ],
)
