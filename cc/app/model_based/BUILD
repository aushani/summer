cc_binary(
    name = "model_based",
    srcs = glob(["*.cc", "*.h"]),
    linkopts = ["-pthread"],
    deps = ["@eigen//:eigen", "//library/sim_world", "//library/timer"]
)
