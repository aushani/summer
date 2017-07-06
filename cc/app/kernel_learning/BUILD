cc_binary(
    name = "kernel_learning",
    srcs = glob(["*.cc", "*.h"]),
    copts = ["-pthread"],
    deps = ["@eigen//:eigen", "//library/hilbert_map", "//library/sim_world"]
)
