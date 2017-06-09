cc_binary(
    name = "process-file",
    srcs = ["process-file.cc", "hilbert_map.cu.cc", "hilbert_map.h"],
    copts = ["-DuseCUDA"],
    linkopts = ["-L/usr/local/cuda/lib64 -L/usr/local/cuda/lib -lcuda -lcudart"],
    deps = ["@eigen//:eigen",
            "@cuda//:main"],
)
