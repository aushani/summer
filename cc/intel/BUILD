cc_binary(
    name = "process-file",
    #srcs = ["process-file.cc", "hilbert_map.cu.cc", "hilbert_map.h"],
    srcs = glob(["*.cc", "*.h"]),#["process-file.cc", "hilbert_map.cu.cc", "hilbert_map.h"],
    copts = ["-x",  "cuda", "-std=c++11"],
    deps = ["//tools:cuda"]
)
