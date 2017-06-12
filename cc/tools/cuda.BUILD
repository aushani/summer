cc_library(
    name = "cuda",
    hdrs = glob(["targets/x86_64-linux/include/**"]),
    data = ["bin/nvcc"],
    includes = [
        "targets/x86_64-linux/include",
    ],
    linkopts = [
        "-Wl,-rpath=/usr/local/cuda-8.0/targets/x86_64-linux/lib",
        "-L/usr/local/cuda-8.0/targets/x86_64-linux/lib",
        "-lnppc",
        "-lnppi",
        "-lnpps",
        "-lcufft",
        "-lcudart",
        "-lm",
    ],
    visibility = ["//visibility:public"],
)
