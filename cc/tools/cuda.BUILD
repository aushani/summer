cc_library(
    name = "cuda",
    hdrs = glob(["include/**"]),
    data = ["bin/nvcc"],
    includes = [
        "include",
    ],
    linkopts = [
        "-Wl,-rpath=/usr/local/cuda/lib",
        "-L/usr/local/cuda/lib",
        "-lnppc",
        "-lnppi",
        "-lnpps",
        "-lcufft",
        "-lcudart",
        "-lm",
    ],
    visibility = ["//visibility:public"],
)
