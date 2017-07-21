package(default_visibility = ["//visibility:public"])

cc_library(
    name = "boost_headers",
    hdrs = glob(["boost_1_63_0/boost/**"]),
    defines = [
        "BOOST_ALL_DYN_LINK",
        "BOOST_MPL_CFG_NO_PREPROCESSED_HEADERS",
        "BOOST_MPL_LIMIT_VECTOR_SIZE=50",
    ],
    includes = ["boost_1_63_0"],
)

cc_library(
    name = "boost_serialization",
    srcs = glob(["boost_1_63_0/libs/serialization/src/*.cpp"]),
    hdrs = glob(["boost_1_63_0/libs/serialization/src/*.ipp"]),
    copts = ["-fvisibility=hidden"],
    deps = [":boost_headers"],
)
