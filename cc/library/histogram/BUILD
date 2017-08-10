cc_library(
    name = "histogram",
    srcs = glob(["*.cc", "*.h"]),
    visibility = ["//visibility:public"],
    deps = ["@boost//:boost_serialization"],
)

cc_test(
    name = "histogram_test",
    size = "small",
    srcs = glob(["test/**/*.cc"]),
    deps = [
            ":histogram",
            "@boost//:boost_test",
           ],
)
