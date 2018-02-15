cc_library(
    name = "osg_nodes",
    srcs = glob(['*.cc']),
    hdrs = glob(['*.h']),
    visibility = ["//visibility:public"],
    deps = [
            "//library/chow_liu_tree",
            "//library/feature",
            "//library/kitti",
            "//library/ray_tracing",
            "//library/util",
            "@openscenegraph//:openscenegraph",
            "@pcl//:pcl",
           ],
)
