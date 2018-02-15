cc_binary(
    name = "realsense-viewer",
    srcs = glob(["pcd_viewer.cc"]),
    deps = [
            "//library/osg_nodes",
            "//library/timer",
            "//library/viewer",
            "@openscenegraph//:openscenegraph",
            "@pcl//:pcl",
           ],
)
