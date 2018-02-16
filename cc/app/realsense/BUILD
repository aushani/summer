cc_binary(
    name = "realsense-viewer",
    srcs = ["pcd_viewer.cc"],
    deps = [
            "//library/osg_nodes",
            "//library/timer",
            "//library/viewer",
            "@openscenegraph//:openscenegraph",
            "@pcl//:pcl",
           ],
)

cc_binary(
    name = "slam-viewer",
    srcs = ["slam_viewer.cc",
            "node.h", "node.cc",
           ],
    deps = [
            "//library/osg_nodes",
            "//library/timer",
            "//library/viewer",
            "@openscenegraph//:openscenegraph",
            "@pcl//:pcl",
            "@boost//:boost_filesystem",
            "@sophus//:sophus",
           ],
)
