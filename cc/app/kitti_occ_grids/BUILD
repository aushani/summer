cc_library(
    name = "kitti_occ_grids_lib",
    srcs = [
            "model.cc", "model.h",
            "detector.cc", "detector.h",
            "map_node.cc", "map_node.h",
            "model_node.cc", "model_node.h",
            "occ_grid_extractor.cc", "occ_grid_extractor.h",
           ],
    deps = [
            "//library/kitti",
            "//library/osg_nodes",
            "//library/ray_tracing",
            "//library/timer",
            "//library/util",
            "//library/viewer",
            "@boost//:boost_serialization",
           ],
)

cc_binary(
    name = "run_extractor",
    srcs = [
            "run_occ_grid_extractor.cc",
           ],
    deps = [
            ":kitti_occ_grids_lib",
           ],
)

cc_binary(
    name = "view_og",
    srcs = [
            "view_occ_grid.cc",
           ],
    deps = [
            ":kitti_occ_grids_lib",
           ],
)

cc_binary(
    name = "view_model",
    srcs = [
            "view_model.cc",
           ],
    deps = [
            ":kitti_occ_grids_lib",
           ],
)

cc_binary(
    name = "merge_occ_grids",
    srcs = [
            "merge_occ_grids.cc",
           ],
    deps = [
            ":kitti_occ_grids_lib",
            "@boost//:boost_filesystem",
           ],
)

cc_binary(
    name = "detector",
    srcs = [
            "run_detector.cc",
           ],
    linkopts = ["-lpthread"],
    deps = [
            ":kitti_occ_grids_lib",
            "@boost//:boost_filesystem",
           ],
)
