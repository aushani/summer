cc_library(
    name = "kitti_occ_grids_lib",
    srcs = [
            "chow_lui_tree.cc", "chow_lui_tree.h",
            "chow_lui_tree_osg_node.cc", "chow_lui_tree_osg_node.h",
            "clt_node.cc", "clt_node.h",
            "evaluator.cc", "evaluator.h",
            "model.cc", "model.h",
            "joint_model.cc", "joint_model.h",
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
            "@boost//:boost_iostreams",
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
    name = "make_chow_lui_trees",
    srcs = [
            "make_chow_lui_trees.cc",
           ],
    deps = [
            ":kitti_occ_grids_lib",
            "@boost//:boost_filesystem",
           ],
)

cc_binary(
    name = "view_chow_lui_tree",
    srcs = [
            "view_chow_lui_tree.cc",
           ],
    deps = [
            ":kitti_occ_grids_lib",
           ],
)

cc_binary(
    name = "sample_clt",
    srcs = [
            "sample_clt.cc",
           ],
    deps = [
            ":kitti_occ_grids_lib",
           ],
)

cc_binary(
    name = "evaluate_detections",
    srcs = [
            "evaluate_detections.cc",
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

cc_binary(
    name = "test_timing",
    srcs = [
            "test_timing.cc",
           ],
    deps = [
            ":kitti_occ_grids_lib",
           ],
)
