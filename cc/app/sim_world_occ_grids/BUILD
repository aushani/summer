cc_library(
    name = "sim_world_occ_grids_lib",
    srcs = [
            "chow_lui_tree.cc", "chow_lui_tree.h",
            "clt_node.cc", "clt_node.h",
            #"model.cc", "model.h",
            "joint_model.cc", "joint_model.h",
            #"detector.cc", "detector.h",
            #"map_node.cc", "map_node.h",
            "model_node.cc", "model_node.h",
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
            "occ_grid_extractor.cc",
            "occ_grid_extractor.h",
           ],
    deps = [
            "//library/ray_tracing",
            "//library/sim_world",
            "//library/timer",
            "//library/util",
           ],
)

cc_binary(
    name = "make_joint_model",
    srcs = [
            "make_joint_model.cc",
           ],
    deps = [
            ":sim_world_occ_grids_lib",
            "//library/ray_tracing",
            "@boost//:boost_filesystem",
           ],
)

cc_binary(
    name = "view_model",
    srcs = [
            "view_model.cc",
           ],
    deps = [
            ":sim_world_occ_grids_lib",
           ],
)

cc_binary(
    name = "make_chow_lui_tree",
    srcs = [
            "make_chow_lui_tree.cc",
           ],
    deps = [
            ":sim_world_occ_grids_lib",
            "@boost//:boost_filesystem",
           ],
)

cc_binary(
    name = "view_chow_lui_tree",
    srcs = [
            "view_chow_lui_tree.cc",
            #"chow_lui_tree_osg_node.cc", "chow_lui_tree_osg_node.h",
            "chow_lui_tree_node.cc", "chow_lui_tree_node.h",
           ],
    deps = [
            ":sim_world_occ_grids_lib",
           ],
)

cc_binary(
    name = "evaluate_detections",
    srcs = [
            "evaluate_detections.cc",
            "evaluator.cc", "evaluator.h",
           ],
    deps = [
            ":sim_world_occ_grids_lib",
            "@boost//:boost_filesystem",
           ],
)


"""
cc_library(
    name = "sim_world_occ_grids_lib",
    srcs = [
            "chow_lui_tree.cc", "chow_lui_tree.h",
            "chow_lui_tree_node.cc", "chow_lui_tree_node.h",
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
           ],
)

cc_binary(
    name = "view_og",
    srcs = [
            "view_occ_grid.cc",
           ],
    deps = [
            ":sim_world_occ_grids_lib",
           ],
)

cc_binary(
    name = "sample_clt",
    srcs = [
            "sample_clt.cc",
           ],
    deps = [
            ":sim_world_occ_grids_lib",
           ],
)

cc_binary(
    name = "detector",
    srcs = [
            "run_detector.cc",
           ],
    linkopts = ["-lpthread"],
    deps = [
            ":sim_world_occ_grids_lib",
            "@boost//:boost_filesystem",
           ],
)
"""
