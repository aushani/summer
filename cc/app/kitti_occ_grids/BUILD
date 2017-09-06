cc_binary(
    name = "run_extractor",
    srcs = [
            "run_occ_grid_extractor.cc",
            "occ_grid_extractor.cc", "occ_grid_extractor.h",
           ],
    deps = [
            "//library/kitti",
            "//library/ray_tracing",
            "@boost//:boost_filesystem",
           ],
)

cc_binary(
    name = "view_og",
    srcs = [
            "view_occ_grid.cc",
           ],
    deps = [
            "//library/osg_nodes",
            "//library/ray_tracing",
            "//library/viewer",
           ],
)

cc_binary(
    name = "make_joint_model",
    srcs = [
            "make_joint_model.cc",
           ],
    deps = [
            "//library/chow_liu_tree",
            "//library/ray_tracing",
            "@boost//:boost_filesystem",
           ],
)

cc_binary(
    name = "make_marginal_model",
    srcs = [
            "make_marginal_model.cc",
           ],
    deps = [
            "//library/chow_liu_tree",
            "//library/ray_tracing",
            "@boost//:boost_filesystem",
           ],
)

cc_binary(
    name = "view_joint_model",
    srcs = [
            "view_joint_model.cc",
           ],
    deps = [
            "//library/osg_nodes",
            "//library/ray_tracing",
            "//library/viewer",
           ],
)

cc_binary(
    name = "view_marginal_model",
    srcs = [
            "view_marginal_model.cc",
           ],
    deps = [
            "//library/osg_nodes",
            "//library/ray_tracing",
            "//library/viewer",
           ],
)

cc_binary(
    name = "view_chow_lui_tree",
    srcs = [
            "view_chow_lui_tree.cc",
           ],
    deps = [
            "//library/osg_nodes",
            "//library/ray_tracing",
            "//library/viewer",
           ],
)

cc_binary(
    name = "evaluate_detections",
    srcs = [
            "evaluate_detections.cc",
            "evaluator.cc", "evaluator.h",
           ],
    deps = [
            "//library/chow_liu_tree",
            "@boost//:boost_filesystem",
           ],
)

cc_binary(
    name = "detector",
    srcs = [
            "run_detector.cc",
            "map_node.cc", "map_node.h",
           ],
    deps = [
            "//library/chow_liu_tree",
            "//library/util",
            "//library/kitti",
            "//library/osg_nodes",
            "//library/viewer",
            "//library/detector",
            "@boost//:boost_filesystem",
           ],
)
