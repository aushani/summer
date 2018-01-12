cc_binary(
    name = "generate_environments",
    srcs = [
            "generate_environments.cc",
            "environment_generator.cc",
            "environment_generator.h",
           ],
    deps = [
            "//library/ray_tracing",
            "//library/sim_world",
            "//library/timer",
            "//library/util",
            "@boost//:boost_filesystem",
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
            "@boost//:boost_filesystem",
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
    name = "view_joint_model",
    srcs = [
            "view_joint_model.cc",
           ],
    deps = [
            "//library/chow_liu_tree",
            "//library/osg_nodes",
            "//library/viewer",
            "@boost//:boost_filesystem",
           ],
)

cc_binary(
    name = "view_chow_liu_tree",
    srcs = [
            "view_chow_liu_tree.cc",
           ],
    deps = [
            "//library/chow_liu_tree",
            "//library/osg_nodes",
            "//library/viewer",
            "@boost//:boost_filesystem",
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
