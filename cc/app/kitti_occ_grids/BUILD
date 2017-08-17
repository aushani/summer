cc_binary(
    name = "run_extractor",
    srcs = [
            "run_occ_grid_extractor.cc",
            "occ_grid_extractor.cc", "occ_grid_extractor.h",
           ],
    deps = [
            "//library/kitti",
            "//library/ray_tracing",
            "//library/timer",
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
    name = "merge_occ_grids",
    srcs = [
            "merge_occ_grids.cc",
           ],
    deps = [
            "//library/ray_tracing",
            "@boost//:boost_filesystem",
           ],
)

cc_binary(
    name = "detector",
    srcs = [
            "detector.cc", "detector.h",
            "map_node.cc", "map_node.h",
            "run_detector.cc",
           ],
    linkopts = ["-lpthread"],
    deps = [
            "//library/kitti",
            "//library/osg_nodes",
            "//library/ray_tracing",
            "//library/viewer",
            "@boost//:boost_headers",
            "@boost//:boost_filesystem",
           ],
)
