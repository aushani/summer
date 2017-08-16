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
