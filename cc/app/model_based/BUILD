cc_library(
    name = "model_based_lib",
    srcs = ["detection_map.cc", "detection_map.h",
            "model_bank.cc", "model_bank.h",
            "ray_model.cc", "ray_model.h",
            "observation.cc", "observation.h",
            "object_state.cc", "object_state.h",
            "model_bank_builder.cc", "model_bank_builder.h",
            ],
    linkopts = ["-pthread"],
    deps = ["@eigen//:eigen", "//library/sim_world", "//library/timer", "@boost//:boost_serialization", "@boost//:boost_thread", "//library/histogram"]
)

cc_binary(
    name = "detector",
    srcs = ["run_exp.cc"],
    deps = ["model_based_lib"]
)

cc_binary(
    name = "build_model_bank",
    srcs = ["build_model_bank.cc"],
    deps = ["model_based_lib"]
)
