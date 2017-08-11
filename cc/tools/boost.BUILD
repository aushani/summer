# -*- python -*-
package(default_visibility = ["//visibility:public"])

cc_library(
    name = "boost_headers",
    hdrs = glob(["boost_1_63_0/boost/**"]),
    defines = [
        "BOOST_ALL_DYN_LINK",
        "BOOST_MPL_CFG_NO_PREPROCESSED_HEADERS",
        "BOOST_MPL_LIMIT_VECTOR_SIZE=50",
    ],
    includes = ["boost_1_63_0"],
)

cc_library(
    name = "boost_system",
    srcs = ["boost_1_63_0/libs/system/src/error_code.cpp"],
    copts = ["-fvisibility=hidden"],
    deps = [":boost_headers"],
)

cc_library(
    name = "boost_serialization",
    srcs = glob(["boost_1_63_0/libs/serialization/src/*.cpp"]),
    hdrs = glob(["boost_1_63_0/libs/serialization/src/*.ipp"]),
    copts = ["-fvisibility=hidden"],
    deps = [":boost_headers"],
)

cc_library(
    name = "boost_thread",
    srcs = [
            "boost_1_63_0/libs/thread/src/pthread/once.cpp",
            "boost_1_63_0/libs/thread/src/pthread/thread.cpp",
           ],
    hdrs = [
            "boost_1_63_0/libs/thread/src/pthread/once_atomic.cpp",
           ],
    copts = ["-fvisibility=hidden"],
    deps = [":boost_headers", ":boost_system"],
)

cc_library(
  name = "boost_test",
  srcs = [
          "boost_1_63_0/libs/test/src/compiler_log_formatter.cpp",
          "boost_1_63_0/libs/test/src/debug.cpp",
          "boost_1_63_0/libs/test/src/decorator.cpp",
          "boost_1_63_0/libs/test/src/execution_monitor.cpp",
          "boost_1_63_0/libs/test/src/framework.cpp",
          "boost_1_63_0/libs/test/src/junit_log_formatter.cpp",
          "boost_1_63_0/libs/test/src/plain_report_formatter.cpp",
          "boost_1_63_0/libs/test/src/progress_monitor.cpp",
          "boost_1_63_0/libs/test/src/results_collector.cpp",
          "boost_1_63_0/libs/test/src/results_reporter.cpp",
          "boost_1_63_0/libs/test/src/test_tools.cpp",
          "boost_1_63_0/libs/test/src/test_tree.cpp",
          "boost_1_63_0/libs/test/src/unit_test_log.cpp",
          "boost_1_63_0/libs/test/src/unit_test_main.cpp",
          "boost_1_63_0/libs/test/src/unit_test_monitor.cpp",
          "boost_1_63_0/libs/test/src/unit_test_parameters.cpp",
          "boost_1_63_0/libs/test/src/xml_log_formatter.cpp",
          "boost_1_63_0/libs/test/src/xml_report_formatter.cpp",
         ],
  defines = ["BOOST_TEST_DYN_LINK"],
  deps = [":boost_headers"],
)
