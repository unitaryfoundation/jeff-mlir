set(test_name "jeff-mlir-parse-test")

if(NOT TARGET ${test_name})
    add_executable(
        ${test_name} test_parse_for_op.cpp test_parse_switch_op.cpp test_parse_while_op.cpp
    )

    target_link_libraries(
        ${test_name}
        PRIVATE GTest::gtest_main MLIRJeff MLIRIR MLIRParser MLIRFuncDialect MLIRSupport
    )

    set_target_properties(${test_name} PROPERTIES FOLDER unittests)

    gtest_discover_tests(${test_name} DISCOVERY_TIMEOUT 60)
endif()
