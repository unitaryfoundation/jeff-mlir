find_program(UV_EXECUTABLE uv REQUIRED)

# Generate test cases before the test executables are built so that the tests reading them can be
# discovered
set(generate_test_cases_stamp "${CMAKE_CURRENT_BINARY_DIR}/generate_test_cases.stamp")
add_custom_command(
    OUTPUT ${generate_test_cases_stamp}
    COMMAND ${UV_EXECUTABLE} run --script ${CMAKE_CURRENT_SOURCE_DIR}/generate_test_cases.py
    COMMAND ${CMAKE_COMMAND} -E touch ${generate_test_cases_stamp}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/generate_test_cases.py
    COMMENT "Generating test cases"
    VERBATIM
)
add_custom_target(generate_test_cases DEPENDS ${generate_test_cases_stamp})

add_subdirectory(Conversion)
add_subdirectory(IR)
add_subdirectory(Translation)
