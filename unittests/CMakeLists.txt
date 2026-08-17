find_program(UV_EXECUTABLE uv REQUIRED)

# Generate test cases
add_test(
    NAME generate_test_cases
    COMMAND ${UV_EXECUTABLE} run --script ${CMAKE_SOURCE_DIR}/unittests/generate_test_cases.py
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
set_tests_properties(generate_test_cases PROPERTIES FIXTURES_SETUP generate_test_cases_success)

add_subdirectory(Conversion)
add_subdirectory(IR)
add_subdirectory(Translation)
