cmake_minimum_required(VERSION 3.20)
project(MetalFlow
    VERSION 1.0.0
    DESCRIPTION "CUDA-style Metal layer for CFD/FEA on Apple Silicon"
    LANGUAGES CXX OBJCXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 17)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)

add_library(metalflow
    src/metalflow/MetalCompute.mm
)
add_library(MetalFlow::metalflow ALIAS metalflow)

target_include_directories(metalflow PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(metalflow PUBLIC
    "-framework Metal"
    "-framework Foundation"
)
target_compile_options(metalflow PRIVATE -fobjc-arc)

add_executable(metalflow_demo examples/demo.cpp)
target_link_libraries(metalflow_demo PRIVATE metalflow)

add_executable(laplace_solver examples/laplace_solver.cpp)
target_link_libraries(laplace_solver PRIVATE metalflow)

# Copy shaders next to binaries for runtime compileSource().
foreach(tgt metalflow_demo laplace_solver)
    add_custom_command(TARGET ${tgt} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory
                $<TARGET_FILE_DIR:${tgt}>/shaders
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${CMAKE_CURRENT_SOURCE_DIR}/shaders/cfd_fea.metal
                $<TARGET_FILE_DIR:${tgt}>/shaders/cfd_fea.metal
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${CMAKE_CURRENT_SOURCE_DIR}/shaders/laplace.metal
                $<TARGET_FILE_DIR:${tgt}>/shaders/laplace.metal
    )
endforeach()

# Always refresh shaders into the build tree (even if binaries are up to date).
add_custom_target(copy_shaders ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/shaders
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/shaders/cfd_fea.metal
            ${CMAKE_CURRENT_BINARY_DIR}/shaders/cfd_fea.metal
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/shaders/laplace.metal
            ${CMAKE_CURRENT_BINARY_DIR}/shaders/laplace.metal
)

# ---------------------------------------------------------------------------
# Documentation (Doxygen + Graphviz)
# ---------------------------------------------------------------------------
find_package(Doxygen)
if(DOXYGEN_FOUND)
    add_custom_target(docs
        COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/docs
        COMMENT "Generating MetalFlow API documentation with Doxygen"
        VERBATIM
    )
    message(STATUS "Doxygen found: use 'cmake --build build --target docs'")
else()
    message(STATUS "Doxygen not found: docs target disabled (brew install doxygen graphviz)")
endif()
