# SPDX-License-Identifier: Apache-2.0
# Reference decoder of the eldpc verification suite.
#
# A C ABI (decoder_abi.h) shim over ldpc 2.4.1's C++ headers, with the deterministic tie
# order of the decoder-bench fork applied (ldpc-2.4.1-decoder-bench.patch). The sources in this
# directory are complete: the target builds from this directory alone, with no file from the
# project that ships it.
#
#   cmake -S <this directory> -B <build> -DCMAKE_BUILD_TYPE=Release
#   cmake --build <build> -j
#
# Produces <build>/libeldpc_verify_reference.so. The suite's runner builds it on first use into
# a cache directory and loads it through ctypes.
cmake_minimum_required(VERSION 3.16)
project(eldpc_verify_reference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
# Determinism flags: no FMA contraction, no fast math.
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

add_library(eldpc_verify_reference SHARED ldpc_ref_abi.cpp)
target_include_directories(eldpc_verify_reference PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/ldpc/src_cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ldpc/include/robin_map
  ${CMAKE_CURRENT_BINARY_DIR})
target_compile_options(eldpc_verify_reference PRIVATE -ffp-contract=off -fno-fast-math -fvisibility-inlines-hidden -w)
set_target_properties(eldpc_verify_reference PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  OUTPUT_NAME eldpc_verify_reference)

# Build description returned by dbench_decoder_build_info(): the compiler, the flags, the build
# type and the SHA-256 of the tie-order patch that was applied to the headers. The format is that of
# the reference construction's reference library, so the two descriptions can be compared.
file(SHA256 ${CMAKE_CURRENT_SOURCE_DIR}/ldpc-2.4.1-decoder-bench.patch ELDPC_REFERENCE_PATCH_SHA256)
set(ELDPC_REFERENCE_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE} -ffp-contract=off -fno-fast-math -w")
string(STRIP "${ELDPC_REFERENCE_FLAGS}" ELDPC_REFERENCE_FLAGS)
set(ELDPC_REFERENCE_BUILD_INFO
  "compiler=${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}; flags=${ELDPC_REFERENCE_FLAGS}; build_type=${CMAKE_BUILD_TYPE}; ldpc=2.4.1; patch_sha256=${ELDPC_REFERENCE_PATCH_SHA256}")
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/ldpc_ref_build_info.hpp.in
               ${CMAKE_CURRENT_BINARY_DIR}/ldpc_ref_build_info.hpp @ONLY)
