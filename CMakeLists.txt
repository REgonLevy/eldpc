# SPDX-License-Identifier: Apache-2.0
# eldpc native build.
#
#   libeldpc_core_avx2.so          the fastbpgpu6g decoder core, host half compiled with -mavx2
#   libeldpc_core_avx512.so        the same sources, host half compiled with the AVX-512 flag set
#   libeldpc_core_avx2_ada64.so    the AVX2 core with DBENCH_GPU_SM_BUDGET=64 (the Ada 142-SM class)
#   libeldpc_core_avx512_ada64.so  the AVX-512 core with DBENCH_GPU_SM_BUDGET=64
#   libeldpc_cpu_avx2.so           the optional CPU core fastbpcpu6g (no CUDA), -mavx2
#   libeldpc_cpu_avx512.so         the same CPU sources, with the AVX-512 flag set
#   libeldpc.so                    the C API front (include/eldpc.h): CPUID/XGETBV choice of the
#                                  host ISA and, per handle, the default/ada64 GPU variant or the
#                                  CPU core (config key device=cpu)
#
# The four cores carry CUDA SASS for sm_89 and sm_120 and PTX for compute_120, and the same host
# flags as the decoder's reference build: -O2 -DNDEBUG -ffp-contract=off -fno-fast-math,
# then -O3, then the ISA flags; nvcc carries the exactness flags -fmad=false -ftz=false
# -prec-div=true -prec-sqrt=true last on its command line.
#
# The two builds of one ISA are compiled from the same sources with the same flags; their one
# difference is DBENCH_GPU_SM_BUDGET=64 in the ada64 build (see ELDPC_CORE_DEFINITIONS below).
#
# The CUDA runtime is linked statically into every core (CUDA_RUNTIME_LIBRARY Static /
# CUDA::cudart_static): an installed package then needs only the NVIDIA driver (libcuda.so.1),
# never a libcudart from the system, and it cannot load one at run time.
#
# A normal build (from the package root):
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
# `pip install .` drives the same build through scikit-build-core.
#
# CPU-only build: -DELDPC_WITHOUT_CUDA=ON (or ELDPC_WITHOUT_CUDA=1 in the environment, which a
# `pip install` picks up) builds libeldpc.so and the two CPU cores only. It does not enable the
# CUDA language, does not look for the CUDA toolkit and never runs nvcc, so a machine without the
# toolkit can build and install the package; the front then reports the CPU core as the process's
# core and a GPU create fails with a message saying the build has no GPU cores.
#
# With ELDPC_WITHOUT_CUDA unset (the default), a configure that finds no working CUDA compiler
# (check_language(CUDA): nvcc not on PATH, CUDACXX not set) builds the same CPU-only package and
# prints a WARNING that says so; -DELDPC_WITHOUT_CUDA=OFF (or ELDPC_WITHOUT_CUDA=0) requires the
# GPU cores, and a missing CUDA compiler is then a configure error.
cmake_minimum_required(VERSION 3.21)

# Package version: the one of pyproject.toml (scikit-build-core passes it as SKBUILD_PROJECT_VERSION).
if(DEFINED SKBUILD_PROJECT_VERSION AND NOT "${SKBUILD_PROJECT_VERSION}" STREQUAL "")
  set(ELDPC_VERSION "${SKBUILD_PROJECT_VERSION}")
else()
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml" _eldpc_pyproject)
  string(REGEX MATCH "version[ \t]*=[ \t]*\"([^\"]+)\"" _eldpc_version_match "${_eldpc_pyproject}")
  if(NOT _eldpc_version_match)
    message(FATAL_ERROR "cannot read the package version from pyproject.toml")
  endif()
  set(ELDPC_VERSION "${CMAKE_MATCH_1}")
endif()

project(eldpc VERSION ${ELDPC_VERSION} LANGUAGES C CXX)

# CPU-only build: the C front and the two CPU cores, no CUDA language and no CUDA toolkit.
#   -DELDPC_WITHOUT_CUDA=ON / ELDPC_WITHOUT_CUDA=1 (or on, true) in the environment: CPU-only.
#   -DELDPC_WITHOUT_CUDA=OFF / any other non-empty value of the environment variable: the GPU cores
#     are required; without a CUDA compiler the configure fails.
#   unset or empty (the default): the GPU cores when CMake finds a working CUDA compiler, else the
#     CPU-only package with a WARNING.
# An explicit -DELDPC_WITHOUT_CUDA=... (or a value cached by an earlier configure) wins over the
# environment variable, which is cached as the option's value when it decides. The automatic choice
# is not cached, so every configure makes it again.
if(DEFINED ELDPC_WITHOUT_CUDA AND NOT "${ELDPC_WITHOUT_CUDA}" STREQUAL "")
  set(_eldpc_cuda_choice explicit)
elseif(DEFINED ENV{ELDPC_WITHOUT_CUDA} AND NOT "$ENV{ELDPC_WITHOUT_CUDA}" STREQUAL "")
  set(_eldpc_cuda_choice explicit)
  set(_eldpc_doc "Build only the C front and the CPU cores (no CUDA toolkit needed)")
  if("$ENV{ELDPC_WITHOUT_CUDA}" MATCHES "^(1|[Oo][Nn]|[Tt][Rr][Uu][Ee])$")
    set(ELDPC_WITHOUT_CUDA ON CACHE BOOL "${_eldpc_doc}" FORCE)
  else()
    set(ELDPC_WITHOUT_CUDA OFF CACHE BOOL "${_eldpc_doc}" FORCE)
  endif()
else()
  set(_eldpc_cuda_choice auto)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    set(ELDPC_WITHOUT_CUDA OFF)
  else()
    set(ELDPC_WITHOUT_CUDA ON)
    message(WARNING
      "eldpc: no CUDA compiler was found, so this is a CPU-only build: it has the CPU cores only "
      "and no GPU cores (device=\"gpu\" will fail; device=\"cpu\" and device=\"auto\" run on the "
      "CPU core).\n"
      "What is missing: a working nvcc. nvcc is not on PATH and CUDACXX is not set (or the nvcc "
      "found failed CMake's compiler check).\n"
      "To build the GPU cores: install the CUDA 13 toolkit "
      "(https://developer.nvidia.com/cuda-downloads), put its bin directory on PATH or set "
      "CUDACXX to its nvcc (for example CUDACXX=/usr/local/cuda-13.3/bin/nvcc), and rebuild "
      "(pip install . again, or a fresh CMake build directory).\n"
      "To silence this warning, choose explicitly: ELDPC_WITHOUT_CUDA=1 builds the CPU-only "
      "package, ELDPC_WITHOUT_CUDA=0 requires the GPU cores and fails without a CUDA compiler.")
  endif()
endif()
if(ELDPC_WITHOUT_CUDA)
  message(STATUS "eldpc: CPU-only build (the C front and the CPU cores; ${_eldpc_cuda_choice} choice)")
else()
  enable_language(CUDA)
  message(STATUS "eldpc: GPU and CPU cores (${_eldpc_cuda_choice} choice; nvcc ${CMAKE_CUDA_COMPILER})")
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Determinism: the decoder's outputs must not depend on FMA contraction or fast-math
# reassociation (the decoder's reference build uses the same rules).
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_C_FLAGS_RELEASE "-O2 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
add_compile_options(-ffp-contract=off -fno-fast-math -Wall)

if(NOT ELDPC_WITHOUT_CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_RUNTIME_LIBRARY Static)
  find_package(CUDAToolkit REQUIRED)
endif()

# The host flags of the core, in the string form the core reports: the release flags, then
# -ffp-contract=off -fno-fast-math.
string(TOUPPER "${CMAKE_BUILD_TYPE}" _eldpc_bt)
set(ELDPC_CXX_FLAGS_STR "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${_eldpc_bt}} -ffp-contract=off -fno-fast-math")
string(STRIP "${ELDPC_CXX_FLAGS_STR}" ELDPC_CXX_FLAGS_STR)

find_package(Threads REQUIRED)

if(NOT ELDPC_WITHOUT_CUDA)
  # Options of the .cu sources besides the top-level ones: the fatbin identifier names the CUDA
  # images after their virtual architecture, so the PTX image compiled for compute_120 is listed by
  # cuobjdump as compute_120.sm_120.ptx (the PTX target of compute capability 12.0). It changes no
  # code: the SASS and PTX bytes are those of the generated images.
  set(_eldpc_cuda_options "-Xfatbin=--ident=compute_120")

  # The options for the .cu sources' build-info string: CMAKE_CUDA_FLAGS, the build type's CUDA
  # flags, the directory options, the CUDA options and the exactness flags.
  get_directory_property(_eldpc_top_opts COMPILE_OPTIONS)
  set(_eldpc_cuda_exact -fmad=false -ftz=false -prec-div=true -prec-sqrt=true)
  set(_eldpc_cuda_flags ${CMAKE_CUDA_FLAGS} ${CMAKE_CUDA_FLAGS_${_eldpc_bt}} ${_eldpc_top_opts}
      ${_eldpc_cuda_options} ${_eldpc_cuda_exact})
  string(REPLACE ";" " " _eldpc_cuda_flags "${_eldpc_cuda_flags}")
  string(REGEX REPLACE "  +" " " _eldpc_cuda_flags "${_eldpc_cuda_flags}")
  string(STRIP "${_eldpc_cuda_flags}" _eldpc_cuda_flags)
endif()

# The compile definitions every core gets. The ada64 variant of an ISA takes this same list with
# the SM budget replaced by 64 -- that one value is the only difference between the two builds.
set(ELDPC_CORE_DEFINITIONS DBENCH_GPU_SM_BUDGET=0 DBENCH_GPU_L2_BUDGET=0 DBENCH_INEXACT_MEASURE=0)

# eldpc_core(<target> <isa> <sm budget> <isa flags> <cuda arch list> <build-info cuda_arch>)
function(eldpc_core target isa sm_budget isa_flags cuda_archs cuda_arch_info)
  add_library(${target} SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/src/core/fastbpgpu6g.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/core/bp_gpu.cu)
  set(_info_dir "${CMAKE_CURRENT_BINARY_DIR}/build-info/${target}")
  target_include_directories(${target} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/port
    ${CMAKE_CURRENT_SOURCE_DIR}/src/third_party/ldpc/src_cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/third_party/ldpc/include/robin_map
    ${_info_dir})
  set_target_properties(${target} PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    CUDA_ARCHITECTURES "${cuda_archs}"
    CUDA_RUNTIME_LIBRARY Static
    CUDA_VISIBILITY_PRESET hidden
    CUDA_SEPARABLE_COMPILATION OFF
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
  # OPTIONS and the ISA flags are for the C/C++ sources only; the .cu sources keep the CUDA
  # flags (the rule of the decoder's reference build).
  set(_all_opts -O3 ${isa_flags})
  target_compile_options(${target} PRIVATE "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:${_all_opts}>")
  target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${_eldpc_cuda_options}>")
  # The exactness flags go last on every nvcc command line, as one SHELL group (CMake keeps a
  # grouped SHELL argument together and in order).
  target_compile_options(${target} PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-fmad=false -ftz=false -prec-div=true -prec-sqrt=true>")
  # The one override of the shared list: the variant's SM budget.
  string(REPLACE "DBENCH_GPU_SM_BUDGET=0" "DBENCH_GPU_SM_BUDGET=${sm_budget}"
         _core_defs "${ELDPC_CORE_DEFINITIONS}")
  target_compile_definitions(${target} PRIVATE ${_core_defs})
  target_link_libraries(${target} PRIVATE CUDA::cudart_static)

  # The build-info tail is generated from the same list, so the definitions the core is compiled
  # with and the ones it reports cannot drift apart: gpu_sm_budget=0; gpu_l2_budget=0;
  # inexact_measure=0, with the variant's SM budget in place of 0.
  set(_defs_info "")
  foreach(_def IN LISTS _core_defs)
    string(FIND "${_def}" "=" _eq)
    string(SUBSTRING "${_def}" 0 ${_eq} _key)
    math(EXPR _value_start "${_eq} + 1")
    string(SUBSTRING "${_def}" ${_value_start} -1 _value)
    string(TOLOWER "${_key}" _key)
    string(REPLACE "dbench_" "" _key "${_key}")
    if(_defs_info STREQUAL "")
      set(_defs_info "${_key}=${_value}")
    else()
      string(APPEND _defs_info "; ${_key}=${_value}")
    endif()
  endforeach()

  string(REPLACE ";" " " _extra "${_all_opts}")
  set(_build_info
    "compiler=${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}; flags=${ELDPC_CXX_FLAGS_STR} ${_extra}; build_type=${CMAKE_BUILD_TYPE}; isa=${isa}; lib=fastbpgpu6g; cuda=${CMAKE_CUDA_COMPILER_ID} ${CMAKE_CUDA_COMPILER_VERSION}; cuda_arch=${cuda_arch_info}; cuda_flags=${_eldpc_cuda_flags}; ${_defs_info}")
  file(WRITE "${_info_dir}/dbench_candidate_build_info.h.tmp"
    "// generated by eldpc/CMakeLists.txt\n#define DBENCH_CANDIDATE_NAME \"fastbpgpu6g\"\n#define DBENCH_CANDIDATE_BUILD_INFO \"${_build_info}\"\n")
  configure_file("${_info_dir}/dbench_candidate_build_info.h.tmp"
                 "${_info_dir}/dbench_candidate_build_info.h" COPYONLY)
endfunction()

# The CUDA images of the GPU cores (CMake's CUDA_ARCHITECTURES) and the same list in the words of
# their build info (cuda_arch=, which the front and eldpc.devices() check devices against).
set(_eldpc_cuda_archs "89-real;120-real;120-virtual")
set(_eldpc_cuda_arch_info "sm_89,sm_120,compute_120")

if(NOT ELDPC_WITHOUT_CUDA)
  # The AVX-512 flag set: exactly the one the decoder's reference AVX-512 build uses
  # (Sapphire Rapids, Zen 4 and Zen 5 all support it). Never a wider set here.
  eldpc_core(eldpc_core_avx2 avx2 0 "-mavx2" "${_eldpc_cuda_archs}" "${_eldpc_cuda_arch_info}")
  eldpc_core(eldpc_core_avx512 avx512 0 "-mavx2;-mavx512f;-mavx512vl;-mavx512dq;-mavx512bw"
             "${_eldpc_cuda_archs}" "${_eldpc_cuda_arch_info}")
  # The Ada 142-SM class: the same sources and flags with DBENCH_GPU_SM_BUDGET=64.
  eldpc_core(eldpc_core_avx2_ada64 avx2 64 "-mavx2" "${_eldpc_cuda_archs}" "${_eldpc_cuda_arch_info}")
  eldpc_core(eldpc_core_avx512_ada64 avx512 64 "-mavx2;-mavx512f;-mavx512vl;-mavx512dq;-mavx512bw"
             "${_eldpc_cuda_archs}" "${_eldpc_cuda_arch_info}")
endif()

# The optional CPU core: the fastbpcpu6g sources of src/cpu/, no CUDA code and no CUDA library.
# It is compiled with exactly the host flags of the GPU core of the same ISA (the release flags,
# -ffp-contract=off -fno-fast-math, then -O3, then the ISA flag list), so a CPU handle and a GPU
# handle of the same ISA are the same arithmetic on different engines. The build-info string has
# the GPU core's form and the same flags= field; its lib= field names fastbpcpu6g.
function(eldpc_cpu_core target isa isa_flags)
  add_library(${target} SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/cpu/fastbpcpu6g.cpp)
  set(_info_dir "${CMAKE_CURRENT_BINARY_DIR}/build-info/${target}")
  target_include_directories(${target} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/port
    ${CMAKE_CURRENT_SOURCE_DIR}/src/third_party/ldpc/src_cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/third_party/ldpc/include/robin_map
    ${_info_dir})
  set_target_properties(${target} PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
  set(_all_opts -O3 ${isa_flags})
  target_compile_options(${target} PRIVATE ${_all_opts})
  # The multi-threaded engine starts threads with std::thread and pins them (pthread).
  target_link_libraries(${target} PRIVATE Threads::Threads)

  string(REPLACE ";" " " _extra "${_all_opts}")
  set(_build_info
    "compiler=${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}; flags=${ELDPC_CXX_FLAGS_STR} ${_extra}; build_type=${CMAKE_BUILD_TYPE}; isa=${isa}; lib=fastbpcpu6g; threads=1..64")
  file(WRITE "${_info_dir}/dbench_candidate_build_info.h.tmp"
    "// generated by eldpc/CMakeLists.txt\n#define DBENCH_CANDIDATE_NAME \"fastbpcpu6g\"\n#define DBENCH_CANDIDATE_BUILD_INFO \"${_build_info}\"\n")
  configure_file("${_info_dir}/dbench_candidate_build_info.h.tmp"
                 "${_info_dir}/dbench_candidate_build_info.h" COPYONLY)
endfunction()

eldpc_cpu_core(eldpc_cpu_avx2 avx2 "-mavx2")
eldpc_cpu_core(eldpc_cpu_avx512 avx512 "-mavx2;-mavx512f;-mavx512vl;-mavx512dq;-mavx512bw")

# The C API front. No ISA flags: it must run on any x86-64 CPU to report the choice or the error.
add_library(eldpc SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/port/eldpc_front.cpp)
target_include_directories(eldpc PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/src/port)
target_compile_definitions(eldpc PRIVATE
  ELDPC_VERSION="${ELDPC_VERSION}"
  ELDPC_FRONT_COMPILER="${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
if(ELDPC_WITHOUT_CUDA)
  # The front knows it has no GPU cores: the CPU core is the process's core, and a device=gpu
  # create fails with a message that names the CPU core.
  target_compile_definitions(eldpc PRIVATE ELDPC_CPU_ONLY=1)
endif()
target_link_libraries(eldpc PRIVATE ${CMAKE_DL_LIBS})
set_target_properties(eldpc PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# Install layout of the wheel: the Python package directory holds the libraries, the C header,
# a pkg-config file and a CMake package config for C consumers.
set(_eldpc_libraries eldpc eldpc_cpu_avx2 eldpc_cpu_avx512)
if(NOT ELDPC_WITHOUT_CUDA)
  list(APPEND _eldpc_libraries eldpc_core_avx2 eldpc_core_avx512 eldpc_core_avx2_ada64
       eldpc_core_avx512_ada64)
endif()
install(TARGETS ${_eldpc_libraries} LIBRARY DESTINATION eldpc/_libs)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/eldpc.h DESTINATION eldpc/include)

# The build record next to the libraries: whether this build has the GPU cores, whether that was
# CMake's automatic choice (a CUDA compiler found or not) or ELDPC_WITHOUT_CUDA's, and the GPU
# cores' toolkit and CUDA images (the cuda= and cuda_arch= of their build info). `import eldpc`
# reads it to warn about a CPU-only build that found no nvcc on a machine with a GPU, and
# eldpc.devices() to say whether this build supports each device, without loading a library;
# eldpc.check() uses it to say why a build has no GPU cores.
if(ELDPC_WITHOUT_CUDA)
  set(_eldpc_gpu_cores_json false)
  set(_eldpc_cuda_json "")
  set(_eldpc_cuda_arch_json "")
else()
  set(_eldpc_gpu_cores_json true)
  set(_eldpc_cuda_json "${CMAKE_CUDA_COMPILER_ID} ${CMAKE_CUDA_COMPILER_VERSION}")
  set(_eldpc_cuda_arch_json "${_eldpc_cuda_arch_info}")
endif()
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/build.json"
  "{\"schema\": \"eldpc.build.v1\", \"gpu_cores\": ${_eldpc_gpu_cores_json}, \"cuda_choice\": \"${_eldpc_cuda_choice}\", \"cuda\": \"${_eldpc_cuda_json}\", \"cuda_arch\": \"${_eldpc_cuda_arch_json}\"}\n")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/build.json" DESTINATION eldpc/_libs)

# pkg-config: PKG_CONFIG_PATH=<prefix>/eldpc/lib/pkgconfig. The paths are relative to the file.
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/eldpc.pc"
"prefix=\${pcfiledir}/../..
exec_prefix=\${prefix}
libdir=\${prefix}/_libs
includedir=\${prefix}/include

Name: eldpc
Description: Exact GPU BP+OSD decoding for quantum LDPC codes
Version: ${ELDPC_VERSION}
Libs: -L\${libdir} -leldpc
Cflags: -I\${includedir}
")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/eldpc.pc" DESTINATION eldpc/lib/pkgconfig)

# CMake package: find_package(eldpc) with CMAKE_PREFIX_PATH=<prefix>/eldpc provides eldpc::eldpc.
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/eldpcConfig.cmake"
"# CMake package config for eldpc (generated by eldpc/CMakeLists.txt).
get_filename_component(_eldpc_prefix \"\${CMAKE_CURRENT_LIST_DIR}/../../..\" ABSOLUTE)
if(NOT TARGET eldpc::eldpc)
  add_library(eldpc::eldpc SHARED IMPORTED)
  set_target_properties(eldpc::eldpc PROPERTIES
    IMPORTED_LOCATION \"\${_eldpc_prefix}/_libs/libeldpc.so\"
    INTERFACE_INCLUDE_DIRECTORIES \"\${_eldpc_prefix}/include\")
endif()
unset(_eldpc_prefix)
")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/eldpcConfigVersion.cmake"
"set(PACKAGE_VERSION \"${ELDPC_VERSION}\")
if(PACKAGE_VERSION VERSION_LESS PACKAGE_FIND_VERSION)
  set(PACKAGE_VERSION_COMPATIBLE FALSE)
else()
  set(PACKAGE_VERSION_COMPATIBLE TRUE)
  if(PACKAGE_FIND_VERSION STREQUAL PACKAGE_VERSION)
    set(PACKAGE_VERSION_EXACT TRUE)
  endif()
endif()
")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/eldpcConfig.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/eldpcConfigVersion.cmake"
        DESTINATION eldpc/lib/cmake/eldpc)
