cmake_minimum_required(VERSION 3.18)
project(
    libtokenizers
    LANGUAGES C CXX
    VERSION 0.0.2
)

# Options
option(LIBTOKENIZERS_BUILD_TESTS "Build libtokenizers tests" OFF)
set(LIBTOKENIZERS_MACOSX_DEPLOYMENT_TARGET "13.0" CACHE STRING "macOS deployment target")
set(LIBTOKENIZERS_IPHONEOS_DEPLOYMENT_TARGET "14.0" CACHE STRING "iOS deployment target")

# Find cargo
find_program(CARGO_EXECUTABLE cargo REQUIRED)

# Infer Cargo target
set(LIBTOKENIZERS_CARGO_TARGET "")
set(CARGO_EXTRA_ENVS "")
if(CMAKE_SYSTEM_NAME STREQUAL "Android")
    # Infer target triple
    if(ANDROID_ABI STREQUAL "arm64-v8a")
        set(LIBTOKENIZERS_CARGO_TARGET aarch64-linux-android)
    elseif(ANDROID_ABI STREQUAL "armeabi-v7a")
        set(LIBTOKENIZERS_CARGO_TARGET armv7-linux-androideabi)
    elseif(ANDROID_ABI STREQUAL "x86_64")
        set(LIBTOKENIZERS_CARGO_TARGET x86_64-linux-android)
    elseif(ANDROID_ABI STREQUAL "x86")
        set(LIBTOKENIZERS_CARGO_TARGET i686-linux-android)
    else()
        message(FATAL_ERROR "Failed to configure libtokenizers for Android because of unsupported architecture: ${ANDROID_ABI}")
    endif()
    set(CARGO_EXTRA_ENVS
        AR_${LIBTOKENIZERS_CARGO_TARGET}=${CMAKE_AR}
        CC_${LIBTOKENIZERS_CARGO_TARGET}=${CMAKE_C_COMPILER}
        CXX_${LIBTOKENIZERS_CARGO_TARGET}=${CMAKE_CXX_COMPILER}
        "RUSTFLAGS=-C relocation-model=pic"
    )
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
        set(LIBTOKENIZERS_CARGO_TARGET aarch64-apple-darwin)
    elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
        set(LIBTOKENIZERS_CARGO_TARGET x86_64-apple-darwin)
    else()
        message(FATAL_ERROR "Failed to configure libtokenizers for macOS because of unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}")
    endif()
    set(CARGO_EXTRA_ENVS
        MACOSX_DEPLOYMENT_TARGET=${LIBTOKENIZERS_MACOSX_DEPLOYMENT_TARGET}
        "RUSTFLAGS=-C relocation-model=pic"
    )
elseif(CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
    set(LIBTOKENIZERS_CARGO_TARGET wasm32-unknown-emscripten)
    set(CARGO_EXTRA_ENVS
        # Compiler paths
        AR_${LIBTOKENIZERS_CARGO_TARGET}=${CMAKE_AR}
        CC_${LIBTOKENIZERS_CARGO_TARGET}=${CMAKE_C_COMPILER}
        CXX_${LIBTOKENIZERS_CARGO_TARGET}=${CMAKE_CXX_COMPILER}
        # For Rust packages that compile C libraries with `cc-rs`, we need to force-enable PIC
        CFLAGS_${LIBTOKENIZERS_CARGO_TARGET}="-fPIC"
        CXXFLAGS_${LIBTOKENIZERS_CARGO_TARGET}="-fPIC"
        # A few critical items:
        # 1. Enable PIC
        # 2. Default visibility to hidden
        # 3. Enable WASM EH (`-Zemscripten-wasm-eh`) which also disables Emscripten (JS) exceptions (`call -> invoke_*` calls).
        # 4. Enable legacy WASM EH for coverage of older browsers
        "RUSTFLAGS=-C relocation-model=pic -Zdefault-visibility=hidden -Zemscripten-wasm-eh -C llvm-args=-wasm-use-legacy-eh"
    )
elseif(CMAKE_SYSTEM_NAME STREQUAL "iOS") # CHECK # We don't support iphonesimulator
    set(LIBTOKENIZERS_CARGO_TARGET aarch64-apple-ios)
    set(CARGO_EXTRA_ENVS
        IPHONEOS_DEPLOYMENT_TARGET=${LIBTOKENIZERS_IPHONEOS_DEPLOYMENT_TARGET}
        "RUSTFLAGS=-C relocation-model=pic"
    )
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
        set(LIBTOKENIZERS_CARGO_TARGET aarch64-unknown-linux-gnu)
    elseif (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
        set(LIBTOKENIZERS_CARGO_TARGET x86_64-unknown-linux-gnu)
    else()
        message(FATAL_ERROR "libtokenizers cannot be compiled for architecture ${CMAKE_SYSTEM_PROCESSOR}")
    endif()
    set(CARGO_EXTRA_ENVS "RUSTFLAGS=-C relocation-model=pic")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    if (CMAKE_SYSTEM_PROCESSOR STREQUAL "ARM64")
        set(LIBTOKENIZERS_CARGO_TARGET aarch64-pc-windows-msvc)
    elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "AMD64")
        set(LIBTOKENIZERS_CARGO_TARGET x86_64-pc-windows-msvc)
    else()
        message(FATAL_ERROR "libtokenizers cannot be compiled for architecture ${CMAKE_SYSTEM_PROCESSOR}")
    endif()
    set(CARGO_EXTRA_ENVS
        "CFLAGS_x86_64_pc_windows_msvc=-isystem ${CMAKE_SYSROOT}/include"
        "CFLAGS_aarch64_pc_windows_msvc=-isystem ${CMAKE_SYSROOT}/include"
        "RUSTFLAGS=-C relocation-model=pic"
    )
else()
    message(FATAL_ERROR "libtokenizers does not support platform: ${CMAKE_SYSTEM_NAME}")
endif()

set(LIBTOKENIZERS_CARGO_FLAGS "")
set(LIBTOKENIZERS_CARGO_TARGET_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(LIBTOKENIZERS_CARGO_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/${LIBTOKENIZERS_CARGO_TARGET}")
list(APPEND LIBTOKENIZERS_CARGO_FLAGS --target ${LIBTOKENIZERS_CARGO_TARGET})

# Build configuration
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(LIBTOKENIZERS_CARGO_BINARY_DIR "${LIBTOKENIZERS_CARGO_BINARY_DIR}/debug")
else()
    list(APPEND LIBTOKENIZERS_CARGO_FLAGS --release)
    set(LIBTOKENIZERS_CARGO_BINARY_DIR "${LIBTOKENIZERS_CARGO_BINARY_DIR}/release")
endif()

# Set output library name
if(MSVC)
    set(TOKENIZERS_RUST_LIB "${LIBTOKENIZERS_CARGO_BINARY_DIR}/tokenizers_c.lib")
else()
    set(TOKENIZERS_RUST_LIB "${LIBTOKENIZERS_CARGO_BINARY_DIR}/libtokenizers_c.a")
endif()

# Extra Emscripten config
set(CARGO_PRE_FLAGS "")
if(CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
    set(CARGO_PRE_FLAGS "+nightly")
    list(APPEND LIBTOKENIZERS_CARGO_FLAGS -Zbuild-std)
endif()

# Collect Rust sources for dependency tracking
file(
    GLOB_RECURSE LIBTOKENIZERS_RUST_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/rs/src/*.rs"
)

# Add Rust build command
add_custom_command(
    OUTPUT ${TOKENIZERS_RUST_LIB}
    COMMAND
    ${CMAKE_COMMAND} -E env
    PATH=$ENV{PATH} # Remove components appended by CMake which refer to iOS sysroot
    CARGO_TARGET_DIR=${LIBTOKENIZERS_CARGO_TARGET_DIR}
    ${CARGO_EXTRA_ENVS}
    ${CARGO_EXECUTABLE} ${CARGO_PRE_FLAGS} build ${LIBTOKENIZERS_CARGO_FLAGS}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/rs
    DEPENDS
    ${LIBTOKENIZERS_RUST_SOURCES}
    "${CMAKE_CURRENT_SOURCE_DIR}/rs/Cargo.toml"
    COMMENT "Building libtokenizers Rust library"
)

# Add libtokenizers
add_custom_target(libtokenizers_build DEPENDS ${TOKENIZERS_RUST_LIB})
add_library(libtokenizers INTERFACE)
add_dependencies(libtokenizers libtokenizers_build)
add_library(libtokenizers::libtokenizers ALIAS libtokenizers)
target_include_directories(libtokenizers INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(libtokenizers INTERFACE ${TOKENIZERS_RUST_LIB})

# Link Windows system libs
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    target_link_libraries(
        libtokenizers INTERFACE
        bcrypt
        userenv
        ntdll
        ws2_32
        kernel32
        advapi32
        user32
    )
endif()

# Add libtokenizers-cpp
add_library(libtokenizers_cpp INTERFACE)
add_library(libtokenizers::libtokenizers_cpp ALIAS libtokenizers_cpp)
target_compile_features(libtokenizers_cpp INTERFACE cxx_std_17)
target_link_libraries(libtokenizers_cpp INTERFACE libtokenizers)

# Add tests
if(LIBTOKENIZERS_BUILD_TESTS)
    enable_testing()
    include(CTest)
    add_subdirectory("test")
endif()