include(FetchContent)

#=================== SDL2 ===================
# switch-sdl2 from devkitPro portlibs
find_package(SDL2 REQUIRED)
target_link_libraries(ImGui PUBLIC SDL2::SDL2)
# imgui_impl_opengl3's bundled loader uses dlopen, which the Switch lacks; use switch-glad instead
target_include_directories(ImGui PRIVATE ${DEVKITPRO}/portlibs/switch/include)
target_compile_definitions(ImGui PRIVATE IMGUI_IMPL_OPENGL_LOADER_CUSTOM)
# No fork/exec on Switch for ImGui's "open in shell" helper
target_compile_definitions(ImGui PRIVATE IMGUI_DISABLE_DEFAULT_SHELL_FUNCTIONS)
set_source_files_properties(${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
    PROPERTIES COMPILE_OPTIONS "-include;glad/glad.h")

#=================== uam (deko3d shader compiler, as a library) ===================
# The deko3d renderer compiles the GLSL that prism generates at run time; uam is the only GLSL ->
# DKSH compiler. Vendored with the SwitchWakerHD patches (memory output, no exit/abort, log callback,
# deterministic scheduling, no dual issue): extern/uam/PATCHES.md.
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/extern/uam ${CMAKE_BINARY_DIR}/uam EXCLUDE_FROM_ALL)

# uam embeds Mesa's GLSL compiler, and so does switch-mesa's libEGL (pulled in by SDL for the
# OpenGL fallback): the same symbols twice, from two different Mesa versions. Partially link uam
# into one object and rename everything it defines except its API (uam::*). Renaming (rather than
# localizing) also renames COMDAT group signatures, so the linker never merges uam's inline
# functions/vtables with libEGL's.
set(DKP_BIN ${DEVKITPRO}/devkitA64/bin)
set(UAM_LOCAL_OBJ ${CMAKE_BINARY_DIR}/uam/uam_local.o)
add_custom_command(
    OUTPUT ${UAM_LOCAL_OBJ}
    COMMAND ${DKP_BIN}/aarch64-none-elf-ld -r --whole-archive $<TARGET_FILE:uamlib> -o ${CMAKE_BINARY_DIR}/uam/uam_all.o
    COMMAND sh -c "${DKP_BIN}/aarch64-none-elf-nm --defined-only -g ${CMAKE_BINARY_DIR}/uam/uam_all.o | awk '{print $3}' | grep -v '^_ZN3uam' | awk '{print $1 \" uamp_\" $1}' > ${CMAKE_BINARY_DIR}/uam/uam_rename.txt"
    COMMAND ${DKP_BIN}/aarch64-none-elf-objcopy --redefine-syms=${CMAKE_BINARY_DIR}/uam/uam_rename.txt
            ${CMAKE_BINARY_DIR}/uam/uam_all.o ${UAM_LOCAL_OBJ}
    DEPENDS uamlib
    COMMENT "Renaming uam's private Mesa symbols"
    VERBATIM
)
add_custom_target(uam_local_obj DEPENDS ${UAM_LOCAL_OBJ})
add_library(uam_private STATIC ${UAM_LOCAL_OBJ})
set_source_files_properties(${UAM_LOCAL_OBJ} PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
set_target_properties(uam_private PROPERTIES LINKER_LANGUAGE CXX)
add_dependencies(uam_private uam_local_obj)
target_include_directories(uam_private INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/extern/uam)

#=================== nlohmann-json ===================
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.12.0
    OVERRIDE_FIND_PACKAGE
)
FetchContent_MakeAvailable(nlohmann_json)

#=================== tinyxml2 ===================
# portlibs ships tinyxml2 6.0.0, which is too old; always build our own
set(tinyxml2_BUILD_TESTING OFF)
FetchContent_Declare(
    tinyxml2
    GIT_REPOSITORY https://github.com/leethomason/tinyxml2.git
    GIT_TAG 11.0.0
    OVERRIDE_FIND_PACKAGE
)
FetchContent_MakeAvailable(tinyxml2)

#=================== spdlog ===================
set(SPDLOG_NO_TLS ON CACHE BOOL "" FORCE)
set(SPDLOG_NO_THREAD_ID ON CACHE BOOL "" FORCE)
FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.16.0
    OVERRIDE_FIND_PACKAGE
)
FetchContent_MakeAvailable(spdlog)

#=================== libzip ===================
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
set(BUILD_TOOLS OFF)
set(BUILD_REGRESS OFF)
set(BUILD_EXAMPLES OFF)
set(BUILD_DOC OFF)
set(BUILD_OSSFUZZ OFF)
set(BUILD_SHARED_LIBS OFF)
set(ENABLE_COMMONCRYPTO OFF)
set(ENABLE_GNUTLS OFF)
set(ENABLE_MBEDTLS OFF)
set(ENABLE_OPENSSL OFF)
set(ENABLE_WINDOWS_CRYPTO OFF)
set(ENABLE_LZMA OFF)
set(ENABLE_ZSTD OFF)
FetchContent_Declare(
    libzip
    GIT_REPOSITORY https://github.com/nih-at/libzip.git
    GIT_TAG v1.11.4
    OVERRIDE_FIND_PACKAGE
)
FetchContent_MakeAvailable(libzip)
list(APPEND ADDITIONAL_LIB_INCLUDES ${libzip_SOURCE_DIR}/lib ${libzip_BINARY_DIR})
