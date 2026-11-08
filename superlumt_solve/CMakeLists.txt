cmake_minimum_required(VERSION 3.21)
project(SUPERLUMT_SOLVE LANGUAGES CXX)

find_package(Threads REQUIRED)
if(NOT NATIVE_PREFIX)
  message(FATAL_ERROR "NATIVE_PREFIX is required")
endif()

add_library(superlumt_solve SHARED superlumt_solve.cpp complex_interface.cpp)
target_compile_features(superlumt_solve PRIVATE cxx_std_17)
target_compile_definitions(superlumt_solve PRIVATE __PTHREAD)
target_include_directories(superlumt_solve PRIVATE
  "${NATIVE_PREFIX}/include" "${NATIVE_PREFIX}/include/superlu_mt")
foreach(component superlu_mt_PTHREAD blas metis GKlib)
  set(archive "${NATIVE_PREFIX}/lib/lib${component}.a")
  if(NOT EXISTS "${archive}")
    message(FATAL_ERROR "Required static PIC archive is missing: ${archive}")
  endif()
  target_link_libraries(superlumt_solve PRIVATE "${archive}")
endforeach()
target_link_libraries(superlumt_solve PRIVATE Threads::Threads ${CMAKE_DL_LIBS} m)

if(APPLE)
  target_link_options(superlumt_solve PRIVATE
    "LINKER:-exported_symbols_list,${CMAKE_CURRENT_SOURCE_DIR}/exported_symbols_list.txt")
  set_target_properties(superlumt_solve PROPERTIES INSTALL_NAME_DIR "@rpath")
else()
  target_link_options(superlumt_solve PRIVATE
    "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exported_version_script.txt"
    "LINKER:--no-undefined")
endif()
set_target_properties(superlumt_solve PROPERTIES POSITION_INDEPENDENT_CODE ON)
install(TARGETS superlumt_solve DESTINATION .)
