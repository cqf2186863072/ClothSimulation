#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SPIRV-Tools-lint" for configuration "Release"
set_property(TARGET SPIRV-Tools-lint APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(SPIRV-Tools-lint PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/SPIRV-Tools-lint.lib"
  )

list(APPEND _cmake_import_check_targets SPIRV-Tools-lint )
list(APPEND _cmake_import_check_files_for_SPIRV-Tools-lint "${_IMPORT_PREFIX}/lib/SPIRV-Tools-lint.lib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
