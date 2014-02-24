
cmake_minimum_required(VERSION 2.6.0)

# This project is designed to be built outside the Insight source tree.
set (NAME instant_module_315ebc21e0fe220e611965734859ae349eef07f9)


# Configuration for package UFC
FIND_PACKAGE(UFC REQUIRED)
IF(UFC_FOUND)
 INCLUDE(${UFC_USE_FILE})
ENDIF(UFC_FOUND)

if (NOT $ENV{CXX})
  set(CMAKE_CXX_COMPILER ${UFC_CXX_COMPILER})
endif()


PROJECT(${NAME})

find_package(SWIG REQUIRED)
include(${SWIG_USE_FILE})

set(SWIG_MODULE_NAME ${NAME})
set(CMAKE_SWIG_FLAGS
  -module ${SWIG_MODULE_NAME}
  -shadow
  -modern
  -modernargs
  -fastdispatch
  -fvirtual
  -nosafecstrings
  -noproxydel
  -fastproxy
  -fastinit
  -fastunpack
  -fastquery
  -nobuildnone
${UFC_CXX_DEFINITIONS}
  )

set(CMAKE_SWIG_OUTDIR ${CMAKE_CURRENT_BINARY_DIR})

set(SWIG_SOURCES ${NAME}.i)

set_source_files_properties(${SWIG_SOURCES} PROPERTIES CPLUSPLUS ON)

include_directories(${UFC_PYTHON_INCLUDE_DIRS} ${${NAME}_SOURCE_DIR})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${UFC_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${UFC_LINK_FLAGS}")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${UFC_LINK_FLAGS}")


swig_add_module(${SWIG_MODULE_NAME} python ${SWIG_SOURCES})

if (DEFINED UFC_LIBRARIES OR DEFINED UFC_3RD_PARTY_LIBRARIES OR DEFINED UFC_PYTHON_LIBRARIES)
  swig_link_libraries(${SWIG_MODULE_NAME} ${UFC_LIBRARIES} ${UFC_3RD_PARTY_LIBRARIES} ${UFC_PYTHON_LIBRARIES})
endif()

