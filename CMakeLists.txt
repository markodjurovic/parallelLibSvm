
cmake_minimum_required(VERSION 2.6)
project(parallelLibSvm CXX)

option(DEBUG_BUILD "Build in debug mode" OFF)
MESSAGE( STATUS "APPLICATION DEBUG OPTION:         " ${DEBUG_BUILD})

option(USE_OPENCL "Build with OpenCL support" ON)
MESSAGE( STATUS "BUILD WITH OPENCL SUPPORT:         " ${USE_OPENCL})

option(USE_AMD "Build for AMD graphics cards" ON)
MESSAGE( STATUS "BUILD AMD SUPPORT:         " ${USE_AMD})

FIND_PACKAGE( PackageHandleStandardArgs )

set (OPENCL_INCLUDE_PATH_DEFAULT /opt/AMDAPPSDK-2.9-1/include)

find_path(OPENCL_INCLUDE /CL/cl.h PATHS ${OPENCL_INCLUDE_PATH_DEFAULT} NO_DEFAULT_PATH)
if (OPENCL_INCLUDE)
  MESSAGE( STATUS "OPENCL INCLUDE DIRS:         " ${OPENCL_INCLUDE})
else()
  MESSAGE( FATAL_ERROR "OPENCL INCLUDE DIRS:         " ${OPENCL_INCLUDE})
endif(OPENCL_INCLUDE)

find_library(OPENCL_LIB OpenCL PATHS /opt/AMDAPPSDK-2.9-1/lib/x86_64 NO_DEFAULT_PATH) 
if (OPENCL_LIB)
  MESSAGE( STATUS "OPENCL LIB DIR:         " ${OPENCL_LIB})
else()
  MESSAGE( FATAL_ERROR "OPENCL LIB DIR:         " ${OPENCL_LIB})
endif(OPENCL_LIB)  
  
set (IDIRS "./")

MESSAGE( STATUS "INCLUDE DIRS:         " ${IDIRS})

set( CMAKE_CXX_FLAGS "-msse2 -m64 -std=c++11 -fopenmp" )

if (DEBUG_BUILD)
  set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g -Wall -D_DEBUG" )
else()
  set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3" )
endif(DEBUG_BUILD)

if (USE_OPENCL)
  set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_OPENCL" )
endif(USE_OPENCL)

if (USE_AMD)
  set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_AMD" )
endif(USE_AMD)

MESSAGE( STATUS "COMPILER PARAMETERS:         " ${CMAKE_CXX_FLAGS})

set( CMAKE_SHARED_LINKER_FLAGS "-m64" )

include_directories (${IDIRS} ${OPENCL_INCLUDE})

add_executable(parallelLibSvm core/util/MemTracker.cpp core/util/PredictorFactory.cpp core/util/FileSaver.h core/util/rtti/ObjectFactory.h core/util/rtti/RTTI.cpp core/util/rtti/ObjectFactory.cpp core/util/rtti/RTTI.h core/util/rtti/RTTIStorage.cpp core/util/rtti/RTTIStorage.h core/util/Singleton.h core/util/Config.h core/util/TabParser.cpp core/util/Matrix.h core/util/MemTracker.h core/util/Timer.h core/util/TabParser.h core/util/PredictorFactory.h core/util/Timer.cpp core/util/Cofig.cpp core/util/prediction/IPrediction.h core/util/prediction/libsvm/SvmPredict.cpp core/util/prediction/libsvm/SvmPredict.h core/util/raii/RAIIS.h core/process/IProcessor.h core/svm/svm-train.cpp core/svm/svm-train.h core/opencl/OpenClToolsBase.cpp core/opencl/OpenClToolsBase.h core/opencl/OpenCLTools.h core/opencl/OpenCLTools.cpp core/opencl/libsvm/OpenCLToolsTrain.cpp core/opencl/libsvm/OpenCLToolsPredict.cpp core/opencl/libsvm/OpenCLToolsPredict.h core/opencl/libsvm/OpenCLToolsTrain.h main.cpp typedefs.h thirdparty/lib_svm/svm.cpp thirdparty/lib_svm/svm.h)

get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
foreach(dir ${dirs})
  message(STATUS "dir='${dir}'")
endforeach()

target_link_libraries(parallelLibSvm ${OPENCL_LIB} rt)

install(TARGETS parallelLibSvm RUNTIME DESTINATION bin)