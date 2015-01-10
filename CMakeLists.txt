# Author: Bo Sun
# Afflication: TAMS, University of Hamburg
# E-Mail: bosun@informatik.uni-hamburg.de
#         user_mail@QQ.com


cmake_minimum_required (VERSION 2.8)

project (tams_feature3.0)

find_package(PCL 1.7 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (tams_feature3.0 tams_feature.cpp)

target_link_libraries (tams_feature3.0  ${PCL_LIBRARIES})
