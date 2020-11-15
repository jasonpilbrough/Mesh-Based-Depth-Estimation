
#set(HEADER_LIST
#"${Sandbox_SOURCE_DIR}/include/sandbox.h"
#	${PROJECT_SOURCE_DIR}/include/sandbox.h
#)

set(MY_SRCS
	pipeline.cpp
	stereo_matching.cpp
	framemanager.cpp
	triangle.cpp
	delaunay.cpp
	colourmap.cpp
	draw_utils.cpp
	regularisers.cpp

)

#add_library(${PROJECT_NAME} SHARED ${MY_SRCS} ${HEADER_LIST})
add_library(${PROJECT_NAME} SHARED ${MY_SRCS})

#add header files
target_include_directories(${PROJECT_NAME} PUBLIC ../../include)

target_link_libraries(${PROJECT_NAME}
  ${OpenCV_LIBS}
  ${Boost_LIBRARIES})

set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)
