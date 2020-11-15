
# Add the executable - Output name matches target name
add_executable(driver driver.cpp)

#target_link_libraries(driver PRIVATE ${PROJECT_NAME} fmt::fmt)
#target_link_libraries(driver PRIVATE ${PROJECT_NAME} ${CMAKE_SOURCE_DIR}/lib/libsparsestereo.a)
target_link_libraries(driver PRIVATE ${PROJECT_NAME} )

target_include_directories(driver PUBLIC ../include)

#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
