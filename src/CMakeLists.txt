add_executable(App
    main.cpp
    aligned_alloc.cpp
    segment.cpp
)
target_include_directories(App 
    PUBLIC ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(App
    PUBLIC ${OpenCV_LIBS}
           OpenMP::OpenMP_CXX
)
