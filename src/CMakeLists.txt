add_executable(App
    main.cpp
    AlignedAlloc32.cpp
    Segment.cpp
)
target_include_directories(App 
    PUBLIC ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(App
    PUBLIC ${OpenCV_LIBS}
           OpenMP::OpenMP_CXX
)
