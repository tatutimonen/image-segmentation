add_executable(App
    main.cpp
    AlignedAlloc32.cpp
    Segment.cpp
)
target_include_directories(App 
    PRIVATE ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(App
    PRIVATE ${OpenCV_LIBS}
            OpenMP::OpenMP_CXX
)
