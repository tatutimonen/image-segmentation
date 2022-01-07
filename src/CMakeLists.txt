add_executable(App
    main.cpp
    aligned_alloc.cpp
    segment.cpp
)
target_include_directories(App 
    PRIVATE ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(App
    PRIVATE ${OpenCV_LIBS}
            OpenMP::OpenMP_CXX
)
