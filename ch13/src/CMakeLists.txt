#生成动态库，供于${PROJECT_SOURCE_DIR}/app/run_kitty_stereo.cpp使用
add_library(myslam SHARED
        frame.cpp
        mappoint.cpp
        map.cpp
        camera.cpp
        config.cpp
        feature.cpp
        frontend.cpp
        backend.cpp
        viewer.cpp
        visual_odometry.cpp
        dataset.cpp)

target_link_libraries(myslam
        ${THIRD_PARTY_LIBS})
