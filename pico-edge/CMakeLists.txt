project(pico-edge)

add_executable(pico-edge
        main.c
)

target_link_libraries(pico-edge pico_stdlib freertos )
pico_add_extra_outputs(pico-edge)