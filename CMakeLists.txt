# == DO NOT EDIT THE FOLLOWING LINES for the Raspberry Pi Pico VS Code Extension to work ==
if(WIN32)
    set(USERHOME $ENV{USERPROFILE})
else()
    set(USERHOME $ENV{HOME})
endif()
set(sdkVersion 2.2.0)
set(toolchainVersion 14_2_Rel1)
set(picotoolVersion 2.2.0-a4)
set(picoVscode ${USERHOME}/.pico-sdk/cmake/pico-vscode.cmake)
if (EXISTS ${picoVscode})
    include(${picoVscode})
endif()
# ====================================================================================
set(PICO_BOARD pico_w CACHE STRING "Board type")

cmake_minimum_required(VERSION 3.13.1)

set(PROJECT_FOLDER .)

include(pico_sdk_import.cmake)

# Pull in FreeRTOS
set(FREERTOS_KERNEL_PATH ${CMAKE_CURRENT_LIST_DIR}/freertos/FreeRTOS-Kernel)
include(freertos/FreeRTOS_Kernel_import.cmake)

project(pico-edge C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)

pico_sdk_init()

add_definitions(-DEIDSP_QUANTIZE_FILTERBANK=0
                -DEI_CLASSIFIER_TFLITE_ENABLE_CMSIS_NN=0
                -DARM_MATH_LOOPUNROLL
                )



add_executable(pico-edge 
              pico-edge/main.cpp
              edge-impulse-sdk/porting/ei_classifier_porting.h
)

include(${PROJECT_FOLDER}/edge-impulse-sdk/cmake/utils.cmake)

target_link_libraries(pico-edge PRIVATE 
    pico_stdlib 
    FreeRTOS-Kernel 
    FreeRTOS-Kernel-Heap4
    hardware_i2c
    )





target_include_directories(pico-edge PRIVATE
    ${PROJECT_FOLDER}/tflite-model
    ${PROJECT_FOLDER}/model-parameters
    ${PROJECT_FOLDER}/freertos
    ${PROJECT_FOLDER}
)



# enable usb output, disable uart output
pico_enable_stdio_usb(pico-edge 0)
pico_enable_stdio_uart(pico-edge 1)


RECURSIVE_FIND_FILE(SOURCE_FILES "${PROJECT_FOLDER}/edge-impulse-sdk" "*.cpp")
RECURSIVE_FIND_FILE(MODEL_FILES "${PROJECT_FOLDER}/tflite-model" "*.cpp")
RECURSIVE_FIND_FILE(CC_FILES "${PROJECT_FOLDER}/edge-impulse-sdk" "*.cc")
RECURSIVE_FIND_FILE(C_FILES "${PROJECT_FOLDER}/edge-impulse-sdk" "*.c")

list(APPEND SOURCE_FILES ${S_FILES})
list(APPEND SOURCE_FILES ${C_FILES})
list(APPEND SOURCE_FILES ${CC_FILES})
list(APPEND SOURCE_FILES ${MODEL_FILES})

# add all sources to the project
target_sources(pico-edge PRIVATE ${SOURCE_FILES})


target_compile_options(pico-edge PRIVATE -Wall)




