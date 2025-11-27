

#include <hardware/gpio.h>
#include <hardware/uart.h>
#include <pico/stdio.h>
#include <hardware/i2c.h>


// freertos
#include <FreeRTOS.h>
#include <task.h>
#include <semphr.h>
#include <queue.h>
#include <stdlib.h>

// específico
#include "mpu6050.h"
#include "edge-impulse-sdk/classifier/ei_run_classifier.h"



// definindo necessário pra MPU - I2C
static bool debug_nn = false;
const int MPU_ADDRESS = 0x68;
const int I2C_SDA_GPIO = 4;
const int I2C_SCL_GPIO = 5;


static void mpu6050_reset() {
    // Two byte reset. First byte register, second byte data
    // There are a load more options to set up the device in different ways that could be added here
    uint8_t buf[] = {0x6B, 0x00};
    i2c_write_blocking(i2c_default, MPU_ADDRESS, buf, 2, false);
}

static void mpu6050_read_raw(int16_t accel[3], int16_t gyro[3], int16_t *temp) {
    // For this particular device, we send the device the register we want to read
    // first, then subsequently read from the device. The register is auto incrementing
    // so we don't need to keep sending the register we want, just the first.

    uint8_t buffer[6];

    // Start reading acceleration registers from register 0x3B for 6 bytes
    uint8_t val = 0x3B;
    i2c_write_blocking(i2c_default, MPU_ADDRESS, &val, 1, true); // true to keep master control of bus
    i2c_read_blocking(i2c_default, MPU_ADDRESS, buffer, 6, false);

    for (int i = 0; i < 3; i++) {
        accel[i] = (buffer[i * 2] << 8 | buffer[(i * 2) + 1]);
    }

    // Now gyro data from reg 0x43 for 6 bytes
    // The register is auto incrementing on each read
    val = 0x43;
    i2c_write_blocking(i2c_default, MPU_ADDRESS, &val, 1, true);
    i2c_read_blocking(i2c_default, MPU_ADDRESS, buffer, 6, false);  // False - finished with bus

    for (int i = 0; i < 3; i++) {
        gyro[i] = (buffer[i * 2] << 8 | buffer[(i * 2) + 1]);;
    }

    // Now temperature from reg 0x41 for 2 bytes
    // The register is auto incrementing on each read
    val = 0x41;
    i2c_write_blocking(i2c_default, MPU_ADDRESS, &val, 1, true);
    i2c_read_blocking(i2c_default, MPU_ADDRESS, buffer, 2, false);  // False - finished with bus

    *temp = buffer[0] << 8 | buffer[1];
}



static const float features[] = {
    // Copy raw features here (e.g. from the 'Model testing' page)
};

static void gesture_recognize_task(void *p)
{
    gpio_init(15);
    gpio_set_dir(15, GPIO_OUT);
    gpio_put(15,0);

    gpio_init(16);
    gpio_set_dir(16, GPIO_OUT);
    gpio_put(16,0);

    gpio_init(17);
    gpio_set_dir(17, GPIO_OUT);
    gpio_put(17,0);
    // uint slice_num = pwm_gpio_to_slice_num(PICO_DEFAULT_LED_PIN);
    // Initialize I2C port 0 and configuring Pins 0 and 1 for MPU6050
    i2c_init(i2c_default, 400 * 1000);
    gpio_set_function(I2C_SDA_GPIO, GPIO_FUNC_I2C);
    gpio_set_function(I2C_SCL_GPIO, GPIO_FUNC_I2C);
    gpio_pull_up(I2C_SDA_GPIO);
    gpio_pull_up(I2C_SCL_GPIO);
    mpu6050_reset();



    int16_t accelerometer[3],gyro[3],temp;

    // 3 entradas
    if (EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME != 3) {
        ei_printf("ERR: EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME should be equal to 3 (the 3 sensor axes)\n");
    }

    while (true) 
    
    {
        ei_printf("\nStarting inferencing in 2 seconds...\n");
        vTaskDelay(pdMS_TO_TICKS(2000));
        ei_printf("Sampling...\n");

    // Allocate a buffer here for the values we'll read from the IMU
    float buffer[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE] = { 0 };

    for (size_t ix = 0; ix < EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE; ix += 3) {
        // Determine the next tick (and then sleep later)
        //uint64_t next_tick = micros() + (EI_CLASSIFIER_INTERVAL_MS * 1000);
        mpu6050_read_raw(accelerometer,gyro,&temp);
        buffer[ix + 0]= accelerometer[0];
        buffer[ix + 1]= accelerometer[1];
        buffer[ix + 2]= accelerometer[2];

        
        //IMU.readAcceleration(buffer[ix], buffer[ix + 1], buffer[ix + 2]);
    }

    signal_t signal;
    int err = numpy::signal_from_buffer(buffer, EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE, &signal);
    if (err != 0) {
        ei_printf("Failed to create signal from buffer (%d)\n", err);
        break;
    }

    // Run the classifier
    ei_impulse_result_t result = { 0 };

    err = run_classifier(&signal, &result, debug_nn);
    if (err != EI_IMPULSE_OK) {
        ei_printf("ERR: Failed to run classifier (%d)\n", err);
        break;
    }

    // print the predictions
    ei_printf("Predictions ");
    ei_printf("(DSP: %d ms., Classification: %d ms., Anomaly: %d ms.)",
        result.timing.dsp, result.timing.classification, result.timing.anomaly);
    ei_printf(": \n");
    
    if (result.classification[0].value > result.classification[1].value && result.classification[0].value > result.classification[2].value) {
        gpio_put(15, 1);
        gpio_put(16, 0);
        gpio_put(17, 0);
    } else if (result.classification[1].value > result.classification[0].value && result.classification[1].value > result.classification[2].value) {
        gpio_put(15, 0);
        gpio_put(16, 1);
        gpio_put(17, 0);
    } else if (result.classification[2].value > result.classification[0].value && result.classification[2].value > result.classification[1].value) {
        gpio_put(15, 0);
        gpio_put(16, 0);
        gpio_put(17, 1);
    }
    for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
        ei_printf("teste    %s: %.5f\n", result.classification[ix].label, result.classification[ix].value);
    }
    
    
#if EI_CLASSIFIER_HAS_ANOMALY == 1
    ei_printf("    anomaly score: %.3f\n", result.anomaly);
#endif

    }
    vTaskDelay(pdMS_TO_TICKS(10));
}



int raw_feature_get_data(size_t offset, size_t length, float *out_ptr) {
  memcpy(out_ptr, features + offset, length * sizeof(float));
  return 0;
}

void print_inference_result(ei_impulse_result_t result) {

    // Print how long it took to perform inference
    ei_printf("Timing: DSP %d ms, inference %d ms, anomaly %d ms\r\n",
            result.timing.dsp,
            result.timing.classification,
            result.timing.anomaly);

    // Print the prediction results (object detection)
#if EI_CLASSIFIER_OBJECT_DETECTION == 1
    ei_printf("Object detection bounding boxes:\r\n");
    for (uint32_t i = 0; i < result.bounding_boxes_count; i++) {
        ei_impulse_result_bounding_box_t bb = result.bounding_boxes[i];
        if (bb.value == 0) {
            continue;
        }
        ei_printf("  %s (%f) [ x: %u, y: %u, width: %u, height: %u ]\r\n",
                bb.label,
                bb.value,
                bb.x,
                bb.y,
                bb.width,
                bb.height);
    }

    // Print the prediction results (classification)
#else
    ei_printf("Predictions:\r\n");
    for (uint16_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
        ei_printf("  %s: ", ei_classifier_inferencing_categories[i]);
        ei_printf("%.5f\r\n", result.classification[i].value);
    }
#endif

    // Print anomaly result (if it exists)
#if EI_CLASSIFIER_HAS_ANOMALY == 1
    ei_printf("Anomaly prediction: %.3f\r\n", result.anomaly);
#endif

}


int main() {
    //stdio_usb_init();
    stdio_init_all();

    

    xTaskCreate(gesture_recognize_task, "gesture_task 1", 8192, NULL, 1, NULL);

    vTaskStartScheduler();

    while (true)
        ;
}
