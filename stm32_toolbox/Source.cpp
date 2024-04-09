#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#include "network.h"
#include "network_data.h"

#include <chrono>
#include <iostream>
using namespace std::chrono;

using namespace std;

/**
 * @brief Statically allocated buffers.
 * Buffers can be dynamically allocated using malloc and size information
 * given by the report in ai_network_get_report().
 */
ai_u8 activations[AI_NETWORK_DATA_ACTIVATIONS_SIZE];
ai_u8 in_data[AI_NETWORK_IN_1_SIZE_BYTES];
ai_u8 out_data[AI_NETWORK_OUT_1_SIZE_BYTES];

/* AI buffer IO handlers */
ai_buffer* ai_input;
ai_buffer* ai_output;
/*
float dequantize(int x, int x_zero, float x_scale)
{
    return (x - x_zero) * x_scale;
}

template<typename T>
vector<vector<vector<T>>> reshape(T* data, uint_fast8_t c = 21, uint_fast8_t w = 12, uint_fast8_t h = 12)
{
    std::vector<std::vector<std::vector<T>>> tensor(c, std::vector<std::vector<T>>(w, std::vector<T>(h)));

    // Reshape the vector into the tensor
    int index = 0;
    for (int i = 0; i < c; ++i) {
        for (int j = 0; j < w; ++j) {
            for (int k = 0; k < h; ++k) {
                tensor[i][j][k] = data[index++];
            }
        }
    }
    return tensor;
}*/

int main()
{

    for (int coun = 0; coun < 10; coun++) {
        ai_handle network = AI_HANDLE_NULL;
        ai_error err;
        ai_network_report report;

        /** @brief Initialize network */
        const ai_handle acts[] = { activations };
        auto stop = high_resolution_clock::now();

        err = ai_network_create_and_init(&network, acts, NULL);
        if (err.type != AI_ERROR_NONE) {
            printf("ai init_and_create error\n");
            return -1;
        }
        std::cout << "time to load model " << duration_cast<microseconds>(high_resolution_clock::now() - stop).count() << "�s\n";

        /** @brief {optional} for debug/log purpose */
        if (ai_network_get_report(network, &report) != true) {
            printf("ai get report error\n");
            return -1;
        }

        printf("Model name      : %s\n", report.model_name);
        printf("Model signature : %s\n", report.model_signature);

        ai_input = &report.inputs[0];
        ai_output = &report.outputs[0];
        printf("input[0] : (%d, %d, %d)\n", AI_BUFFER_SHAPE_ELEM(ai_input, AI_SHAPE_HEIGHT),
            AI_BUFFER_SHAPE_ELEM(ai_input, AI_SHAPE_WIDTH),
            AI_BUFFER_SHAPE_ELEM(ai_input, AI_SHAPE_CHANNEL));
        printf("output[0] : (%d, %d, %d)\n", AI_BUFFER_SHAPE_ELEM(ai_output, AI_SHAPE_HEIGHT),
            AI_BUFFER_SHAPE_ELEM(ai_output, AI_SHAPE_WIDTH),
            AI_BUFFER_SHAPE_ELEM(ai_output, AI_SHAPE_CHANNEL));

        /** @brief Fill input buffer with random values */
        stop = high_resolution_clock::now();
        srand(1);
        for (int i = 0; i < AI_NETWORK_IN_1_SIZE; i++) {
            in_data[i] = rand() % 0xFFFF;
        }

        std::cout << "time to load data with rand " << duration_cast<microseconds>(high_resolution_clock::now() - stop).count() << "�s\n";


        /** @brief Normalize, convert and/or quantize inputs if necessary... */

        /** @brief Perform inference */
        ai_i32 n_batch;

        /** @brief Create the AI buffer IO handlers
         *  @note  ai_inuput/ai_output are already initilaized after the
         *         ai_network_get_report() call. This is just here to illustrate
         *         the case where get_report() is not called.
         */
        ai_input = ai_network_inputs_get(network, NULL);
        ai_output = ai_network_outputs_get(network, NULL);

        /** @brief Set input/output buffer addresses */
        ai_input[0].data = AI_HANDLE_PTR(in_data);
        ai_output[0].data = AI_HANDLE_PTR(out_data);

        stop = high_resolution_clock::now();
        /** @brief Perform the inference */
        n_batch = ai_network_run(network, &ai_input[0], &ai_output[0]);

        std::cout << "Execution time " << duration_cast<microseconds>(high_resolution_clock::now() - stop).count() << "�s\n";

        if (n_batch != 1) {
            err = ai_network_get_error(network);
            printf("ai run error %d, %d\n", err.type, err.code);
            return -1;
        }

        /** @brief Post-process the output results/predictions */
        /*printf("Inference output..\n");
        double output[AI_NETWORK_OUT_1_SIZE];

        for (int i = 0; i < AI_NETWORK_OUT_1_SIZE; i++) {
            output[i] = dequantize(out_data[i], 73, 0.2761317491531372);
        }

        auto f = reshape(output);*/
    }
    return 0;
}