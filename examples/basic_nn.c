#include <stdio.h>
#include <stdlib.h>
#include "../include/scnn.h"

int main() {
    // Create a neural network with multiple hidden layers
    int hidden_sizes[] = {4, 3}; // Two hidden layers with 4 and 3 neurons respectively
    scnn_NeuralNetwork* net = scnn_create(2, 2, hidden_sizes, 1);
    
    // Training data (XOR problem)
    double inputs[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    double targets[4] = {0, 1, 1, 0};
    
    // Train the network
    for (int i = 0; i < 10000; i++) {
        for (int j = 0; j < 4; j++) {
            scnn_train(net, inputs[j], &targets[j], 0.1);
        }
    }
    
    // Save model
    scnn_save_model(net, "xor_model.bin");
    
    // Load model
    scnn_NeuralNetwork* loaded_net = scnn_load_model("xor_model.bin");
    
    // Test the network
    printf("Testing XOR problem:\n");
    for (int i = 0; i < 4; i++) {
        // Test with normal prediction
        double* output = scnn_predict(net, inputs[i]);
        printf("Input: %d %d, Normal Output: %f, Rounded: %d\n", 
               (int)inputs[i][0], (int)inputs[i][1], *output, scnn_round_output(*output));
        free(output);
        
        // Test with temperature prediction
        double* temp_output = scnn_predict_temp(net, inputs[i], 0.5); // Lower temperature
        printf("Input: %d %d, Temp Output (0.5): %f, Rounded: %d\n", 
               (int)inputs[i][0], (int)inputs[i][1], *temp_output, scnn_round_output(*temp_output));
        free(temp_output);
    }
    
    // Clean up
    scnn_free(net);
    scnn_free(loaded_net);
    return 0;
} 