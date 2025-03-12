#ifndef SCNN_H
#define SCNN_H

#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

// Neural Network Structure
typedef struct {
    int input_size;
    int* hidden_sizes;
    int num_hidden_layers;
    int output_size;
    double** weights;  // weights[0] = input->hidden1, weights[1] = hidden1->hidden2, etc.
    double** biases;   // biases[0] = hidden1, biases[1] = hidden2, etc.
    double *bias_o;
    int trained_epochs;
} scnn_NeuralNetwork;

// Tokenizer Structure
typedef struct {
    char** vocab;
    int* vocab_sizes;
    int vocab_count;
    int max_token_length;
    char** token_map;
    int token_map_size;
} scnn_Tokenizer;

// Function declarations
scnn_NeuralNetwork* scnn_create(int input_size, int num_hidden_layers, int* hidden_sizes, int output_size);
void scnn_train(scnn_NeuralNetwork* net, double* input, double* target, double learning_rate);
double* scnn_predict(scnn_NeuralNetwork* net, double* input);
double* scnn_predict_temp(scnn_NeuralNetwork* net, double* input, double temperature);
void scnn_free(scnn_NeuralNetwork* net);
int scnn_round_output(double value);
double scnn_sigmoid(double x);
double scnn_sigmoid_derivative(double x);
void scnn_randomize(double* arr, int size);

// Tokenizer functions
scnn_Tokenizer* scnn_tokenizer_create(const char* vocab_file);
void scnn_tokenizer_free(scnn_Tokenizer* tokenizer);
int* scnn_tokenizer_encode(scnn_Tokenizer* tokenizer, const char* text, int* output_length);
char* scnn_tokenizer_decode(scnn_Tokenizer* tokenizer, const int* tokens, int token_count);
int scnn_tokenizer_size(scnn_Tokenizer* tokenizer);

// New function declarations
void scnn_save_model(scnn_NeuralNetwork* net, const char* filename);
scnn_NeuralNetwork* scnn_load_model(const char* filename);
void scnn_train_image(scnn_NeuralNetwork* net, const char* image_path, double* target, double learning_rate);

#endif 