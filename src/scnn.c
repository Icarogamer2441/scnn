#include "scnn.h"
#include <stdio.h>
#include <string.h>

// Create a new neural network
scnn_NeuralNetwork* scnn_create(int input_size, int num_hidden_layers, int* hidden_sizes, int output_size) {
    scnn_NeuralNetwork* net = malloc(sizeof(scnn_NeuralNetwork));
    net->input_size = input_size;
    net->num_hidden_layers = num_hidden_layers;
    net->hidden_sizes = malloc(num_hidden_layers * sizeof(int));
    memcpy(net->hidden_sizes, hidden_sizes, num_hidden_layers * sizeof(int));
    net->output_size = output_size;
    
    // Allocate memory for weights and biases between layers
    net->weights = malloc((num_hidden_layers + 1) * sizeof(double*));
    net->biases = malloc(num_hidden_layers * sizeof(double*));
    
    // Input to first hidden layer
    net->weights[0] = malloc(input_size * hidden_sizes[0] * sizeof(double));
    net->biases[0] = malloc(hidden_sizes[0] * sizeof(double));
    
    // Hidden layers
    for (int i = 1; i < num_hidden_layers; i++) {
        net->weights[i] = malloc(hidden_sizes[i-1] * hidden_sizes[i] * sizeof(double));
        net->biases[i] = malloc(hidden_sizes[i] * sizeof(double));
    }
    
    // Last hidden to output layer
    net->weights[num_hidden_layers] = malloc(hidden_sizes[num_hidden_layers-1] * output_size * sizeof(double));
    net->bias_o = malloc(output_size * sizeof(double));
    
    // Initialize weights and biases with random values
    scnn_randomize(net->weights[0], input_size * hidden_sizes[0]);
    for (int i = 1; i < num_hidden_layers; i++) {
        scnn_randomize(net->weights[i], hidden_sizes[i-1] * hidden_sizes[i]);
        scnn_randomize(net->biases[i], hidden_sizes[i]);
    }
    scnn_randomize(net->weights[num_hidden_layers], hidden_sizes[num_hidden_layers-1] * output_size);
    scnn_randomize(net->biases[0], hidden_sizes[0]);
    scnn_randomize(net->bias_o, output_size);
    
    return net;
}

// Train the neural network
void scnn_train(scnn_NeuralNetwork* net, double* input, double* target, double learning_rate) {
    // Allocate memory for activations
    double** activations = malloc((net->num_hidden_layers + 1) * sizeof(double*));
    for (int i = 0; i < net->num_hidden_layers; i++) {
        activations[i] = malloc(net->hidden_sizes[i] * sizeof(double));
    }
    activations[net->num_hidden_layers] = malloc(net->output_size * sizeof(double));
    
    // Forward propagation
    // Input to first hidden layer
    for (int h = 0; h < net->hidden_sizes[0]; h++) {
        activations[0][h] = 0;
        for (int i = 0; i < net->input_size; i++) {
            activations[0][h] += input[i] * net->weights[0][i * net->hidden_sizes[0] + h];
        }
        activations[0][h] = scnn_sigmoid(activations[0][h] + net->biases[0][h]);
    }
    
    // Hidden layers
    for (int l = 1; l < net->num_hidden_layers; l++) {
        for (int h = 0; h < net->hidden_sizes[l]; h++) {
            activations[l][h] = 0;
            for (int p = 0; p < net->hidden_sizes[l-1]; p++) {
                activations[l][h] += activations[l-1][p] * net->weights[l][p * net->hidden_sizes[l] + h];
            }
            activations[l][h] = scnn_sigmoid(activations[l][h] + net->biases[l][h]);
        }
    }
    
    // Output layer
    for (int o = 0; o < net->output_size; o++) {
        activations[net->num_hidden_layers][o] = 0;
        for (int h = 0; h < net->hidden_sizes[net->num_hidden_layers-1]; h++) {
            activations[net->num_hidden_layers][o] += 
                activations[net->num_hidden_layers-1][h] * 
                net->weights[net->num_hidden_layers][h * net->output_size + o];
        }
        activations[net->num_hidden_layers][o] = activations[net->num_hidden_layers][o] + net->bias_o[o]; // Linear activation
    }
    
    // Backpropagation
    double** errors = malloc((net->num_hidden_layers + 1) * sizeof(double*));
    for (int i = 0; i < net->num_hidden_layers; i++) {
        errors[i] = malloc(net->hidden_sizes[i] * sizeof(double));
    }
    errors[net->num_hidden_layers] = malloc(net->output_size * sizeof(double));
    
    // Calculate output layer errors
    for (int o = 0; o < net->output_size; o++) {
        errors[net->num_hidden_layers][o] = (target[o] - activations[net->num_hidden_layers][o]); // No derivative for linear activation
    }
    
    // Calculate hidden layer errors (backwards)
    for (int l = net->num_hidden_layers - 1; l >= 0; l--) {
        for (int h = 0; h < net->hidden_sizes[l]; h++) {
            errors[l][h] = 0;
            if (l == net->num_hidden_layers - 1) {
                // Last hidden layer
                for (int o = 0; o < net->output_size; o++) {
                    errors[l][h] += errors[l+1][o] * net->weights[l+1][h * net->output_size + o];
                }
            } else {
                // Intermediate hidden layers
                for (int n = 0; n < net->hidden_sizes[l+1]; n++) {
                    errors[l][h] += errors[l+1][n] * net->weights[l+1][h * net->hidden_sizes[l+1] + n];
                }
            }
            errors[l][h] *= scnn_sigmoid_derivative(activations[l][h]);
        }
    }
    
    // Update weights and biases
    // Update output layer weights
    for (int h = 0; h < net->hidden_sizes[net->num_hidden_layers-1]; h++) {
        for (int o = 0; o < net->output_size; o++) {
            net->weights[net->num_hidden_layers][h * net->output_size + o] += 
                learning_rate * errors[net->num_hidden_layers][o] * activations[net->num_hidden_layers-1][h];
        }
    }
    
    // Update hidden layer weights
    for (int l = net->num_hidden_layers - 1; l >= 0; l--) {
        for (int i = 0; i < (l == 0 ? net->input_size : net->hidden_sizes[l-1]); i++) {
            for (int h = 0; h < net->hidden_sizes[l]; h++) {
                net->weights[l][i * net->hidden_sizes[l] + h] += 
                    learning_rate * errors[l][h] * (l == 0 ? input[i] : activations[l-1][i]);
            }
        }
    }
    
    // Update biases
    for (int o = 0; o < net->output_size; o++) {
        net->bias_o[o] += learning_rate * errors[net->num_hidden_layers][o];
    }
    
    for (int l = 0; l < net->num_hidden_layers; l++) {
        for (int h = 0; h < net->hidden_sizes[l]; h++) {
            net->biases[l][h] += learning_rate * errors[l][h];
        }
    }
    
    // Free memory
    for (int i = 0; i < net->num_hidden_layers + 1; i++) {
        free(errors[i]);
    }
    free(errors);
    
    for (int i = 0; i < net->num_hidden_layers + 1; i++) {
        free(activations[i]);
    }
    free(activations);
}

// Predict output for given input
double* scnn_predict(scnn_NeuralNetwork* net, double* input) {
    return scnn_predict_temp(net, input, 1.0); // Default temperature of 1.0
}

// Predict with temperature
double* scnn_predict_temp(scnn_NeuralNetwork* net, double* input, double temperature) {
    double* output = malloc(net->output_size * sizeof(double));
    
    // Allocate memory for activations
    double** activations = malloc(net->num_hidden_layers * sizeof(double*));
    for (int i = 0; i < net->num_hidden_layers; i++) {
        activations[i] = malloc(net->hidden_sizes[i] * sizeof(double));
    }
    
    // Forward propagation
    // Input to first hidden layer
    for (int h = 0; h < net->hidden_sizes[0]; h++) {
        activations[0][h] = 0;
        for (int i = 0; i < net->input_size; i++) {
            activations[0][h] += input[i] * net->weights[0][i * net->hidden_sizes[0] + h];
        }
        activations[0][h] = scnn_sigmoid((activations[0][h] + net->biases[0][h]) / temperature);
    }
    
    // Hidden layers
    for (int l = 1; l < net->num_hidden_layers; l++) {
        for (int h = 0; h < net->hidden_sizes[l]; h++) {
            activations[l][h] = 0;
            for (int p = 0; p < net->hidden_sizes[l-1]; p++) {
                activations[l][h] += activations[l-1][p] * net->weights[l][p * net->hidden_sizes[l] + h];
            }
            activations[l][h] = scnn_sigmoid((activations[l][h] + net->biases[l][h]) / temperature);
        }
    }
    
    // Output layer
    for (int o = 0; o < net->output_size; o++) {
        output[o] = 0;
        for (int h = 0; h < net->hidden_sizes[net->num_hidden_layers-1]; h++) {
            output[o] += activations[net->num_hidden_layers-1][h] * 
                        net->weights[net->num_hidden_layers][h * net->output_size + o];
        }
        output[o] = (output[o] + net->bias_o[o]) / temperature; // Linear activation
    }
    
    // Free memory
    for (int i = 0; i < net->num_hidden_layers; i++) {
        free(activations[i]);
    }
    free(activations);
    
    return output;
}

// Free memory allocated for the neural network
void scnn_free(scnn_NeuralNetwork* net) {
    if (net) {
        // Free weights
        for (int i = 0; i < net->num_hidden_layers + 1; i++) {
            if (net->weights[i]) {
                free(net->weights[i]);
            }
        }
        if (net->weights) {
            free(net->weights);
        }
        
        // Free biases
        for (int i = 0; i < net->num_hidden_layers; i++) {
            if (net->biases[i]) {
                free(net->biases[i]);
            }
        }
        if (net->biases) {
            free(net->biases);
        }
        
        // Free other memory
        if (net->hidden_sizes) {
            free(net->hidden_sizes);
        }
        if (net->bias_o) {
            free(net->bias_o);
        }
        
        free(net);
    }
}

// Activation function
double scnn_sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

// Derivative of activation function
double scnn_sigmoid_derivative(double x) {
    return x * (1 - x);
}

// Randomize array values
void scnn_randomize(double* arr, int size) {
    srand(time(NULL));
    for (int i = 0; i < size; i++) {
        arr[i] = ((double)rand() / RAND_MAX) * 2 - 1; // Random value between -1 and 1
    }
}

// Modified tokenizer creation function
scnn_Tokenizer* scnn_tokenizer_create(const char* vocab_file) {
    scnn_Tokenizer* tokenizer = malloc(sizeof(scnn_Tokenizer));
    if (!tokenizer) {
        fprintf(stderr, "Memory allocation failed for tokenizer\n");
        return NULL;
    }
    
    // Try to open vocabulary file
    FILE* vocab_fp = fopen(vocab_file, "r");
    if (!vocab_fp) {
        fprintf(stderr, "Failed to open vocabulary file: %s\n", vocab_file);
        perror("Error");
        free(tokenizer);
        return NULL;
    }
    
    // Count lines in vocabulary file
    int vocab_count = 0;
    char ch;
    while(!feof(vocab_fp)) {
        ch = fgetc(vocab_fp);
        if(ch == '\n') {
            vocab_count++;
        }
    }
    rewind(vocab_fp);
    
    // Allocate memory for vocabulary
    tokenizer->vocab = malloc(vocab_count * sizeof(char*));
    tokenizer->vocab_sizes = malloc(vocab_count * sizeof(int));
    tokenizer->vocab_count = vocab_count;
    
    // Read vocabulary
    char line[256];
    int index = 0;
    while(fgets(line, sizeof(line), vocab_fp)) {
        line[strcspn(line, "\n")] = 0; // Remove newline
        tokenizer->vocab[index] = strdup(line);
        tokenizer->vocab_sizes[index] = strlen(line);
        index++;
    }
    fclose(vocab_fp);
    
    // New: Find maximum token length and create token map
    tokenizer->max_token_length = 0;
    for (int i = 0; i < tokenizer->vocab_count; i++) {
        int len = tokenizer->vocab_sizes[i];
        if (len > tokenizer->max_token_length) {
            tokenizer->max_token_length = len;
        }
    }
    
    // Create token map for quick lookup
    tokenizer->token_map_size = tokenizer->vocab_count;
    tokenizer->token_map = malloc(tokenizer->token_map_size * sizeof(char*));
    for (int i = 0; i < tokenizer->vocab_count; i++) {
        tokenizer->token_map[i] = tokenizer->vocab[i];
    }

    return tokenizer;
}

// New helper function to find best token match
int find_best_token(const scnn_Tokenizer* tokenizer, const char* text, int text_len, int* token_len) {
    for (int len = tokenizer->max_token_length; len > 0; len--) {
        if (len > text_len) continue;
        for (int i = 0; i < tokenizer->token_map_size; i++) {
            if (tokenizer->vocab_sizes[i] == len && 
                strncmp(tokenizer->token_map[i], text, len) == 0) {
                *token_len = len;
                return i;
            }
        }
    }
    return -1; // No match found
}

// Modified encode function with automatic merging
int* scnn_tokenizer_encode(scnn_Tokenizer* tokenizer, const char* text, int* output_length) {
    int text_len = strlen(text);
    int* tokens = malloc(text_len * sizeof(int));
    int token_count = 0;
    int pos = 0;
    
    while (pos < text_len) {
        int token_len;
        int token_id = find_best_token(tokenizer, text + pos, text_len - pos, &token_len);
        
        if (token_id == -1) { // Handle unknown characters
            tokens[token_count++] = (unsigned char)text[pos];
            pos++;
        } else {
            tokens[token_count++] = token_id;
            pos += token_len;
        }
    }
    
    *output_length = token_count;
    return tokens;
}

// Modified decode function
char* scnn_tokenizer_decode(scnn_Tokenizer* tokenizer, const int* tokens, int token_count) {
    int total_len = 0;
    for (int i = 0; i < token_count; i++) {
        if (tokens[i] < tokenizer->vocab_count) {
            total_len += tokenizer->vocab_sizes[tokens[i]];
        } else {
            total_len += 1; // For unknown tokens
        }
    }
    
    char* text = malloc(total_len + 1);
    int pos = 0;
    
    for (int i = 0; i < token_count; i++) {
        if (tokens[i] < tokenizer->vocab_count) {
            int len = tokenizer->vocab_sizes[tokens[i]];
            memcpy(text + pos, tokenizer->vocab[tokens[i]], len);
            pos += len;
        } else {
            text[pos++] = (char)(tokens[i] & 0xFF);
        }
    }
    
    text[pos] = '\0';
    return text;
}

// Remove merges-related code from tokenizer_free
void scnn_tokenizer_free(scnn_Tokenizer* tokenizer) {
    if (tokenizer) {
        for (int i = 0; i < tokenizer->vocab_count; i++) {
            free(tokenizer->vocab[i]);
        }
        free(tokenizer->vocab);
        free(tokenizer->vocab_sizes);
        free(tokenizer->token_map);
        free(tokenizer);
    }
}

// Save model to file
void scnn_save_model(scnn_NeuralNetwork* net, const char* filename) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "Error opening file for saving model\n");
        return;
    }
    
    // Save network structure
    fwrite(&net->input_size, sizeof(int), 1, fp);
    fwrite(&net->num_hidden_layers, sizeof(int), 1, fp);
    fwrite(net->hidden_sizes, sizeof(int), net->num_hidden_layers, fp);
    fwrite(&net->output_size, sizeof(int), 1, fp);
    fwrite(&net->trained_epochs, sizeof(int), 1, fp);
    
    // Save weights
    for (int i = 0; i < net->num_hidden_layers + 1; i++) {
        int size = (i == 0) ? net->input_size * net->hidden_sizes[i] :
                   (i == net->num_hidden_layers) ? net->hidden_sizes[i-1] * net->output_size :
                   net->hidden_sizes[i-1] * net->hidden_sizes[i];
        fwrite(net->weights[i], sizeof(double), size, fp);
    }
    
    // Save biases
    for (int i = 0; i < net->num_hidden_layers; i++) {
        fwrite(net->biases[i], sizeof(double), net->hidden_sizes[i], fp);
    }
    fwrite(net->bias_o, sizeof(double), net->output_size, fp);
    
    fclose(fp);
}

// Load model from file
scnn_NeuralNetwork* scnn_load_model(const char* filename) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Error opening file for loading model\n");
        return NULL;
    }
    
    // Read network structure
    int input_size, num_hidden_layers, output_size, trained_epochs;
    fread(&input_size, sizeof(int), 1, fp);
    fread(&num_hidden_layers, sizeof(int), 1, fp);
    
    int* hidden_sizes = malloc(num_hidden_layers * sizeof(int));
    fread(hidden_sizes, sizeof(int), num_hidden_layers, fp);
    
    fread(&output_size, sizeof(int), 1, fp);
    fread(&trained_epochs, sizeof(int), 1, fp);
    
    // Create network
    scnn_NeuralNetwork* net = scnn_create(input_size, num_hidden_layers, hidden_sizes, output_size);
    net->trained_epochs = trained_epochs;
    free(hidden_sizes);
    
    // Load weights
    for (int i = 0; i < num_hidden_layers + 1; i++) {
        int size = (i == 0) ? input_size * net->hidden_sizes[i] :
                   (i == num_hidden_layers) ? net->hidden_sizes[i-1] * output_size :
                   net->hidden_sizes[i-1] * net->hidden_sizes[i];
        fread(net->weights[i], sizeof(double), size, fp);
    }
    
    // Load biases
    for (int i = 0; i < num_hidden_layers; i++) {
        fread(net->biases[i], sizeof(double), net->hidden_sizes[i], fp);
    }
    fread(net->bias_o, sizeof(double), output_size, fp);
    
    fclose(fp);
    return net;
}

// Train network with image
void scnn_train_image(scnn_NeuralNetwork* net, const char* image_path, double* target, double learning_rate) {
    // Load image
    FILE* fp = fopen(image_path, "rb");
    if (!fp) {
        fprintf(stderr, "Error opening image file\n");
        return;
    }
    
    // Get image size
    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    
    // Read image data
    unsigned char* image_data = malloc(size);
    fread(image_data, 1, size, fp);
    fclose(fp);
    
    // Convert image to normalized input
    double* input = malloc(size * sizeof(double));
    for (long i = 0; i < size; i++) {
        input[i] = image_data[i] / 255.0;
    }
    free(image_data);
    
    // Train network
    scnn_train(net, input, target, learning_rate);
    free(input);
}

// Update the rounding function
int scnn_round_output(double value) {
    // Round to nearest integer if value is close to it
    double decimal = value - (int)value;
    if (decimal >= 0.9 && value < 1.0) {
        return (int)value + 1;
    }
    return (int)value;
}

// Add tokenizer size function
int scnn_tokenizer_size(scnn_Tokenizer* tokenizer) {
    return tokenizer->vocab_count;
} 