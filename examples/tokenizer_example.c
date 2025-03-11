#include <stdio.h>
#include <stdlib.h>
#include "../include/scnn.h"

int main() {
    // Create tokenizer with default files
    const char* vocab_path = "vocab.txt";
    const char* merges_path = "merges.txt";
    
    printf("Using vocabulary file: %s\n", vocab_path);
    printf("Using merges file: %s\n", merges_path);
    
    scnn_Tokenizer* tokenizer = scnn_tokenizer_create(vocab_path, merges_path);
    if (!tokenizer) {
        fprintf(stderr, "Failed to create tokenizer\n");
        return 1;
    }
    
    // Example text
    const char* text = "Hello, world! This is a test.";
    printf("Original text: %s\n", text);
    
    // Encode text
    int token_count;
    int* tokens = scnn_tokenizer_encode(tokenizer, text, &token_count);
    
    printf("Encoded tokens: ");
    for (int i = 0; i < token_count; i++) {
        printf("%d ", tokens[i]);
    }
    printf("\n");
    
    // Decode tokens
    char* decoded_text = scnn_tokenizer_decode(tokenizer, tokens, token_count);
    printf("Decoded text: %s\n", decoded_text);
    
    // Clean up
    free(tokens);
    free(decoded_text);
    scnn_tokenizer_free(tokenizer);
    
    return 0;
} 