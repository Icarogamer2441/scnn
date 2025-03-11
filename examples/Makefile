CC = gcc
CFLAGS = -Wall -Wextra -O2 -I../include
LDFLAGS = -L../lib -lscnn -lm

# Example programs
EXAMPLES = basic_nn tokenizer_example

all: $(EXAMPLES)

%: %.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(EXAMPLES)

.PHONY: all clean 