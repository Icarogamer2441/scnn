CC = gcc
CFLAGS = -Wall -Wextra -O2 -Iinclude
SRC_DIR = src
OBJ_DIR = obj
LIB_DIR = lib
EXAMPLES_DIR = examples

# Library files
SRC = $(wildcard $(SRC_DIR)/*.c)
OBJ = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRC))
LIB = $(LIB_DIR)/libscnn.a

# Targets
all: $(LIB)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(LIB): $(OBJ)
	@mkdir -p $(LIB_DIR)
	ar rcs $@ $^

clean:
	rm -rf $(OBJ_DIR) $(LIB_DIR)
	rm -f $(EXAMPLES)

install: $(LIB)
	@mkdir -p /usr/local/include/scnn
	cp include/scnn.h /usr/local/include/scnn/
	@mkdir -p /usr/local/lib
	cp $(LIB) /usr/local/lib/
	ldconfig

.PHONY: all clean install 