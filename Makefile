CC = gcc
CFLAGS = -O3 -pthread -Iinclude -Wall -Wextra
SRC = src/main.c src/data_loader.c src/threads.c src/tree.c
OBJ = $(patsubst src/%.c, obj/%.o, $(SRC))
OUT = bin/ensemble_forest

all: $(OUT)

$(OUT): $(OBJ)
	@mkdir -p bin
	$(CC) $(CFLAGS) $(OBJ) -o $(OUT)
	@echo "Build successful! Run with: ./$(OUT)"

obj/%.o: src/%.c
	@mkdir -p obj
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf bin/* obj/*
	@echo "Cleaned bin and obj directories."