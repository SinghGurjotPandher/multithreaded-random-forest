#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include "stdlib.h"
#include "stdio.h"
#include <stdbool.h>
#include "string.h"
#include "uthash.h"

typedef struct {
    int num_rows;
    int num_features;
    float** rows; // 2D array to hold dataset values
    int* class_labels; // Array to hold class labels as integers

    int num_classes;
    char** class_names; // Array to hold class names
} Dataset;

typedef struct {
    float* values;
    char* label_name;
} Row;

typedef struct {
    char* name;
    int key;
    UT_hash_handle hh;
} LabelMapEntry;

void strip_newline_chars(char* buffer, ssize_t* length);
void count_cols(char* line_buffer, int* cols);
void get_file_dimensions(FILE* file, int* rows, int* cols);

Row* get_row(int num_features, char* line_buffer, int class_label_col);
void update_class_labels(Dataset* dataset, LabelMapEntry** class_names, Row* row_info, int* next_key, int feature);
void update_class_nums_and_class_names(Dataset* dataset, LabelMapEntry* class_names);
void populate_dataset(Dataset* dataset, int class_label_col, FILE* file);

Dataset* load_dataset_from_csv(const char* filepath, int class_label_col);

#endif // DATA_LOADER_H
