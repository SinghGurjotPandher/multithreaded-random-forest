#include "data_loader.h"

void strip_newline_chars(char* buffer, ssize_t* length) {
    if (*length > 0 && buffer[*length - 1] == '\n') {
        buffer[*length - 1] = '\0';
        *length -= 1;
    }
    if (*length > 0 && buffer[*length - 1] == '\r') {
        buffer[*length - 1] = '\0';
        *length -= 1;
    }
}

void count_cols(char* line_buffer, int* cols) {
    char* line_copy = strdup(line_buffer);
    char* token = strtok(line_copy, ",");
    while (token) {
        (*cols) += 1;
        token = strtok(NULL, ",");
    }
    free(line_copy);
}

void get_file_dimensions(FILE* file, int* rows, int* cols) {
    *rows = 0, *cols = 0;

    char* line_buffer = NULL;
    ssize_t line_length;
    size_t buffer_size = 0;
    bool first_line = true;

    while ((line_length = getline(&line_buffer, &buffer_size, file)) != -1) {
        strip_newline_chars(line_buffer, &line_length);
        if (line_length == 0) continue;
        if (first_line) {
            count_cols(line_buffer, cols);
            first_line = false;
        }
        *rows += 1;
    }
    free(line_buffer);
    rewind(file);
}

Row* get_row(int num_features, char* line_buffer, int class_label_col) {
    Row* row = malloc(sizeof(Row));
    row->values = malloc(num_features * sizeof(float));
    row->label_name = NULL;

    char* line_copy = strdup(line_buffer);
    char* token = strtok(line_copy, ",");

    int col_index = 0;
    int feature_index = 0;

    while (token) {

        if (col_index == class_label_col) {
            row->label_name = strdup(token);
        } 
        else {
            row->values[feature_index] = atof(token);
            feature_index += 1;
        }
        token = strtok(NULL, ",");
        col_index += 1;
    }

    free(line_copy);
    return row;
}

void update_class_labels(Dataset* dataset, LabelMapEntry** class_names, Row* row_info, int* next_key, int row_index) {
    LabelMapEntry* found = NULL;
    HASH_FIND_STR(*class_names, row_info->label_name, found);
    if (found == NULL) {
        LabelMapEntry* entry = malloc(sizeof(LabelMapEntry));
        entry->name = strdup(row_info->label_name);
        entry->key = *next_key;
        HASH_ADD_STR(*class_names, name, entry);

        dataset->class_labels[row_index] = *next_key;
        dataset->num_classes += 1;
        *next_key += 1;
    } 
    else {
        dataset->class_labels[row_index] = found->key;
    }
}

void update_class_nums_and_class_names(Dataset* dataset, LabelMapEntry* class_names) {
    dataset->class_names = malloc(dataset->num_classes * sizeof(char*));
    LabelMapEntry *current, *temp;
    HASH_ITER(hh, class_names, current, temp) {
        dataset->class_names[current->key] = current->name;
        HASH_DEL(class_names, current);
        free(current);
    }
}

void populate_dataset(Dataset* dataset, int class_label_col, FILE* file) {
    // for training: features, class_labels, num_classes, and class_names
    // for inferencing: features
    char* line_buffer = NULL;
    ssize_t line_length;
    size_t buffer_size = 0;

    int row_index = 0;
    int next_key = 0;

    LabelMapEntry* class_names = NULL;
    
    while((line_length = getline(&line_buffer, &buffer_size, file)) != -1) {
        strip_newline_chars(line_buffer, &line_length);
        if (line_length == 0) continue;

        Row* row_info = get_row(dataset->num_features, line_buffer, class_label_col);
        dataset->rows[row_index] = row_info->values;
        // for training only
        if (class_label_col != -1) update_class_labels(dataset, &class_names, row_info, &next_key, row_index);

        free(row_info->label_name);
        free(row_info);
        row_index += 1;
    }

    if (class_label_col != -1) update_class_nums_and_class_names(dataset, class_names);

    free(line_buffer);
}

Dataset* load_dataset_from_csv(const char* filepath, int class_label_col) {
    FILE* file = fopen(filepath, "r");
    if (file == NULL) return NULL;

    int rows, cols;
    get_file_dimensions(file, &rows, &cols);

    Dataset* dataset = malloc(sizeof(Dataset));
    dataset->num_rows = rows;
    dataset->rows = malloc(dataset->num_rows * sizeof(float*));
    dataset->num_classes = 0;
    if (class_label_col != -1) { // this is training data
        dataset->num_features = cols - 1;
        dataset->class_labels = malloc(dataset->num_rows * sizeof(int));
    } else { // this is inferencing
        dataset->num_features = cols;
        dataset->class_names = NULL;
        dataset->class_labels = NULL;
    }
    populate_dataset(dataset, class_label_col, file);
    fclose(file);
    return dataset;
}
