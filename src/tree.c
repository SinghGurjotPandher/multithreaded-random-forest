#include "tree.h"

typedef struct {
    float value;
    int class_label;
} FeatureValue;

int compare_feature_values(const void* a, const void* b) {
    float valA = ((FeatureValue*)a)->value;
    float valB = ((FeatureValue*)b)->value;
    if (valA < valB) return -1;
    if (valA > valB) return 1;
    return 0;
}

bool left_partition(float* row, float threshold, int feature) { 
    return (row[feature] <= threshold); 
}

bool right_partition(float* row, float threshold, int feature) {
    return (row[feature] > threshold);
}


Split* find_best_split(Dataset* dataset, int* valid_rows, int valid_rows_len, bool* pure_split, int* total_left, int* total_right, unsigned int* seed) {
    Split* result = malloc(sizeof(Split));
    float best_split_score = 2.0f; // Initialized high; Gini is always <= 1.0

    result->feature_index = 0;
    result->feature_val = 0;
    *total_left = 0;
    *total_right = 0;

    bool split_found = false;
    int num_classes = dataset->num_classes;

    // Precompute total class distributions for the current node
    int* total_classes_count = calloc(num_classes, sizeof(int));
    for (int i = 0; i < valid_rows_len; ++i) {
        total_classes_count[dataset->class_labels[valid_rows[i]]]++;
    }

    FeatureValue* feature_values = malloc(valid_rows_len * sizeof(FeatureValue));
    int* left_classes_count = malloc(num_classes * sizeof(int));
    int* right_classes_count = malloc(num_classes * sizeof(int));

    // Calculate sqrt(num_features) for Random Forest sub-sampling
    int max_features = 1;
    while (max_features * max_features <= dataset->num_features) max_features++;
    max_features--;

    // Create an array of feature indices and shuffle it
    int* feature_indices = malloc(dataset->num_features * sizeof(int));
    for (int i = 0; i < dataset->num_features; ++i) feature_indices[i] = i;
    for (int i = dataset->num_features - 1; i > 0; i--) {
        int j = rand_r(seed) % (i + 1);
        int temp = feature_indices[i];
        feature_indices[i] = feature_indices[j];
        feature_indices[j] = temp;
    }

    for (int k = 0; k < max_features; ++k) {
        int feature = feature_indices[k];
        // Populate and sort feature values
        for (int i = 0; i < valid_rows_len; ++i) {
            feature_values[i].value = dataset->rows[valid_rows[i]][feature];
            feature_values[i].class_label = dataset->class_labels[valid_rows[i]];
        }
        qsort(feature_values, valid_rows_len, sizeof(FeatureValue), compare_feature_values);

        // Reset left/right trackers for this new feature
        for (int c = 0; c < num_classes; ++c) {
            left_classes_count[c] = 0;
            right_classes_count[c] = total_classes_count[c];
        }

        int left_count = 0;
        int right_count = valid_rows_len;

        // Linearly scan the sorted values
        for (int i = 0; i < valid_rows_len - 1; ++i) {
            int c_label = feature_values[i].class_label;
            left_classes_count[c_label]++;
            right_classes_count[c_label]--;
            left_count++;
            right_count--;

            // Only evaluate the split if the values are distinctly different
            if (feature_values[i].value < feature_values[i + 1].value) {
                float left_gini = 1.0f, sum_prob_left = 0.0f;
                for (int c = 0; c < num_classes; ++c) { float p = (float)left_classes_count[c] / left_count; sum_prob_left += p * p; }
                left_gini -= sum_prob_left;

                float right_gini = 1.0f, sum_prob_right = 0.0f;
                for (int c = 0; c < num_classes; ++c) { float p = (float)right_classes_count[c] / right_count; sum_prob_right += p * p; }
                right_gini -= sum_prob_right;

                float gini_split_score = (((float)left_count / valid_rows_len) * left_gini) + (((float)right_count / valid_rows_len) * right_gini);

                if (gini_split_score < best_split_score) {
                    best_split_score = gini_split_score;
                    result->feature_index = feature;
                    result->feature_val = feature_values[i].value;
                    *total_left = left_count;
                    *total_right = right_count;
                    split_found = true;
                }
            }
        }
    }

    free(total_classes_count);
    free(feature_values);
    free(left_classes_count);
    free(right_classes_count);
    free(feature_indices);

    if (best_split_score == 0.0f || !split_found) *pure_split = true;

    return result;
}

void base_case(TreeNode* treenode, int majority_class) {
    treenode->is_leaf = true;
    treenode->left = NULL;
    treenode->right = NULL;
    treenode->data.leaf_info.predicted_class = majority_class;
}

int find_max(int* array, int length) {
    int max = 0;
    for (int i = 1; i < length; ++i) {
        if (array[i] > array[max]) max = i;
    }
    return max;
}

int determine_majority_class(Dataset* dataset, int* valid_rows, int valid_rows_len) {
    // out of the valid rows, find the class with the most count
    int* classes_count = calloc(dataset->num_classes, sizeof(int));
    for (int i = 0; i < valid_rows_len; ++i) 
        classes_count[dataset->class_labels[valid_rows[i]]] += 1;
    int majority_class = find_max(classes_count, dataset->num_classes);
    free(classes_count);
    return majority_class;
}

int* find_valid_rows(Dataset* dataset, int* valid_rows, int valid_rows_len, int n_valid_rows_len, partitioner partitioner, Split* split) {
    int* n_valid_rows = malloc(n_valid_rows_len * sizeof(int));
    int j = 0;
    for(int i = 0; i < valid_rows_len; ++i) {
        if (partitioner(dataset->rows[valid_rows[i]], split->feature_val, split->feature_index)) {
            n_valid_rows[j] = valid_rows[i];
            j += 1;
        }
    }
    return n_valid_rows;
}

TreeNode* build_decision_tree(Dataset* dataset, int* valid_rows, int valid_rows_len, int max_depth, int min_sample_leaf, unsigned int* seed) {
    TreeNode* treenode = malloc(sizeof(TreeNode));
    if (max_depth == 0 || valid_rows_len <= min_sample_leaf) {
        base_case(treenode, determine_majority_class(dataset, valid_rows, valid_rows_len));
        return treenode;
    }

    bool pure_split = false;
    int total_left_rows, total_right_rows;
    Split* split = find_best_split(dataset, valid_rows, valid_rows_len, &pure_split, &total_left_rows, &total_right_rows, seed);

    if (pure_split == true) {
        base_case(treenode, determine_majority_class(dataset, valid_rows, valid_rows_len));
        free(split);
        return treenode;
    }
     
    int* left_valid_rows = find_valid_rows(dataset, valid_rows, valid_rows_len, total_left_rows, left_partition, split);
    int* right_valid_rows = find_valid_rows(dataset, valid_rows, valid_rows_len, total_right_rows, right_partition, split);

    treenode->is_leaf = false;
    treenode->data.split_info.feature_index = split->feature_index;
    treenode->data.split_info.feature_val = split->feature_val;
    free(split);

    treenode->left = build_decision_tree(dataset, left_valid_rows, total_left_rows, max_depth-1, min_sample_leaf, seed);
    treenode->right = build_decision_tree(dataset, right_valid_rows, total_right_rows, max_depth-1, min_sample_leaf, seed);

    free(left_valid_rows);
    free(right_valid_rows);

    return treenode;
}

int predict_tree(TreeNode* tree, float* row) {
    if (tree->is_leaf)
        return tree->data.leaf_info.predicted_class;

    if (row[tree->data.split_info.feature_index] <= tree->data.split_info.feature_val)
        return predict_tree(tree->left, row);
    else
        return predict_tree(tree->right, row);   
}
