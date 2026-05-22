#ifndef TREE
#define TREE

#include "data_loader.h"

typedef struct {
    int feature_index;
    float feature_val;
} Split;

typedef struct s_TreeNode {
    bool is_leaf;
    struct s_TreeNode* left;
    struct s_TreeNode* right;
    union {
        Split split_info;
        struct {
            int predicted_class;
        } leaf_info;
    } data;
} TreeNode;

typedef bool (partitioner)(float* row, float threshold, int column);

bool left_partition(float* row, float threshold, int feature); 
bool right_partition(float* row, float threshold, int feature);

Split* find_best_split(Dataset* dataset, int* valid_rows, int valid_rows_len, bool* pure_split, int* total_left, int* total_right, unsigned int* seed);
void base_case(TreeNode* treenode, int majority_class);
int find_max(int* array, int length);

int determine_majority_class(Dataset* dataset, int* valid_rows, int valid_rows_len);
int* find_valid_rows(Dataset* dataset, int* valid_rows, int valid_rows_len, int n_valid_rows_len, partitioner partitioner, Split* split);

TreeNode* build_decision_tree(Dataset* dataset, int* valid_rows, int valid_rows_len, int max_depth, int min_sample_leaf, unsigned int* seed);

// Prediction
int predict_tree(TreeNode* tree, float* row);

#endif // TREE
