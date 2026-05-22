#include "threads.h"

int* bootstrap_sample(int num_rows, unsigned int* seed) {
    int* sample = malloc(num_rows * sizeof(int));
    for (int i = 0; i < num_rows; ++i) {
        sample[i] = rand_r(seed) % num_rows;
    }
    return sample;
}


void* build_decision_trees(void* args) {
    ThreadArgs* threadArgs = (ThreadArgs*) args;
    Dataset* train_dataset = threadArgs->train_dataset;
    int max_depth = threadArgs->max_depth;
    int min_samples_leaf = threadArgs->min_samples_leaf;
    int start = threadArgs->start;
    int end = threadArgs->end;
    unsigned int seed = threadArgs->seed;

    for (int j = start; j <= end; ++j) {
        int* row_list = bootstrap_sample(train_dataset->num_rows, &seed);

        TreeNode* tree = build_decision_tree(train_dataset, row_list, train_dataset->num_rows, max_depth, min_samples_leaf, &seed);
        threadArgs->trees[j] = tree;

        free(row_list);
    }

    free(threadArgs);
    return NULL;
}
