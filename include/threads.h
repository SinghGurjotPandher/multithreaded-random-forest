#ifndef THREADS
#define THREADS

#include "data_loader.h"
#include "tree.h"
#include "pthread.h"

typedef struct {
    Dataset* train_dataset;
    TreeNode** trees;
    int max_depth;
    unsigned int seed;
    int min_samples_leaf;
    int start;
    int end;
} ThreadArgs;

int* bootstrap_sample(int num_rows, unsigned int* seed);
void* build_decision_trees(void* args);

#endif // THREADS
