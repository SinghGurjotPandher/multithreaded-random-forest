#include "data_loader.h"
#include "tree.h"
#include "threads.h"
#include <unistd.h>
#include <pthread.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

ThreadArgs* create_thread_args(
    Dataset* train_dataset,
    TreeNode** trees,
    int max_depth,
    int min_samples_leaf,
    int start,
    int end,
    unsigned int seed
) {
    ThreadArgs* threadArgs = malloc(sizeof(ThreadArgs));
    threadArgs->train_dataset = train_dataset;
    threadArgs->max_depth = max_depth;
    threadArgs->min_samples_leaf = min_samples_leaf;
    threadArgs->start = start;
    threadArgs->end = end;
    threadArgs->trees = trees;
    threadArgs->seed = seed;

    return threadArgs;
}

int predict_forest(TreeNode** trees, int num_trees, float* row, int num_classes) {
    int* class_counts = calloc(num_classes, sizeof(int));
    int majority_class = -1;
    int max_votes = -1;

    for(int i = 0; i < num_trees; ++i) {
        int prediction = predict_tree(trees[i], row);
        if (prediction >= 0 && prediction < num_classes) {
            class_counts[prediction] += 1;
            if (class_counts[prediction] > max_votes) {
                max_votes = class_counts[prediction];
                majority_class = prediction;
            }
        }
    }
    free(class_counts);
    return majority_class;
}

void free_tree(TreeNode* node) {
    if (node == NULL) return;
    free_tree(node->left);
    free_tree(node->right);
    free(node);
}

void printTree(TreeNode* node, int depth) {
    if (node == NULL) return;

    for (int i = 0; i < depth; ++i) printf("  ");

    if (node->is_leaf) {
        printf("Predict: Class %d\n", node->data.leaf_info.predicted_class);
    } else {
        printf("Feature %d <= %.4f?\n", node->data.split_info.feature_index, node->data.split_info.feature_val);
        printTree(node->left, depth + 1);
        printTree(node->right, depth + 1);
    }
}

void printClasses(Dataset* dataset) {
    printf("Classes in the dataset:\n");
    for (int i = 0; i < dataset->num_classes; ++i) {
        printf("Class %d: %s\n", i, dataset->class_names[i]);
    }
}

int main(int argc, char** argv) {
    if (argc != 7) {
        printf("Incorrect number of arguments. There should be exactly 6 arguments.\n");
        printf("Usage: %s <train_filepath> <class_label_col> <test_filepath> <num_trees> <max_depth> <min_samples_leaf>\n", argv[0]);
        return 1;
    }

    const char* train_filepath = argv[1];
    const int class_label_col = atoi(argv[2]);
    const char* test_filepath = argv[3];
    const int num_trees = atoi(argv[4]);
    const int max_depth = atoi(argv[5]);
    const int min_samples_leaf = atoi(argv[6]);
    const int num_threads = sysconf(_SC_NPROCESSORS_CONF);

    Dataset* train_dataset = load_dataset_from_csv(train_filepath, class_label_col);
    if (train_dataset == NULL) {
        printf("Error: Could not load training dataset from '%s'. Please check the file path.\n", train_filepath);
        return 1;
    }

    
    printf("Training dataset loaded with %zu rows and %zu features.\n", (size_t)train_dataset->num_rows, (size_t)train_dataset->num_features);

    TreeNode** trees = malloc(sizeof(TreeNode*) * num_trees);
    pthread_t* threadIDs = malloc(sizeof(pthread_t) * num_threads);

    int base_trees_per_thread = num_trees / num_threads;
    int remainder = num_trees % num_threads;
    int current_tree_idx = 0;

    for (int i = 0; i < num_threads; ++i) {
        int count = base_trees_per_thread + (i < remainder ? 1 : 0);
        int start = current_tree_idx;
        int end = current_tree_idx + count - 1;

        if (count > 0) {
            pthread_create(
                &threadIDs[i],
                NULL,
                build_decision_trees,
                create_thread_args(
                    train_dataset,
                    trees,
                    max_depth,
                    min_samples_leaf,
                    start,
                    end,
                    (unsigned int)(time(NULL) + i)
                )
            );
        }

        current_tree_idx += count;
    }

    for (int i = 0; i < num_threads; ++i) pthread_join(threadIDs[i], NULL);

    printf("Training completed with %d trees.\n", num_trees);

    // Test
    Dataset* test_dataset = load_dataset_from_csv(test_filepath, class_label_col);
    if (test_dataset == NULL) {
        printf("Error: Could not load testing dataset from '%s'. Please check the file path.\n", test_filepath);
        return 1;
    }

    // Remap test dataset class labels to match training dataset class labels
    int* class_map = malloc(test_dataset->num_classes * sizeof(int));
    for (int i = 0; i < test_dataset->num_classes; ++i) {
        class_map[i] = -1; // Default to -1 if the test class doesn't exist in training
        for (int j = 0; j < train_dataset->num_classes; ++j) {
            if (strcmp(test_dataset->class_names[i], train_dataset->class_names[j]) == 0) {
                class_map[i] = j;
                break;
            }
        }
    }
    for (int i = 0; i < test_dataset->num_rows; ++i) {
        test_dataset->class_labels[i] = class_map[test_dataset->class_labels[i]];
    }

    int correct_predictions = 0;
    for (int i = 0; i < test_dataset->num_rows; ++i) {
        // Use train_dataset->num_classes here to prevent buffer overflows if test_dataset has fewer classes
        int prediction = predict_forest(trees, num_trees, test_dataset->rows[i], train_dataset->num_classes);
        if (prediction != -1 && prediction == test_dataset->class_labels[i])
            correct_predictions += 1;
    }
    printf("Test Accuracy: %.2f%% (%d/%d correct predictions)\n",
           (correct_predictions / (float)test_dataset->num_rows) * 100.0,
           correct_predictions,
           test_dataset->num_rows);

    free(class_map);

    // Cleanup
    for (int i = 0; i < num_trees; ++i) {
        free_tree(trees[i]);
    }

    free(trees);
    free(threadIDs);

    if (train_dataset) {
        for (int i = 0; i < train_dataset->num_rows; ++i) {
            free(train_dataset->rows[i]);
        }
        free(train_dataset->rows);
        if (train_dataset->class_labels) free(train_dataset->class_labels);
        if (train_dataset->class_names) {
            for (int i = 0; i < train_dataset->num_classes; ++i) {
                free(train_dataset->class_names[i]);
            }
            free(train_dataset->class_names);
        }
        free(train_dataset);
    }

    if (test_dataset) {
        for (int i = 0; i < test_dataset->num_rows; ++i) {
            free(test_dataset->rows[i]);
        }
        free(test_dataset->rows);
        if (test_dataset->class_labels) free(test_dataset->class_labels);
        if (test_dataset->class_names) {
            for (int i = 0; i < test_dataset->num_classes; ++i) {
                free(test_dataset->class_names[i]);
            }
            free(test_dataset->class_names);
        }
        free(test_dataset);
    }

    return 0;
}
