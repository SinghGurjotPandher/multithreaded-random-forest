# Multithreaded Random Forest Engine in C

A high-performance, multithreaded Random Forest classification engine built from scratch in C. This project implements the core CART (Classification and Regression Trees) algorithm, utilizing POSIX threads (`pthreads`) to dynamically distribute tree construction across available CPU cores.

## Core Architecture & Optimizations
*   **Parallel Processing:** Uses `pthreads` to divide the workload of tree creation, scaling dynamically with logical core count.
*   **Algorithmic Efficiency:** Implements true Random Forest feature sub-sampling ($\sqrt{F}$) to decorrelate trees and prevent overfitting.
*   **O(N log N) Split Search:** Accelerates the Gini impurity calculation by sorting feature arrays and utilizing linear scans, reducing node-split calculations from billions of redundant iterations to optimal thresholds.
*   **Strict Memory Safety:** Manages dynamically sized rows, features, and custom Hash Maps (`uthash`) with explicit alloc/free lifecycles and zero heap leaks.

## Performance Metrics
Tested on the UCI Forest Covertype Dataset (581,012 rows, 54 features):
*   **Speed:** Built an ensemble of 100 decision trees (max depth 20) in **~43 seconds**, achieving a **5.1x parallel speedup** over sequential execution (>500% CPU utilization).
*   **Accuracy:** Achieved a training accuracy of **80.6%**.

## Compilation
This project includes a `Makefile` utilizing `-O3` optimization flags for maximum C compiler loop vectorization. 
```bash
make clean
make
```

## Usage
Run the generated binary with the following 6 arguments:
```bash
./bin/ensemble_forest <train_filepath> <class_label_col> <test_filepath> <num_trees> <max_depth> <min_samples_leaf>
```

### Arguments:
*   `train_filepath`: Path to the CSV file used for training (no headers).
*   `class_label_col`: The 0-indexed column integer containing the target classes.
*   `test_filepath`: Path to the CSV file used for testing accuracy.
*   `num_trees`: Number of estimators to build in the forest (e.g. `100`).
*   `max_depth`: Maximum depth of a tree before forcing a leaf node.
*   `min_samples_leaf`: Minimum number of samples required to allow a split.

*Note: The CSV parser currently uses `strtok` and does not support missing/empty values (e.g. adjacent commas like `,,`). Ensure your dataset is fully populated before training.*

### Example Command
```bash
time ./bin/ensemble_forest data/covtype.data 54 data/covtype.data 100 20 5
```
*The example command above uses data/covtype.data for both the <train_filepath> and <test_filepath> arguments. Because it tests on the exact same data it trained on, the resulting metric will reflect the model's training accuracy. To evaluate true out-of-sample performance, ensure you pass a distinct, separate CSV file for testing.*
