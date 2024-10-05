# Parallelizing Support Vector Machines

## 1. Introduction
Implementation revolved around two main algorithms — Parallel IPM (Interior Point
Method) for solving the Quadratic Programming problem, and ICF (Incomplete Cholesky Factorization),
which gives us a low-rank approximation for our Q matrix, which is the matrix Qij = yiyj φ(xi)Tφ(xi).
ICF is useful in our implementation because it reduces the dimension, and thus the computational load,
of our Q matrix, which is in turn used in our Parallel IPM. We are main developers for this code.
The training process is roughly:
1. Set up MPI and initialize our parameters.
2. Calculate the Q matrix and distribute subsets of its rows across nodes.
3. Perform ICF and use the results to perform IPM to solve the optimization problem, yielding the
optimal hyperplane used to separate our training data.
4. Make predictions and compare with the true labels, storing the equation for the optimal hyperplane
to use on testing data.
For our matrix representations and linear algebra operations in our code, we extensively utilized the
Eigen library. We chose the Eigen library for a few reasons. First of all, its optimized performance
for matrix operations, along with the built-in vectorization that facilitates SIMD parallelization, lending
itself to high performance and reduced times. Additionally, it is very flexible in its representation of
matrices, allowing us to dynamically size matrices when we do not know their size in advance, allowing
our code to easily accommodate different dataset sizes. Finally, the library is extensive in the features
it provides, allowing us to handle many complicated linear algebra tasks in one method, rather than
many lines of computation.

Check out the more detailed report [Project Report](https://code.harvard.edu/CS205/team07_2024/blob/439f6f18dff66d0347495df0a10d47b262ae6aa0/milestone/5/report.pdf)


## Setup and Installation

Follow the steps below to set up and run the application:

1. **Clone the Repository**
    - Clone the repository to your local machine.

2. **Generate Data**
    - Run the following command to generate data:
        ```bash
        spack load python@3.11.6%gcc@7.3.1 gcc@13.2.0 openmpi@4.1.6
        python3 parallel_svm/Data/data_generate.py
        ```

3. **Split Data**
    - Run the following command to split the data. Make sure to follow the instructions in the file:
        ```bash
        python3 Utils/data_split.py [X_data.csv] [Y_data.csv] [Percentage-of-data-to-be-train_set]
        ```

4. **Run the Application**
    - Clean and build the application using the following commands:
        ```bash
        make clean
        make all
        ```

5. **Submit a Slurm Job**
    - Edit the `test_job.sh` file to match the data being tested and the type of resource to use.
    - Submit the job using the following command:
        ```bash
        sbatch test_job.sh
        ```


## Experiments
To recreate experiments results with our data run 

    ```bash
        python3 Data/benchmark_plot.py
    ```

