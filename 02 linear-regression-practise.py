# 02. How to Do Linear Regression the Right Way [LIVE] | Practise
import numpy as np
from tqdm import tqdm

# Print details and run error computation
def print_details(m, b, dataset):
    error = compute_error(m, b, dataset)
    print('m = {:.4f}, b = {:.4f}, error = {:.3f}'.format(m, b, error))

# Computes the error given m and b values
def compute_error(m, b, dataset):
    total_error = 0
    for _,data in enumerate(dataset):
        X = data[0]
        y = data[1]
        total_error += (y - (m*X+b)) ** 2
    return total_error / len(dataset)

# Gradient Descent Computations
def compute_gradient_descent(m, b, dataset, learning_rate, num_iter):
    for _ in tqdm(range(num_iter)):
        m, b = gradient_step(m, b, dataset, learning_rate)
    return m, b

# Gradient Step using partial derivative
def gradient_step(m, b, dataset, learning_rate):
    m_gradient = 0
    b_gradient = 0
    N = len(dataset)
    for _,data in enumerate(dataset):
        X = data[0]
        y = data[1]
        m_gradient += (2/N) * -X * (y-(m*X+b))
        b_gradient += (2/N) * -(y-(m*X+b))
    new_m = m - (learning_rate * m_gradient)
    new_b = b - (learning_rate * b_gradient)
    return new_m, new_b

# Start the program
def run():
    # Step 1: Collect the dataset
    dataset = np.loadtxt('linear_regression_live-master/data.csv',
                         delimiter=',')
    # Step 2: Define hpyerparameters
    learning_rate = 1e-4
    initial_m = 0
    initial_b = 0
    num_iter = 1000
    ## Print the initial details
    print_details(initial_m, initial_b, dataset)
    # Step 3: Compute Gradient Descent
    final_m, final_b = compute_gradient_descent(initial_m,
                                                initial_b,
                                                dataset,
                                                learning_rate,
                                                num_iter)
    ## Print the final details
    print_details(final_m, final_b, dataset)    

if __name__ == '__main__':
    run()
