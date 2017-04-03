# 02. How to Do Linear Regression the Right Way [LIVE]
import numpy as np

def compute_error(b, m, points):
    #initialize error at 0
    total_error = 0
    for _,point in enumerate(points):
        # get the X points, y points
        X = point[0]
        y = point[1]
        # get the diff, square it and add it to the total]
        total_error += (y - (m*X+b)) ** 2
    # get the average of the error
    return total_error / float(len(points))

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iter):
    b = starting_b
    m = starting_m
    # gradient descent
    for i in range(num_iter):
        # update b and m with more accurate b and m by performing
        # this gradient step
        b, m = gradient_step(b, m, np.array(points), learning_rate)
    return b, m

def gradient_step(b_current, m_current, points, learning_rate):
    # starting point for b and m
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    # loop through the points
    for i,point in enumerate(points):
        X = point[0]
        y = point[1]
        # direction with respect to b and m
        # Computing partial derivatives of our error function
        m_gradient += (2/N) * -(y - (m_current*X + b_current))
        b_gradient += (2/N) * -X * (y - (m_current*X + b_current))
    # update our b and m values
    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)
    return new_b, new_m

def run():
    # Step 1: Collect our data
    points = np.genfromtxt('data.csv', delimiter=',')
    # X = Amount of hours studied
    # y = test scores
    
    # Step 2: Define our hyperparameters
    learning_rate = 0.1e-3
    # y = mx + b
    initial_b = 0
    initial_m = 0
    num_iter = 1000

    # Step 3: Train our model
    print('Starting gradient descent at b = {:.4f}, m = {:.4f}, error= {:.3f}'.format(initial_b,
                                                           initial_m,
                                                           compute_error(initial_b, initial_m, points)))
    b, m = gradient_descent_runner(points, initial_b, initial_m,
                                     learning_rate, num_iter)
    print('Ending points at b = {:.4f}, m = {:.4f}, error = {:.3f}'.format(b, m,
                                                         compute_error(b, m, points)))


if __name__ == '__main__':
    run()
