import numpy as np

# error = (y-(mx+b))**2
def compute_error(m, b, data):
    total_error = 0
    for _, d in enumerate(data):
        X = d[0]
        y = d[1]
        total_error += (y - (m*X + b))**2
    return total_error / len(data)

def train(data, m, b, learning_rate, num_iter):
    for i, d in enumerate(data):
        m,b = gradient_step(m, b, data, learning_rate)
    return m, b

def gradient_step(m, b, data, learning_rate):
    N = len(data)
    m_gradient = 0
    b_gradient = 0
    for i, d in enumerate(data):
        X = d[0]
        y = d[1]
        m_gradient += (2/N) * -X * (y - (m*X+b))
        b_gradient += (2/N) * -(y - (m*X+b))
    m = m - (m_gradient * learning_rate)
    b = b - (b_gradient * learning_rate)
    return m, b

def run():
    # Step 1 - Collect the data
    data = np.genfromtxt('data.csv', delimiter=',')

    # Step 2 - Define hyperparameters
    starting_m = 0
    starting_b = 0
    learning_rate = 1e-4
    num_iter = 1000

    # Step 3 - Train the network
    print('Starting m = {:.4f}\nStarting b = {:.4f}\nError = {:.4f}\n'.format(
        starting_m, starting_b, compute_error(starting_m, starting_b, data)))
    m, b = train(data, starting_m, starting_b, learning_rate, num_iter)
    print('\nFinal m = {:.4f}\nFinal b = {:.4f}\nError = {:4f}\n'.format(
        m, b, compute_error(m, b, data)))

if __name__ == '__main__':
    run()
