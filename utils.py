import numpy as np
import math


def p_multivariate_normal(x, mean, sigma):
    """
    Computes the probability of an arbitrary array (state) given that
    the underlying random variable is normally distributed.

    x : r.v realization                              (n, 1)
    mean : mean value for the r.v                    (n, 1)
    sigma : along with 'mean' parametrizes the r.v   (n, n)
    """
    x = np.array(x)
    mean = np.array(mean)
    exponent = -0.5 * np.transpose(x-mean) @ np.linalg.inv(sigma) @ (x-mean)
    return math.exp(exponent) / math.sqrt(np.linalg.det(2*math.pi*sigma))


def draw_cov_ellipse(mean, sigma):
    # Get ellipse orientation and axes lengths
    lambdas, vectors = np.linalg.eig(sigma)
    idx = np.argsort(lambdas)[1]

    # Create the ellipse
    z = np.arange(0, 2*np.pi, 0.01)
    xpos = lambdas[0]*np.cos(z)
    ypos = lambdas[1]*np.sin(z)
    # Rotate throught the eigenvectors
    theta = np.arctan(vectors[idx][1]/(vectors[idx][0] + 1e-9))
    new_xpos = mean[0] + xpos*np.cos(theta)+ypos*np.sin(theta)
    new_ypos = mean[1] - xpos*np.sin(theta)+ypos*np.cos(theta)

    # Actually plot the ellipse
    #plt.plot(xpos, ypos, 'b-')
    plt.plot(new_xpos, new_ypos)#, 'r-')

    # Testing with 2D problem
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    MEAN = [1, 1]
    SIGMA = np.array([[1, 0.2],
                      [0.2, 1]])

    draw_cov_ellipse(MEAN, SIGMA)
    draw_cov_ellipse([2, 3], SIGMA*-2.5)
    plt.show()
    print("Probability: ", p_multivariate_normal([1, 1], MEAN, SIGMA))
