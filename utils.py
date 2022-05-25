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


def draw_cov_ellipse(mean, sigma, ax=None):
    """
    From center point (mean) and Cov. matrix (sigma) draw error 
    ellipse onto a plot.
    """
    # Get ellipse orientation and axes lengths
    lambdas, vectors = np.linalg.eig(sigma)
    idx = np.argsort(lambdas)[1]

    # Create the CI-95% ellipse
    z = np.arange(0, 2*np.pi, 0.01)
    xpos = 2*math.sqrt(5.991*lambdas[0])*np.cos(z)
    ypos = 2*math.sqrt(5.991*lambdas[1])*np.sin(z)
    # Rotate throught the eigenvectors
    theta = np.arctan(vectors[idx][1]/(vectors[idx][0] + 1e-9))
    new_xpos = mean[0] + xpos*np.cos(theta)+ypos*np.sin(theta)
    new_ypos = mean[1] - xpos*np.sin(theta)+ypos*np.cos(theta)

    # Actually plot the ellipse
    if ax is None:
        ax = plt.gca()
    #plt.plot(xpos, ypos, 'b-')
    ax.plot(new_xpos, new_ypos, 'gray')


    # Testing with 2D problem
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    MEAN = [1, 1]
    SIGMA = np.array([[1, 0.2],
                      [0.2, 1]])
    mmm = [2, 3]
    sss = np.array([[1, 0],
                    [0, 1]])

    # Plot
    plt.figure()
    plt.plot(MEAN[0], MEAN[1], "+", color="green")
    draw_cov_ellipse(MEAN, SIGMA)
    plt.plot(mmm[0], mmm[1], "+", color="green")
    draw_cov_ellipse(mmm, sss)
    plt.plot(MEAN[0]*4, MEAN[1]*4, "+", color="green")
    draw_cov_ellipse(np.array(MEAN)*4, SIGMA*0.5)
    plt.show()
    print("Probability: ", p_multivariate_normal([1, 1], MEAN, SIGMA))
