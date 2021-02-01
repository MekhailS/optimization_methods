import numpy as np
import matplotlib.pyplot as plt

PATH_TO_PLOTS = 'plots/'


def visualize_func_in_2d_area(func, area_func_indicator=lambda x, eps: True, title ='',
                              range_x=(-100, 100), range_y=(-100, 100), step=0.1, eps=0.0001):

    def func_on_area_2d(x, y):
        if area_func_indicator([x, y], eps):
            return func([x, y])
        else:
            return np.nan

    func_on_area_2d_vec = np.vectorize(func_on_area_2d)

    X, Y = np.mgrid[range_x[0]:range_x[1]:step, range_y[0]:range_y[1]:step]
    z = func_on_area_2d_vec(X, Y)
    z_masked = np.ma.masked_where(np.isnan(z), z)

    fig, ax = plt.subplots(figsize=(10, 10))

    c = ax.pcolormesh(X, Y, z_masked)

    plt.grid()
    fig.colorbar(c)
    ax.set_title(title)
    plt.savefig(f'{PATH_TO_PLOTS}{title}')
    plt.show()
