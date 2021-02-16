import numpy as np
import matplotlib.pyplot as plt

PATH_TO_PLOTS = 'plots/'


def visualize_func_in_2d_area(func, area_func_indicator=lambda x, eps: True, title ='',
                              range_x=(-100, 100), range_y=(-100, 100), step=0.2, eps=0.0001):

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
    # plt.show()

    return fig, ax


def draw_points(fig, ax, points,
                title='', style_points='yo',
                use_arrow=False, color_arrow='magenta'):
    points = list(reversed(points))
    points = np.array(points)
    x_points = points[:, 0]
    y_points = points[:, 1]

    if use_arrow:
        for i in range(points.shape[0]-1):
            x_start, y_start = points[i, 0], points[i, 1]
            x_delta, y_delta = points[i+1, 0] - x_start, points[i+1, 1] - y_start

            ax.arrow(x_start, y_start, x_delta, y_delta,
                     length_includes_head=True, width=1, alpha=1, color=color_arrow)
    ax.plot(x_points, y_points, style_points, markersize=10)

    ax.set_title(title)
    plt.savefig(f'{PATH_TO_PLOTS}{title}')
