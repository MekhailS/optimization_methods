import numpy as np
import matplotlib.pyplot as plt


def orthogonality_of_curve_segments(curve_x):
    curve_x = np.asarray(curve_x)
    curve_segments = [curve_x[i] - curve_x[i-1] for i in range(1, len(curve_x))]

    segments_pairs_dot_products = [curve_segments[i] @ curve_segments[i-1]
                                   for i in range(1, len(curve_segments))]
    return segments_pairs_dot_products


PLOTS_PATH = 'plots//'

def plot_contour_with_curves(func, curve_x, curve_name):
    func_xy = lambda x, y: func([x, y])
    func_values = sorted(set([func(x) for x in curve_x]))

    x_points = np.asarray(curve_x)[:, 0]
    y_points = np.asarray(curve_x)[:, 1]

    N_POINTS = 100

    std = np.std(x_points)

    x_grid = np.linspace(np.min(x_points) - std, np.max(x_points) + std, N_POINTS)
    y_grid = np.linspace(np.min(y_points) - std, np.max(y_points) + std, N_POINTS)
    xx, yy = np.meshgrid(x_grid, y_grid)
    z = func_xy(xx, yy)

    fig, ax = plt.subplots(figsize=(10, 10))

    cs = ax.contour(xx, yy, z, func_values,
                    linewidths=3,
                    cmap=plt.get_cmap('winter'))
    ax.clabel(cs)

    """
    for i in range(len(curve_x) - 1):
        x_start, y_start = x_points[i], y_points[i]
        x_delta, y_delta = x_points[i+1] - x_start, y_points[i+1] - y_start

        ax.arrow(x_start, y_start, x_delta, y_delta,
                 length_includes_head=True, alpha=1, color='magenta')
        
    """
    ax.plot(x_points, y_points, 'mo-', markersize=5, label=curve_name)

    ax.legend()
    ax.set_title(f'contour plot of function \n ' +
                    f'and path of gradient descent for {curve_name} method')
    plt.savefig(f'{PLOTS_PATH}//{curve_name}')