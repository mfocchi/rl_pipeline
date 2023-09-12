
import ReplayBuffer


import matplotlib.pyplot as plt
import matplotlib
import joblib
import os
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import seaborn as sns
from scipy.spatial import ConvexHull


err_treshold = .1


cmap=plt.cm.jet


norm = matplotlib.colors.Normalize(vmin=0, vmax=err_treshold)


tests = [99999]


# fig = plt.figure(figsize=plt.figaspect(1/len(tests)))
fig = plt.figure(figsize=(10,10))

for i,test in enumerate(tests):
    rb = joblib.load(
            f'../runs/test/model_{test}/ReplayBuffer_test.joblib')

    n_episode = rb.mem_size-1
    com0_points = rb.state[:n_episode][:, :3]
    target_points = rb.state[:n_episode][:, 3:]
    reached_points = rb.next_state[:n_episode][:, :3]
    target_error = np.linalg.norm(target_points - reached_points, axis=1)
    target_distance = np.linalg.norm(target_points-com0_points, axis=1)
    perc_error = target_error/target_distance

    # Filter points with perc_error < err_treshold
    idx = perc_error <= err_treshold
    err_filter = perc_error[idx]
    feasible_region = target_points[idx, :]
    n_of_point = len(err_filter)    

    ax = fig.add_subplot(1, 2, i+1, projection='3d')
    ax.margins(x=0,y=0,z=0)
    ax.set_xlim(-.65, .65)
    ax.set_ylim(-.65, .65)
    ax.set_zlim(0, .55)
    ax.view_init(azim=30,elev=45)
    ax.plot_trisurf(feasible_region[:, 0], feasible_region[:, 1], feasible_region[:, 2])
    
    feasible_filtered = ax.scatter(feasible_region[:, 0], feasible_region[:, 1], feasible_region[:, 2],
                                    c=err_filter, alpha=0.7,
                                    cmap=cmap, norm=norm)
    fig.colorbar(feasible_filtered, shrink=0.3)
    # ax.set_xlabel('$x$',fontsize=20)
    # ax.set_ylabel('$y$',fontsize=20)
    # ax.set_zlabel('$z$',fontsize=20)
    # fig.savefig(os.path.join('plots', f'{test}_filtered.svg'), dpi=200)
    plt.tight_layout()
    plt.show()





