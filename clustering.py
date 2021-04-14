from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture as GMM
from itertools import cycle, combinations
import matplotlib as mpl
import matplotlib.pyplot as pl


#read data
iris = load_iris()

#kmean clustering + checking result
kmeans = KMeans(n_clusters=3, max_iter=300)
KMmodel = kmeans.fit(iris.data)
print(pd.crosstab(iris.target,KMmodel.labels_))

#Gaussian mixture with EM + ploting

def make_ellipses(gmm, ax, x, y):
    for n, color in enumerate('rgb'):
        row_idx = np.array([x,y])
        col_idx = np.array([x,y])
        v, w = np.linalg.eigh(gmm.covariances_[n][row_idx[:,None],col_idx])
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v *= 9
        ell = mpl.patches.Ellipse(gmm.means_[n, [x,y]], v[0], v[1],180 + angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)


gmm = GMM(n_components=3,covariance_type='full', max_iter=20)
gmm.fit(iris.data)

predictions = gmm.predict(iris.data)

colors = cycle('rgb')
labels = ["Cluster 1","Cluster 2","Cluster 3"]
targets = range(len(labels))

feature_index=range(len(iris.feature_names))
feature_names=iris.feature_names
combs=combinations(feature_index,2)

f,axarr=pl.subplots(3,2)
axarr_flat=axarr.flat

for comb, axflat in zip(combs,axarr_flat):
    for target, color, label in zip(targets,colors,labels):
        feature_index_x=comb[0]
        feature_index_y=comb[1]
        axflat.scatter(iris.data[predictions==target,feature_index_x],

    iris.data[predictions==target,feature_index_y],c=color,label=label)
        axflat.set_xlabel(feature_names[feature_index_x])
        axflat.set_ylabel(feature_names[feature_index_y])
        make_ellipses(gmm,axflat,feature_index_x,feature_index_y)

pl.tight_layout()
pl.show()


