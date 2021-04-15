from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture as GMM
from itertools import cycle, combinations
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering,MeanShift, estimate_bandwidth

from sklearn.preprocessing import MinMaxScaler


#read data
iris = load_iris()


# #3 kmean clustering + checking result
kmeans = KMeans(n_clusters=3, max_iter=300)
KMmodel = kmeans.fit(iris.data)
print(pd.crosstab(iris.target,KMmodel.labels_))


#4 Gaussian mixture with EM + ploting
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

f,axarr=plt.subplots(3,2)
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

plt.tight_layout()
plt.show()



#5 clustering with SpectralClustering
df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],columns= iris['feature_names'] + ['target'])

X = df.copy()
X = X.drop('target', axis=1)
# Normalize X

mms = MinMaxScaler()
mms.fit(X)
Xnorm = mms.transform(X)

# Not knowing the number of clusters (3) we try a range such 1,10
# For the ELBOW method check with and without init='k-means++'

Sum_of_squared_distances = []
for k in range(1,10):
    km = KMeans(n_clusters=k, init='k-means++')
    km = km.fit(Xnorm)
    Sum_of_squared_distances.append(km.inertia_)

plt.plot(range(1,10), Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
# plt.show()

# The following bandwidth can be automatically detected using
bandwidth = estimate_bandwidth(Xnorm, quantile=0.2) # Manually set the quantile to get num clusters = 3

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(Xnorm)
labels = ms.labels_

# Compute clustering with SpectralClustering

sc = SpectralClustering(n_clusters = 3)
sc.fit(Xnorm)
labels = ms.labels_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)
print("number of estimated clusters : %d" % n_clusters_)

Clustered = Xnorm.copy()
Clustered = pd.DataFrame(Clustered)
Clustered.loc[:,'Cluster'] = sc.labels_ # append labels to points
#Clustered.sample(5)

frames = [df['target'], Clustered['Cluster']]
result = pd.concat(frames, axis = 1)
#print(result.shape)
#result.sample(5)
for ClusterNum in range(3):

    OneCluster = pd.DataFrame(result[result['Cluster'] == ClusterNum].groupby('target').size())
    OneCluster.columns=['Size']
    
    NewDigit = OneCluster.index[OneCluster['Size'] == OneCluster['Size'].max()].tolist()
    NewDigit[0]

    rowIndex = result.index[result['Cluster'] == ClusterNum]
    result.loc[rowIndex, 'TransLabel'] = NewDigit[0]
    
    print(ClusterNum, NewDigit[0])

# Check performance of classification to 3 clusters

print('Spectral clustering performance')
print('-'*60)

Correct = (df['target'] == result['TransLabel']).sum()
Accuracy = round(Correct/df.shape[0],3)
print('Accuracy ', Accuracy)



