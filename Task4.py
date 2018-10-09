import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

X = np.array([[1.0, 1.0],[1.5, 2.0],[3.0, 4.0],[5.0, 7.0],[3.5, 5.0],[4.5, 5.0]])

kmeans_4 = KMeans(max_iter=100, n_clusters=4, random_state=0)#creates 4 clusters
kmeans = KMeans(max_iter=100, n_clusters=2, random_state=0)# creates two clusters
kmeans_4.fit(X) #fitting data for 4 clusters
kmeans.fit(X) #fitting data for two clusters
centers = kmeans_4.cluster_centers_
kcenters=kmeans.cluster_centers_
labels = kmeans.labels_
labels_4 = kmeans_4.labels_
print("K=4 Cluster points: \n",centers)
print("K=2 Cluster Points: \n",kcenters)


# colors = ["g.","r.","c.","y."]
#
# for i in range(len(X)):
#     plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize = 10)
#
#
# plt.scatter(kcenters[:, 0],kcenters[:, 1], marker = "x", s=150, linewidths = 5, zorder = 10)
# plt.title('K=2 Cluster points')
# plt.legend(kcenters)
# plt.show()
#
# for i in range(len(X)):
#     plt.plot(X[i][0], X[i][1], colors[labels_4[i]], markersize = 10)
#
# plt.scatter(centers[:, 0],centers[:, 1], marker = "x", s=150, linewidths = 5, zorder = 10)
# plt.title('K=4 Cluster points')
# plt.legend(centers)
# plt.show()


"""
In Kmeans clustering , data is reassigned/redistributed to the nearest cluster based on the centroid.
Accuracy is based on how far or how close the data point is from the centroid. The more the number of 
clusters (Increasing K) the more the number of centroids in that essence leads to shorter distance between datapoint
and centroid hence more accuracy. For example in this instance with a K=1, we get one cluster point/centroid and the 
distance between data point and the centroid is much more while using a k=4 it resulted in a smaller distance between 
the points and the centroid hence more accurate.


"""