import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import PCA


def getBarycenter(x, y, k):
    if (y == k).sum():
            return x[np.where(np.column_stack((x,y))[:,4] == k)].mean(0)
    else :
            return None
        
def k_mean(x, k):
	d = x[np.random.choice(x.shape[0], k, replace=False)]
	first = d
	nb = []
	while(1):
		p = euclidean_distances(x, d)
		label = np.argmin(p, axis=1)
		d = np.array([getBarycenter(x, label, el) for el in range(k)])
		nb.append(d.sum())
		if len(nb) > 1 and nb[-2] == nb[-1] and nb[-3] == nb[-2]:
			break
	return np.array(label), np.array(nb), first

def k_mean_plus(x, k):
	d = x[np.random.choice(x.shape[0], 1, replace=False)]
	for i in range(k - 1):
		if i == 0:
			d = np.append(d, [x[np.argmax(euclidean_distances(x,d))]], axis=0)
		else:
			d = np.append(d, [x[np.argmax(euclidean_distances(x,[d.mean(0)]))]], axis=0)
	first = d
	nb = []
	while(1):
		p = euclidean_distances(x, d)
		label = np.argmin(p, axis=1)
		d = np.array([getBarycenter(x, label, el) for el in range(k)])
		nb.append(d.sum())
		if len(nb) > 1 and nb[-2] == nb[-1] and nb[-3] == nb[-2]:
			break
	return np.array(label), np.array(nb), first

x = datasets.load_iris().data
y = datasets.load_iris().target
pca = PCA(n_components=2)
dataPCA = pca.fit_transform(x)
kx, ky, fp = k_mean(x, 3)
kxp, kyp, fpp = k_mean_plus(x, 3)


# Renvoie un tableau d'initialisation de taille K
def initK(x, k, typeInit='Random'):
    # Si on utilise l'initialisation ++
    if typeInit is 'Plus':
        d = x[np.random.choice(len(x), 1, replace=False)]
        for i in range(k - 1):
            if i == 0:
                d = np.append(d, [x[np.argmax(euclidean_distances(x,d))]], axis=0)
            else:
                d = np.append(d, [x[np.argmax(euclidean_distances(x,[d.mean(0)]))]], axis=0)
        return d
    # Si on utilise l'initialisation --
    if typeInit is 'Moins':
        d = x[np.random.choice(len(x), 1, replace=False)]
        for i in range(k - 1):
            if i == 0:
                d = np.append(d, [x[np.argmin(euclidean_distances(np.delete(x, d, axis=0),d))]], axis=0)
            else:
                d = np.append(d, [x[np.argmin(euclidean_distances(np.delete(x, d, axis=0),[d.mean(0)]))]], axis=0)
        return d
    # Sinon initilisation random de taille k
    else:
        return x[np.random.choice(len(x), 5, replace=False)]
        
"""
plt.figure("Comparaison Kmean")
plt.subplot(3,1,1)
plt.title("No Kmean")
plt.scatter(dataPCA[:,0], dataPCA[:, 1], c=y, alpha=0.8)
plt.subplot(3,1,2)
plt.title("Kmean")
plt.scatter(dataPCA[:,0], dataPCA[:, 1], c=kx, alpha=0.8)
plt.scatter(pca.fit_transform(fp)[0,0], pca.fit_transform(fp)[0,1], c='black', alpha=0.8, marker="X")
plt.scatter(pca.fit_transform(fp)[1:,0], pca.fit_transform(fp)[1:,1], c='red', alpha=0.8)
plt.subplot(3,1,3)
plt.title("Kmean++")
plt.scatter(dataPCA[:,0], dataPCA[:, 1], c=kxp, alpha=0.8)
plt.scatter(pca.fit_transform(fpp)[0,0], pca.fit_transform(fpp)[0,1], c='black', alpha=0.8, marker="X")
plt.scatter(pca.fit_transform(fpp)[1:,0], pca.fit_transform(fpp)[1:,1], c='red', alpha=0.8)
plt.show()
"""

initK(x, 5, 'Plus')
