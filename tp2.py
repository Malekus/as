import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import PCA
import warnings
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import adjusted_rand_score

# Retroune le barycentre d'une classe
def getBarycenter(x, y, k):
    if (y == k).sum():
        return x[np.where(np.column_stack((x,y))[:,4] == k)].mean(0)
    else :
        return None

# Renvoie un tableau d'initialisation de taille K
def initK(x, k, typeInit=None):
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
        return x[np.random.choice(len(x), k, replace=False)]


# Algorithme du k mean
def k_mean(x, k, typeK='Random', tab=[]):
    # Choix de l'initialisation Random, Plus ou Moins
	d = initK(x, k, typeK) if tab == [] else tab
	first = d
	nb = []
    # Tant qu'il n'y a pas une stabilisaiton des barycentres
	while(1):
        # Calcul des distances entres les barycentres et les donnees
		p = euclidean_distances(x, d)
		label = np.argmin(p, axis=1)
		d = np.array([getBarycenter(x, label, el) for el in range(k)])
		nb.append(d.sum())
                # Barycentre n == barycentre n-1 donc stabilisation des barycentres
		if len(nb) > 1 and nb[-2] == nb[-1] and nb[-3] == nb[-2]:
			break
    # retourne les labels, la sommes des barycentres a chaque iteration et les barycentres trouves
	return np.array(label), np.array(nb), first

# Algorithme du k medoid
def k_medoid(x, k, typeK='Random'):
    # Choix de l'initialisation Random, Plus ou Moins
	d = initK(x, k, typeK)
	first = d
	nb = []
	# Tant qu'il n'y a pas une stabilisaiton des barycentres
	while(1):
		p = euclidean_distances(x, d)
		label = np.argmin(p, axis=1)
		d = np.array([getBarycenter(x, label, el) for el in range(k)])
		nb.append(d.sum())
		# Barycentre n == barycentre n-1 donc stabilisation des barycentres
		if len(nb) > 1 and nb[-2] == nb[-1] and nb[-3] == nb[-2]:
			break
	return np.array(label), np.array(nb), first

# Renvoie un tableau d'indice de clustering
def makeScoring(x, y):
    warnings.simplefilter("ignore")
    a = davies_bouldin_score(x, y)
    b = silhouette_score(x, y)
    c = adjusted_mutual_info_score(y, y)
    d = adjusted_rand_score(y, y)
    # renvoie les indices suivants
    return {"Davies-Bouldin": a, "Silhouette": b, "Adjusted Mutual Information": c, "Rand Index": d}

x = datasets.load_iris().data
y = datasets.load_iris().target
pca = PCA(n_components=2)
dataPCA = pca.fit_transform(x)

kx1, ky1, fp1 = k_mean(x, 3, 'Random')

kx2, ky2, fp2 = k_mean(x, 3, 'Plus')

kx3, ky3, fp3 = k_mean(x, 3, 'Moins')

"""
plt.figure("Comparaison Kmean")
plt.subplot(3,1,1)
plt.title("Init Random")
plt.scatter(dataPCA[:,0], dataPCA[:, 1], c=kx1, alpha=0.8)
plt.scatter(pca.fit_transform(fp1)[:,0], pca.fit_transform(fp1)[:,1], c='red', alpha=0.8)
plt.subplot(3,1,2)
plt.title("Init Kmean ++")
plt.scatter(dataPCA[:,0], dataPCA[:, 1], c=kx2, alpha=0.8)
plt.scatter(pca.fit_transform(fp2)[:,0], pca.fit_transform(fp2)[:,1], c='red', alpha=0.8)
plt.subplot(3,1,3)
plt.title("Init Kmean --")
plt.scatter(dataPCA[:,0], dataPCA[:, 1], c=kx3, alpha=0.8)
plt.scatter(pca.fit_transform(fp3)[:,0], pca.fit_transform(fp3)[:,1], c='red', alpha=0.8)
plt.show()
"""