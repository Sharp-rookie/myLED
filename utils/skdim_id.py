import skdim
import numpy as np


if __name__ == '__main__':

    data = np.zeros((1000, 10))
    data[:,:5] = skdim.datasets.hyperBall(n=1000, d=5, radius=1, random_state=0)

    mle = skdim.id.MLE().fit(data, n_neighbors=100)
    mind_ml = skdim.id.MiND_ML().fit_pw(data, n_neighbors=100)
    mada = skdim.id.MADA().fit(data, n_neighbors=100)
    pca = skdim.id.lPCA().fit_pw(data, n_neighbors=100)

    print(mle.dimension_, mind_ml.dimension_, mada.dimension_, pca.dimension_)
    print(np.mean(mle.dimension_pw_), np.mean(mind_ml.dimension_pw_), np.mean(mada.dimension_pw_), np.mean(pca.dimension_pw_))