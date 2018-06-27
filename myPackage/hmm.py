from hmmlearn import hmm
import numpy as np
from myPackage import kmeansSelection as kms

def train(num_clusters, labels):
    # Make an HMM instance and execute fit
    X = labels
    # lengths = list(map(lambda x: len(x), X))
    X = np.hstack(X)
    X = X.reshape(len(X), 1)

    n_iter = np.arange(10, 101, 10)
    hyper_params = {"n_iter": n_iter,
                    "tol": [1e-2, 1e-3, 1e-4]}

    hmm_model = hmm.MultinomialHMM(n_components= num_clusters, verbose= False)

    ensemble = kms.GridSearchCV(estimator=hmm_model, param_grid=hyper_params, cv=5, n_jobs=-1)
    ensemble.fit(X)

    print("Best params for {} clusters:\n{}".format(num_clusters, ensemble.best_params_))
    hmm_model = ensemble.best_estimator_
    # hmm_model = hmm_model.fit(X, lengths)
    print("Convergence for '{}' clusters? {}\n".format(str(num_clusters), str(hmm_model.monitor_.converged)))

    return hmm_model


def test(labels, hmm_model):
    X = labels.reshape(len(labels), 1)
    score = hmm_model.score(X)
    return score
