from hmmlearn import hmm
import numpy as np
from myPackage import kmeansSelection as kms

def train(num_clusters, labels, n_states):

    # print("\nFitting to HMM and decoding ...")
    # Make an HMM instance and execute fit
    X = labels
    lengths = list(map(lambda x: len(x), X))
    X = np.hstack(X)
    X = X.reshape(len(X), 1)
    # print(X)
    # hyper_params = {"n_init": [5, 10, 15, 20, 25],
    #                 "n_iter": [100, 200, 300, 400, 500, 600, 700, 800],
    #                 "tol": [1e-2, 1e-3, 1e-4]}
    X = np.hstack(X)
    X = X.reshape(len(X), 1)
    n_iter = np.arange(10, 100, 10)
    hyper_params = {"n_iter": n_iter,
                    "tol": [1e-2, 1e-3, 1e-4]}

    trans_prob = np.zeros(shape=(n_states, n_states))
    emission_prob = np.zeros(shape=(n_states, num_clusters))

    for i in range(n_states):
        trans_prob[i] = np.random.dirichlet(np.ones(n_states), size=1)
    for i in range(n_states):
        emission_prob[i] = np.random.dirichlet(np.ones(num_clusters), size=1)

    np.random.seed(42)
    hmm_model = hmm.MultinomialHMM(n_components= n_states)
    hmm_model.start_probability = trans_prob
    hmm_model.emissionprob = emission_prob
    hmm_model.n_features = num_clusters
    ensemble = kms.GridSearchCV(estimator=hmm_model, param_grid=hyper_params, cv=5, n_jobs=-1)
    ensemble.fit(X)
    print("Best params for {} clusters and {} states:\n{}".format(num_clusters, n_states, ensemble.best_params_))
    hmm_model = ensemble.best_estimator_

    # trans_prob = np.zeros(shape=(num_clusters,num_clusters))
    # emission_prob = np.zeros(shape=(num_clusters,num_clusters))
    # for i in range(num_clusters):
    #     trans_prob[i] = np.random.dirichlet(np.ones(num_clusters), size= 1)
    # for i in range(num_clusters):
    #     emission_prob[i] = np.random.dirichlet(np.ones(num_clusters), size= 1)

    # np.random.seed(42)
    # hmm_model = hmm.MultinomialHMM(n_components= num_clusters, verbose= False, n_iter= 70)
    # hmm_model.start_probability = trans_prob
    # hmm_model.emissionprob = emission_prob
    # hmm_model.n_features = num_clusters
    #
    # hmm_model = hmm_model.fit(X, lengths)

    return hmm_model


def test(labels, hmm_model):
    # print(labels)
    X = labels.reshape(len(labels), 1)
    # print(X)
    score = hmm_model.score(X)
    return score
