from hmmlearn import hmm
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

def train(num_clusters, labels):
    print("\nFitting to HMM and decoding ...")
    # Make an HMM instance and execute fit


    X = labels[num_clusters]
    print(X)
    X = np.array([X]).T

    hmm_model = hmm.MultinomialHMM(n_components= num_clusters, verbose= False, n_iter= 100, tol= 1e-5)
    hmm_model = hmm_model.fit(X)

    return hmm_model


def test(labels, hmm_model):
    states = ["0", "1", "2", "3", "4"]
    # states = states[:num_clusters]
    logprob, sequence = hmm_model.decode(labels, algorithm="viterbi")
    # print ("Bob says:" , ", ".join(map(lambda x: observations[x], bob_says.T[0])))
    print("Sequence predicted:", ", ".join(map(lambda x: states[x], sequence)))
    print("LogProb: ", logprob)
    # np.random.dirichlet(np.ones(num_estados), size=1)
