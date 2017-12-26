from hmmlearn import hmm
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

def generateHMM(num_clusters, data):
    print("Fitting to HMM and decoding ...")
    # Make an HMM instance and execute fit
    hmm_model = hmm.GaussianHMM(n_components= num_clusters[1], covariance_type= 'full', verbose= True, n_iter= 100, tol= 1e-4)
    X = data
    hmm_model.fit(X)
    print("Convergence Monitor: \n{}\n".format(hmm_model.monitor_))
    print("Converged: {}".format(hmm_model.monitor_.converged))
    # Predict the optimal sequence of internal hidden state
    hidden_states = hmm_model.predict(X)

    # hmm.ConvergenceMonitor(history= [...], iter= 12, n_iter= 100, tol= 0.01, verbose= True)
    print("DONE!")

    # print(hmm_model.n_components)
    print("\nTransition matrix\n{}".format(hmm_model.transmat_))

    print("\nMeans and vars of each hidden state")
    # for i in range(hmm_model.n_components):
    #     print("{0}th hidden state\n"
    #           "mean = {}\n"
    #           "var = {}\n".format(i, hmm_model.means_[i], np.diag(hmm_model.covars_[i])))
    for i in range(hmm_model.n_components):
        print("{0}th hidden state".format(i))
        print("mean = ", hmm_model.means_[i])
        print("var = ", np.diag(hmm_model.covars_[i]))
        print()
