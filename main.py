import argparse
from sklearn.metrics import confusion_matrix
from myPackage import hmm
from myPackage import tools as tl
from myPackage import kmeansSelection as kms


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-tr", "--training_path", required=True,
                    help="-tr Training path of the samples")
    ap.add_argument("-te", "--test_path", required=True,
                    help="-te Test path of the samples")
    ap.add_argument("-o", "--output_path", required=False,
                    help="-o Output path to store the models and the results")

    args = vars(ap.parse_args())
    # Set to 1 if you want to see the clusters selection graphically
    graphs = 0
    class_names = ['Right', 'Left', 'Other']
    # print(class_names[0])
    training_files = tl.natSort(tl.getFiles(args["training_path"]))
    test_files = tl.natSort(tl.getFiles(args["test_path"]))

    substr_R = 'Derecha'
    substr_L = 'Izquierda'
    training = tl.splitFiles(training_files, substr_R, substr_L)
    test = tl.splitFiles(test_files, substr_R, substr_L)
    # print(len(test[0]), len(test[1]), len(test[2]))
    training_data = tl.getData(training, 0)
    # print(len(training_data[0]))
    test_data = tl.getData(test, 1)
    # print(len(test_data))

    print("\nSearching the best number of clusters for TRAINING (right)...\n")
    num_clusters_right, dic_labels_right, km_model_dic_right = kms.usingSilhouette(training_data[0], graphs)
    print("\nSearching the best number of clusters for TRAINING (left)...\n")
    num_clusters_left, dic_labels_left, km_model_dic_left = kms.usingSilhouette(training_data[1], graphs)

    comb = [comb for comb in tl.product(num_clusters_right, num_clusters_left)]
    r = tl.np.repeat(class_names[0],len(test_data[0]))
    l = tl.np.repeat(class_names[1], len(test_data[1]))
    o = tl.np.repeat(class_names[2], len(test_data[2]))
    lab_test = tl.np.concatenate([r,l,o])
    lab_pred = tl.np.empty_like(lab_test)

    # Train the HMM model using each combination of #clusters
    for n_r, n_l in comb:
        # Train the two HMM using #clusters for right and left
        hmm_model_right = hmm.train(n_r, dic_labels_right)
        hmm_model_left = hmm.train(n_l, dic_labels_left)
        for pos in range(lab_test.size):
            for dir in range(len(test_data)):
                for idx in range(len(test_data[dir])):
                    # print("--- TRAINING AND TESTING HMM for all combinations---")
                    # print("\nUsing '{}-{}' clusters for TEST (right-left)...\n".format(n_r, n_l))
                    data = test_data[dir][idx]
                    # get K-means labels for each test data
                    labels_right_test = kms.kmEvaluation(km_model_dic_right[n_r], data)
                    labels_left_test = kms.kmEvaluation(km_model_dic_left[n_l], data)

                    logProb_r = hmm.test(labels_right_test, hmm_model_right)
                    logProb_l = hmm.test(labels_left_test, hmm_model_left)

                    # print("--- TEST_FILE[{}][{}] IDX[{}] CLUSTERS {}-{}---\n"
                    #       "LogProb right: {}\n"
                    #       "LogProb left: {}\n".format(dir, idx, pos, n_r, n_l, logProb_r, logProb_l))
                    if logProb_r > logProb_l:
                        # print("1---", logProb_r / logProb_l, "\n")
                        if abs(logProb_r-logProb_l) < 50:
                            lab_pred[pos] = 'Other'
                        else:
                            lab_pred[pos] = 'Right'
                    elif logProb_r < logProb_l:
                        # print("2---",logProb_l/ logProb_r,"\n")
                        if abs(logProb_r-logProb_l) < 50:
                            lab_pred[pos] = 'Other'
                        else:
                            lab_pred[pos] = 'Left'
        hits = 0
        for t, p in zip(lab_test, lab_pred):
            if t == p:
                hits += 1
        print("--- CLUSTERS {}-{}---\n"
              "{}\n"
              "{}\n"
              "HITS: {}/{}\n".format(n_r, n_l, lab_test, lab_pred, hits, lab_test.size))
        # Check hits

        # Compute confusion matrix
        # cnf_matrix = confusion_matrix(lab_test, lab_pred)
        #
        # tl.np.set_printoptions(precision=2)
        #
        # # Plot normalized confusion matrix
        # tl.plot_confusion_matrix(cnf_matrix, classes=class_names,
        #                          title='Normalized confusion matrix for {}-{} states (right-left))'.format(n_r, n_l))