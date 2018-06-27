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

    args = vars(ap.parse_args())

    # Set to 1 if you want to see the clusters selection graphically
    graphs = 0
    class_names = ['Right', 'Left', 'Other']

    # Get training, testing files
    training_files = tl.natSort(tl.getFiles(args["training_path"]))
    test_files = tl.natSort(tl.getFiles(args["test_path"]))

    # Get features for each direction (0: training; 1: testing)
    substr_R = 'Derecha'
    substr_L = 'Izquierda'
    training = tl.splitFiles(training_files, substr_R, substr_L)
    test = tl.splitFiles(test_files, substr_R, substr_L)
    training_data = tl.getData(training, 0)
    test_data = tl.getData(test, 1)

    # Get best KMeans configuration based on silhouette metric
    print("\nSearching the best number of clusters for TRAINING (right)...\n")
    num_clusters_right, km_model_dic_right = kms.usingSilhouette(training_data[0], graphs)
    print("\nSearching the best number of clusters for TRAINING (left)...\n")
    num_clusters_left, km_model_dic_left = kms.usingSilhouette(training_data[1], graphs)

    # # Get best KMeans configuration based on silhouette metric
    # print("\nSearching the best number of clusters for TESTING (right)...\n")
    # num_clusters_right, km_model_dic_right = kms.usingSilhouette(test_data[0], graphs)
    # print("\nSearching the best number of clusters for TESTING (left)...\n")
    # num_clusters_left, km_model_dic_left = kms.usingSilhouette(test_data[1], graphs)
    # print("\nSearching the best number of clusters for TESTING (other)...\n")
    # num_clusters_left, km_model_dic_left = kms.usingSilhouette(test_data[2], graphs)


    comb = [comb for comb in tl.product(num_clusters_right, num_clusters_left)]
    r = tl.np.repeat(class_names[0], len(test_data[0]))
    l = tl.np.repeat(class_names[1], len(test_data[1]))
    o = tl.np.repeat(class_names[2], len(test_data[2]))
    lab_test = tl.np.concatenate([r,l,o])
    lab_pred = tl.np.empty_like(lab_test)
    # states = tl.np.arange(5, 26, 5)

    # Train and test the HMM model using each combination of #clusters
    for n_r, n_l in comb:
        print("\nComputing K-means labels for TRAINING using {}-{} clusters (right, left)...".format(n_r, n_l))
        labels_right = tl.np.empty_like(training_data[0])
        labels_left = tl.np.empty_like(training_data[1])
        for i in range(len(training_data[0])):
            labels_right[i] = kms.kmEvaluation(km_model_dic_right[n_r], training_data[0][i])
        for i in range(len(training_data[1])):
            labels_left[i] = kms.kmEvaluation(km_model_dic_left[n_l], training_data[1][i])
        # if tl.np.array_equal(labels_right, labels_left):
        #     print("LABELS ARE EQUALS!!")
        # else:
        #     print("LABELS ARE DIFFERENT!!")

        # Train the two HMM using #clusters for right and left
        hmm_model_right = hmm.train(n_r, labels_right)
        hmm_model_left = hmm.train(n_l, labels_left)
        pos = 0
        hits = 0
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

                if logProb_r < -100 and logProb_l < -100:
                    lab_pred[pos] = 'Other'
                elif logProb_r > logProb_l:
                    lab_pred[pos] = 'Right'
                else:
                    lab_pred[pos] = 'Left'
                pos += 1


        for t, p in zip(lab_test, lab_pred):
            if t == p:
                hits += 1
        print("--- CLUSTERS {}-{} ---\n"
              "{}\n"
              "{}\n"
              "HITS: {}/{} (accuracy: {:.3f}%)\n".format(n_r, n_l, lab_test, lab_pred, hits, lab_test.size, (hits/lab_test.size) * 100))

        # Compute confusion matrix
        cnf_matrix = confusion_matrix(lab_test, lab_pred)

        tl.np.set_printoptions(precision=2)

        # Plot normalized confusion matrix
        tl.plot_confusion_matrix(cnf_matrix, classes=class_names,
                                 title='Confusion matrix for {}-{} states (right-left)'.format(n_r, n_l))
