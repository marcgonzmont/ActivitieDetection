import argparse
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

    training_files = tl.natSort(tl.getFiles(args["training_path"]))
    test_files = tl.natSort(tl.getFiles(args["test_path"]))

    substr_R = 'Derecha'
    substr_L = 'Izquierda'
    training = tl.splitFiles(training_files, substr_R, substr_L)
    test = tl.splitFiles(test_files, substr_R, substr_L)
    # print(len(test[0]), len(test[1]), len(test[2]))
    training_data = tl.getData(training)
    # print(len(training_data[0]))
    test_data = tl.getData(test)

    print("\nSearching the best number of clusters for TRAINING (right)...\n")
    num_clusters_right, dic_labels_right, km_model_dic_right = kms.usingSilhouette(training_data[0], graphs)
    print("\nSearching the best number of clusters for TRAINING (left)...\n")
    num_clusters_left, dic_labels_left, km_model_dic_left = kms.usingSilhouette(training_data[1], graphs)
    #
    # for i in num_clusters_right:
    #     print("\nUsing '{}' clusters for TEST (right)...\n".format(i))
    #     labels_right_test = kms.kmEvaluation(km_model_dic_right[i], test_data[0])
    #
    # for i in num_clusters_left:
    #     print("\nUsing '{}' clusters for TEST (left)...\n".format(i))
    #     labels_left_test = kms.kmEvaluation(km_model_dic_left[i], test_data[1])

        # for num_clusters in best_clusters_right:
        # hmm_model_right = hmm.train(num_clusters_right[0], dic_labels_right)
        # hmm.test(dic_labels_right[num_clusters_right[0]], hmm_model_right)
