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

    training_files = tl.natSort(tl.getFiles(args["training_path"]))
    test_files = tl.natSort(tl.getFiles(args["test_path"]))
    # Set to 1 if you want to see the clusters selection graphically
    graphs = 0

    substr_R = 'Derecha'
    substr_L = 'Izquierda'
    training  = tl.splitFiles(training_files, substr_R, substr_L)
    test = tl.splitFiles(test_files, substr_R, substr_L)
    training_data = tl.getData(training)

    print("\nSearching the best number of clusters for TRAINING...")
    best_clusters_right = kms.usingSilhouette(training_data[0][0], graphs)
    print("\nSearching the best number of clusters for TEST...")
    best_clusters_left = kms.usingSilhouette(training_data[1][0], graphs)

    # for num_clusters in best_clusters_right:
    hmm.generateHMM(best_clusters_right, training_data[0][0])


