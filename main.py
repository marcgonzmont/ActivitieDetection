import argparse
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

    substr_R = 'Derecha'
    substr_L = 'Izquierda'
    training  = tl.splitFiles(training_files, substr_R, substr_L)
    test = tl.splitFiles(test_files, substr_R, substr_L)
    training_data = tl.getData(training)
    # print(len(training_data[0][0]))
    # print(training_data[0][0][:,1])
    kms.selectKmeans(training_data[0][0])