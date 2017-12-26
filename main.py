import argparse
from myPackage import tools as tl
from os.path import isfile, join, sep

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-tr", "--training_path", required=True,
                    help="-tr Training path of the samples")
    ap.add_argument("-te", "--test_path", required=True,
                    help="-te Test path of the samples")
    ap.add_argument("-o", "--output_path", required=False,
                    help="-o Output path to store the models and the results")

    args = vars(ap.parse_args())

    right_ends = '*Derecha[0-9*].txt'
    left_ends = '*Izquierda*.txt'
    training_files = tl.natSort(tl.getSamples2(args["training_path"], right_ends))
    test_files = tl.natSort(tl.getSamples(args["test_path"], left_ends))
    print(training_files)

    # import glob
    #
    # for name in glob.glob(sep.join((args["training_path"],'*Derecha*.txt'))):
    #     print(name)