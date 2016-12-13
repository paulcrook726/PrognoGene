import NeuralNet
import argparse
import pickle
import os
import matplotlib.pyplot as plt

NN_VIEW_LENGTH = 500


def seq_to_num(letter):
    if letter == "A":
        return 0.1260
    elif letter == "C":
        return 0.1340
    elif letter == "G":
        return 0.0806
    elif letter == "T":
        return 0.1335
    else:
        return


def subdivide_seq(seq_data, n):
    chunks = []
    for x in range(0, len(seq_data), n):
        chunk = seq_data[x:x + n]
        chunks.append(chunk)
    return chunks


def add_zero(seq_data, n):
    while len(seq_data) < n:
        seq_data.append(0)
    return seq_data


def proc(filename):
    with open(filename) as f:
        lines = [line.strip('\n') for line in f]
    lines_str = ''.join(lines)
    lines_list = lines_str.split('>')
    seq_list = subdivide_seq(add_zero([seq_to_num(gene) for gene in lines_list], NN_VIEW_LENGTH), NN_VIEW_LENGTH)

    return seq_list


def main():
    parser = argparse.ArgumentParser(description='Feed and utilize a neural network for gene exon prediction',
                                     prefix_chars='-/')
    parser.add_argument('seq_path',
                        action='store',
                        type=str,
                        help='Type in the folder containing the desired gene sequence data '
                             '(either full, or relative path')
    parser.add_argument('/f',
                        '-f',
                        dest='f',
                        action='store_true',
                        default=False,
                        help='This option feeds the neural network the data in "seq_path"')
    parser.add_argument('/c',
                        '-c',
                        dest='c',
                        action='store_true',
                        default=False,
                        help='This option designates whether your learning data corresponds to coding, or non-coding'
                             'data.  Default is False (coding data)')
    args = parser.parse_args()
    path = args.seq_path
    if args.c is False:
        c_val = [1]
    else:
        c_val = [0]
    seq_paths = []
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            seq_paths.append(os.path.join(path, name))
    if args.f is False:
        outputs = []
        with open('NN.pickle', 'rb') as handle:
            neural_net = pickle.load(handle)
        for file in seq_paths:
            with open(file) as f:
                proc_seq = proc(file)
                for seq in proc_seq:
                    output = neural_net.predict(seq)
                    outputs.append(output)
            plt.plot(proc_seq, outputs)
            plt.savefig('plot' + str(file) + '.png')
    else:
        if os.path.isfile('NN.pickle') is True:
            with open('NN.pickle', 'rb') as handle:
                neural_net = pickle.load(handle)
        else:
            neural_net = NeuralNet.NeuralNet(NN_VIEW_LENGTH, 1, 70)
        for filename in seq_paths:
            proc_seq = proc(filename)
            for seq in proc_seq:
                neural_net.train([seq, c_val])


main()
