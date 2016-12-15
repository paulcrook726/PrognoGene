import NeuralNet
import argparse
import pickle
import os
import matplotlib.pyplot as plt

NN_VIEW_LENGTH = 100


def seq_to_num(seq):
    new_seq = []
    for letter in seq:
        if letter == "A":
            new_seq.append(0.1260)
        elif letter == "C":
            new_seq.append(0.1340)
        elif letter == "G":
            new_seq.append(0.0806)
        elif letter == "T":
            new_seq.append(0.1335)
        else:
            pass
    return new_seq


def subdivide_seq(gene, n):
    clean_gene = [bp for bp in gene if bp is not None]
    if len(clean_gene) > n:
        chunks = []
        for x in range(0, len(clean_gene), n):
            chunk = clean_gene[x:x + n]
            chunks.append(chunk)
        chunks.append(add_zero(clean_gene[-(len(clean_gene) % n): len(clean_gene)], n))
        return chunks
    elif len(clean_gene) == n:
        return clean_gene
    elif len(clean_gene) < n:
        if clean_gene is []:
            pass
        else:
            clean_gene = add_zero(clean_gene, n)
            return clean_gene


def add_zero(seq_data, n):
    while len(seq_data) < n:
        seq_data.append(0.)
    return seq_data


def proc(filename):
    with open(filename) as f:
        lines = [line.strip('\n') for line in f]
    lines_str = ''.join(lines)
    lines_list = lines_str.split('>')
    seq_list = [subdivide_seq(seq_to_num(gene), NN_VIEW_LENGTH) for gene in lines_list]

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
        c_val = [0.75]
    else:
        c_val = [0.25]
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
                for chunk in proc_seq:
                    output = neural_net.predict(chunk)
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
            for gene in proc_seq:
                if isinstance(gene[0], list):
                    for chunk in gene:
                        if chunk is None:
                            continue
                        if sum(chunk) == 0.0:
                            continue
                        if len(chunk) != NN_VIEW_LENGTH:
                            chunk = add_zero(chunk, NN_VIEW_LENGTH)
                        neural_net.train([chunk, c_val])
                else:
                    if gene is None:
                        continue
                    if sum(gene) == 0.0:
                        continue
                    if len(gene) is not NN_VIEW_LENGTH:
                        continue
                    neural_net.train([gene, c_val])
        with open('NN.pickel', 'wb') as handle:
            pickle.dump(neural_net, handle)

main()
