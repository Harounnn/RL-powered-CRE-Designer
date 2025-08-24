import numpy as np

def onehot(seq):
    # Define the possible nucleotides
    nucleotides = ['A', 'C', 'G', 'T']

    # Create a mapping from nucleotide to index
    nucleotide_to_index = {nucleotide: i for i, nucleotide in enumerate(nucleotides)}

    # Initialize a one-hot encoded matrix
    onehot_matrix = np.zeros((len(seq), len(nucleotides)), dtype=int)

    # Fill the one-hot encoded matrix
    for i, nucleotide in enumerate(seq):
        if nucleotide in nucleotide_to_index:
            onehot_matrix[i, nucleotide_to_index[nucleotide]] = 1
    return onehot_matrix

seq = "ATGC"
onehot_encoded = onehot(seq)

print(onehot_encoded)