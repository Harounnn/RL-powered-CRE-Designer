# Example

seq = "ACGTCTATAAAGGCTGACCTGCTGATGCGTACGATGCTG"
# find TATA occurrences 
print("TATA occurrences at indices:", [i for i in range(len(seq)-3) if seq[i:i+4]=="TATA"])

# GC content
gc = (seq.count("G") + seq.count("C")) / len(seq)
print(f"GC content: {gc:.2%}")

# one-hot first 8 bases
alphabet = "ACGT"
one_hot = [[1 if base==a else 0 for a in alphabet] for base in seq[:8]]
print("First 8 bases:", seq[:8])
print("One-hot (A,C,G,T):")
for b, v in zip(seq[:8], one_hot):
    print(b, v)
