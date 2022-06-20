# 2022/6/6
# bfsujason@163.com

'''

python utils/overlap.py \
  -i data/literary/mac/src/overlap \
  -o data/literary/mac/src/overlap.labse.emb \
  -n 5
'''

import os
import time
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer

def main():
    parser = argparse.ArgumentParser(description='Multilingual sentence embeddings')
    parser.add_argument('-i', '--input', type=str, required=True, help='Overlap file.')
    parser.add_argument('-o', '--output', type=str, required=True, help='Embedding file.')
    args = parser.parse_args()
  
    overlap = get_overlap(args.input)
  
    model = load_model()
    embed_overlap(model, overlap, args.output)
  
def embed_overlap(model, overlap, fout):
    print("Embedding text ...")
    t_0 = time.time()
    num_sents = len(overlap)
    embed = model.encode(overlap)
    embed.tofile(fout)
    print("It takes {:.3f} seconods to embed {} sentences.".format(time.time() - t_0, num_sents))

def load_model():
    print("Loading embedding model ...")
    t0 = time.time()
    model = SentenceTransformer('LaBSE')
    print("It takes {:.3f} seconods to load the model.".format(time.time() - t0))
    return model
    
def get_overlap(file):
    overlap = []
    with open(file, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            overlap.append(line)
    return overlap
  
if __name__ == '__main__':
    main()
