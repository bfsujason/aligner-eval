# 2022/6/6
# bfsujason@163.com

'''

python utils/overlap.py \
  -i data/literary/mac/src \
  -o data/literary/mac/src/overlap \
  -n 5
'''

import os
import re
import argparse

def main():
    parser = argparse.ArgumentParser(description='Multilingual sentence embeddings')
    parser.add_argument('-i', '--input', type=str, required=True, help='Data directory.')
    parser.add_argument('-o', '--output', type=str, required=True, help='Overalp file.')
    parser.add_argument('-n', '--num_overlaps', type=int, default=5, help='Maximum number of allowed overlaps.')
    args = parser.parse_args()
    
    overlap = set()
    for file in os.listdir(args.input):
        if re.match(r'^\d+$', file):
            file_path = os.path.join(args.input, file)
            lines = open(file_path, 'rt', encoding='utf-8').readlines()
            for out_line in yield_overlaps(lines, args.num_overlaps):
                overlap.add(out_line)
                
    # for reproducibility
    overlap = list(overlap)
    overlap.sort()
  
    write_overlap(overlap, args.output)
    
def write_overlap(overlap, outfile):
    with open(outfile, 'wt', encoding="utf-8") as fout:
        for line in overlap:
            fout.write(line + '\n')

def yield_overlaps(lines, num_overlaps):
    lines = [preprocess_line(line) for line in lines]
    for overlap in range(1, num_overlaps + 1):
        for out_line in layer(lines, overlap):
            # check must be here so all outputs are unique
            out_line2 = out_line[:10000]  # limit line so dont encode arbitrarily long sentences
            yield out_line2
      
def layer(lines, num_overlaps, comb=' '):
    if num_overlaps < 1:
        raise Exception('num_overlaps must be >= 1')
    out = ['PAD', ] * min(num_overlaps - 1, len(lines))
    for ii in range(len(lines) - num_overlaps + 1):
        out.append(comb.join(lines[ii:ii + num_overlaps]))
    return out
  
def preprocess_line(line):
    line = line.strip()
    if len(line) == 0:
        line = 'BLANK_LINE'
    return line

if __name__ == '__main__':
    main()
