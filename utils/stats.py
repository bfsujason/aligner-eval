import os
import regex as re
from ast import literal_eval
import argparse

def main():
    parser = argparse.ArgumentParser(description='Compute corpus statistics.')
    parser.add_argument('-i', '--input', type=str, required=True, help='Data directory.')
    parser.add_argument('-o', '--output', type=str, required=True, help='Stats file.')
    args = parser.parse_args()
    
    stats = calculate_stats(args.input)
    write_stats(stats, args.output)

def write_stats(stats, file):
    with open(file, 'wt', encoding='utf-8') as f:
        for record in stats:
            f.write(record + "\n")

def calculate_stats(dir):
    src_dir = os.path.join(dir, 'src')
    tgt_dir = os.path.join(dir, 'tgt')
    gold_dir = os.path.join(dir, 'gold')
    stats = []
    header = "\t".join(['id', 'src_sents', 'src_tokens', 'tgt_sents', 'tgt_tokens', 'alignments', '1to1_alignments'])
    stats.append(header)
    for file in sorted(os.listdir(src_dir)):
        if re.match(r'^\d+$', file):
            src_file = os.path.join(src_dir, file + '.tok')
            tgt_file = os.path.join(tgt_dir, file + '.tok')
            gold_file = os.path.join(gold_dir, file + '.align')
            src_sent_num, src_tok_num = count_sent_and_tok_nums(src_file)
            tgt_sent_num, tgt_tok_num = count_sent_and_tok_nums(tgt_file)
            align_num, one_num = count_alignment_nums(gold_file)
            stats.append("\t".join([file, str(src_sent_num), str(src_tok_num), str(tgt_sent_num), str(tgt_tok_num), str(align_num), str(one_num)]))
    return stats

def count_alignment_nums(file):
    align_num, one_num = 0, 0
    with open(file, 'rt', encoding="utf-8") as f:
        for line in f:
            align_num += 1
            fields = [x.strip() for x in line.split(':') if len(x.strip())]
            src_len = len(literal_eval(fields[0]))
            tgt_len = len(literal_eval(fields[1]))
            if src_len + tgt_len == 2:
                one_num += 1     
    return align_num, one_num
    
def count_sent_and_tok_nums(file):
    sent_num, tok_num = 0, 0
    with open(file, 'rt', encoding='utf-8') as f:
        for line in f:
            sent_num += 1
            line = line.strip()
            tokens = line.split()
            for token in tokens:
                if re.match(r'^\p{P}+$', token):
                    continue
                tok_num += 1
            
    return sent_num, tok_num
            
if __name__ == '__main__':
    main()
