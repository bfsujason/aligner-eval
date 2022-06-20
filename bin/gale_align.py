# 2021/11/27
# bfsujason@163.com

"""
Usage:

python bin/gale_align.py \
  -s data/mac/src \
  -t data/mac/tgt \
  -o data/mac/auto
"""

import os
import re
import time
import math
import shutil
import argparse
import numba as nb
import numpy as np

def main():
    # user-defined parameters
    parser = argparse.ArgumentParser(description='Sentence alignment using Gale-Church Algrorithm')
    parser.add_argument('-s', '--src', type=str, required=True, help='Source directory.')
    parser.add_argument('-t', '--tgt', type=str, required=True, help='Target directory.')
    parser.add_argument('-o', '--out', type=str, required=True, help='Output directory.')
    args = parser.parse_args()
    
    make_dir(args.out)
    jobs = create_jobs(args.src, args.tgt, args.out)
  
    # alignment types
    align_types = np.array(
    [
      [0,1],
      [1,0],
      [1,1],
      [1,2],
      [2,1],
      [2,2],
    ], dtype=np.int8)
  
    # prior probability
    priors = np.array([0, 0.0099, 0.89, 0.089, 0.011])
  
    # mean and variance
    c = 1
    s2 = 6.8
  
    # perform gale-church align
    for job in jobs:
        src_file, tgt_file, out_file = job.split('\t')
        print("Aligning {} to {}".format(src_file, tgt_file))
        src_lines = read_lines(src_file)
        tgt_lines = read_lines(tgt_file)
        src_len = calculate_sent_len(src_lines)
        tgt_len = calculate_sent_len(tgt_lines)
    
        m = len(src_lines)
        n = len(tgt_lines)
        w, search_path = find_search_path(m, n)
        back, cost = align(src_len, tgt_len, w, search_path, align_types, priors, c, s2)
        alignments = back_track(m, n, back, cost, search_path, align_types)
        
        save_alignments(alignments, out_file)

def save_alignments(alignments, file):
    with open(file, 'wt', encoding='utf-8') as f:
        for id in alignments:
            f.write("{}:{}\n".format(id[0], id[1]))

def back_track(i, j, b, c, search_path, a_types):
    alignment = []
    while not (i == 0 and j == 0):
        j_offset = j - search_path[i][0]
        a = b[i][j_offset]
        s = a_types[a][0]
        t = a_types[a][1]
        src_range = [i - offset - 1 for offset in range(s)][::-1]
        tgt_range = [j - offset - 1 for offset in range(t)][::-1]
        
        
        prev_i = i - s
        prev_j = j - t
        prev_j_offset = prev_j - search_path[prev_i][0]
        prev_score = c[prev_i][prev_j_offset]
        cur_score = c[i][j_offset]
        score = cur_score - prev_score
        alignment.append((src_range, tgt_range, score))
        
        i = prev_i
        j = prev_j
        
    return alignment[::-1]

@nb.jit(nopython=True, fastmath=True, cache=True)
def align(src_len, tgt_len, w, search_path, align_types, priors, c, s2):
    # Initialize cost and backpointer matrix.
    m = src_len.shape[0] - 1
    cost = np.zeros((m + 1, 2 * w + 1), dtype=nb.float32)
    back = np.zeros((m + 1, 2 * w + 1), dtype=nb.uint8)
    cost[0][0] = 0
    back[0][0] = -1

    for i in range(m + 1):
        i_start = search_path[i][0]
        i_end = search_path[i][1]

        for j in range(i_start, i_end + 1):
            if i + j == 0:
                continue
   
            best_score = np.inf
            best_a = -1
            for a in range(align_types.shape[0]):
                a_1 = align_types[a][0]
                a_2 = align_types[a][1]
                prev_i = i - a_1
                prev_j = j - a_2
        
                if prev_i < 0 or prev_j < 0 :  # no previous cell 
                    continue
        
                prev_i_start = search_path[prev_i][0]
                prev_i_end =  search_path[prev_i][1]
        
                if prev_j < prev_i_start or prev_j > prev_i_end: # out of bound of cost matrix
                    continue
            
                prev_j_offset = prev_j - prev_i_start

                score = cost[prev_i][prev_j_offset] - math.log(priors[a_1 + a_2]) + \
                    get_score(src_len[i] - src_len[i - a_1], tgt_len[j] - tgt_len[j - a_2], c, s2)
        
                if score < best_score:
                    best_score = score
                    best_a = a
      
            j_offset = j - i_start
            cost[i][j_offset] = best_score
            back[i][j_offset] = best_a
   
    return back, cost
  
@nb.jit(nopython=True, fastmath=True, cache=True)
def get_score(len_s, len_t, c, s2): 
    mean = (len_s + len_t / c) / 2
    z = (len_t - len_s * c) / math.sqrt(mean * s2)
  
    pd = 2 * (1 - norm_cdf(abs(z)))
    if pd > 0:
        return -math.log(pd)
    
    return 25

@nb.jit(nopython=True, fastmath=True, cache=True)
def norm_cdf(z):
    t = 1/float(1+0.2316419*z)
    p_norm = 1 - 0.3989423*math.exp(-z*z/2) * ((0.319381530 * t) + \
                                               (-0.356563782 * t)+ \
                                               (1.781477937 * t) + \
                                               (-1.821255978* t) + \
                                               (1.330274429 * t))
  
    return p_norm
  
def find_search_path(src_len,
                     tgt_len,
                     min_win_size = 250,
                     percent=0.06):
    """
    Find the window size and search path for the DP table.
    Args:
        src_len: int. Number of source sentences.
        tgt_len: int. Number of target sentences.
        min_win_size: int. Minimum window size.
        percent. float. Percent of longer sentences.
    Returns:
        win_size: int. Window size along the diagonal of the DP table.
        search_path: numpy array of shape (src_len + 1, 2), containing the start
                     and end index of target sentences for each source sentence.
                     One extra row is added in the search_path for the calculation
                     of deletions and omissions.
    """
    win_size = max(min_win_size, int(max(src_len, tgt_len) * percent))
    search_path = []
    yx_ratio = tgt_len / src_len
    for i in range(0, src_len + 1):
        center = int(yx_ratio * i)
        win_start = max(0, center - win_size)
        win_end = min(center + win_size, tgt_len)
        search_path.append([win_start, win_end])
    return win_size, np.array(search_path)
     
def calculate_sent_len(sents):
    sent_lens = []
    sent_lens.append(0)
    for i, sent in enumerate(sents):
        # UTF-8 byte length
        sent_lens.append(sent_lens[i] + len(sent.strip().encode("utf-8"))) 
    return np.array(sent_lens)
    
def read_lines(file):
    lines = []
    with open(file, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            lines.append(line)
    return lines

def create_jobs(src, tgt, out):
    jobs = []
    for file in os.listdir(src):
        if re.match(r'^\d+$', file):
            src_path = os.path.abspath(os.path.join(src, file))
            tgt_path = os.path.abspath(os.path.join(tgt, file))
            out_path = os.path.abspath(os.path.join(out, file + '.align'))
            jobs.append('\t'.join([src_path, tgt_path, out_path]))
    return jobs
  
def make_dir(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

if __name__ == '__main__':
    t_0 = time.time()
    main()
    print("It takes {:.3f} seconds to align all the sentences.".format(time.time() - t_0))
