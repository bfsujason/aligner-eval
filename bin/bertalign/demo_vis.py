import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

from corelib import *

def main():
    # User-defined parameters.
    parser = argparse.ArgumentParser('Sentence alignment using Bertalign')
    parser.add_argument('-s', '--src', type=str, required=True, help='Demo source text.')
    parser.add_argument('-t', '--tgt', type=str, required=True, help='Demo target text.')
    parser.add_argument('--max_align', type=int, default=5,
            help='Maximum number of source+target sentences allowed in each alignment segment.')
    parser.add_argument('--win', type=int, default=5, help='Window size for the second-pass alignment.')
    parser.add_argument('--top_k', type=int, default=3, help='Top-k target neighbors of each source sentence.')
    parser.add_argument('--skip', type=float, default=-0.1, help='Similarity score for 0-1 and 1-0 alignment.')
    parser.add_argument('--margin', action='store_true', help='Margin-based modified cosine similarity.')
    args = parser.parse_args()
    
    # Read in source and target sentences.
    src_file = args.src
    tgt_file = args.tgt
    print("Aligning {} to {}".format(src_file, tgt_file))
    src_lines = open(src_file, 'rt', encoding="utf-8").readlines()
    tgt_lines = open(tgt_file, 'rt', encoding="utf-8").readlines()
    if len(src_lines) > 50 or len(tgt_lines) > 50:
        raise Exception('There are more than 50 sentences for visualization.')
    
    # Read in source and target embeddings.
    src_overlap = os.path.join('data/mac/src/overlap')
    src_embed = os.path.join('data/mac/src/overlap.emb')
    tgt_overlap = os.path.join('data/mac/tgt/overlap')
    tgt_embed = os.path.join('data/mac/tgt/overlap.emb')
    src_sent2line, src_line_embeddings = \
        read_in_embeddings(src_overlap, src_embed)
    tgt_sent2line, tgt_line_embeddings = \
        read_in_embeddings(tgt_overlap, tgt_embed)
        
    # Convert source and target sentences into vectors.
    t_0 = time.time()
    src_vecs, src_lens = \
      doc2feats(src_sent2line, src_line_embeddings, src_lines, args.max_align - 1)
    tgt_vecs, tgt_lens = \
      doc2feats(tgt_sent2line, tgt_line_embeddings, tgt_lines, args.max_align - 1)
    char_ratio = np.sum(src_lens[0,]) / np.sum(tgt_lens[0,])
    print("Vectorizing soure and target texts takes {:.3f} seconds.".format(time.time() - t_0))

    # Find the top_k similar target sentences for each source sentence.
    t_1 = time.time()
    D, I = find_top_k_sents(src_vecs[0,:], tgt_vecs[0,:], k=args.top_k)
    print("Finding top-k sentences takes {:.3f} seconds.".format(time.time() - t_1))

    # Find optimal 1-1 alignments using dynamic programming.
    t_2 = time.time()
    m = len(src_lines)
    n = len(tgt_lines)
    first_alignment_types = get_alignment_types(2) # 0-1, 1-0, 1-1
    first_w, first_path = find_first_search_path(m, n)
    first_pointers = first_pass_align(m, n, first_w,
                                      first_path, first_alignment_types,
                                      D, I)
    first_alignment = first_back_track(m, n, first_pointers,
                                       first_path, first_alignment_types)
    print("First-pass alignment takes {:.3f} seconds.".format(time.time() - t_2))
    
    # Find optimal m-to-n alignments using dynamic programming.
    t_3 = time.time()
    second_alignment_types = get_alignment_types(args.max_align)
    second_w, second_path = find_second_path(first_alignment, args.win, m, n)
    second_pointers, second_cost = second_pass_align(src_vecs, tgt_vecs, src_lens, tgt_lens,
                                        second_w, second_path, second_alignment_types,
                                        char_ratio, args.skip, margin=args.margin)
    second_alignment, second_scores = second_back_track(m, n, second_pointers, second_cost,
                                         second_path, second_alignment_types)
    print("Second pass alignment takes {:.3f} seconds.".format(time.time() - t_3))

    # Print alignment results.
    print_alignments(second_alignment, second_scores)
    
    # Plot aligning process.
    print("Visualize alignment process ...")
    show_vis(src_vecs[0, :], tgt_vecs[0, :], first_alignment, second_path, second_alignment, second_scores)
    
def show_vis(src_vecs,
             tgt_vecs,
             first_alignment,
             second_path,
             second_alignment,
             second_scores):
    matrix = np.matmul(src_vecs, tgt_vecs.T)

    col_zero = np.zeros((matrix.shape[0], 1))
    matrix = np.concatenate((col_zero, matrix), axis=1)

    row_zero = np.zeros((1,matrix.shape[1]))
    matrix = np.concatenate((row_zero, matrix), axis=0)

    anchor_x, anchor_y = [0], [0]
    for (src_id, tgt_id) in first_alignment:
        anchor_x.append(src_id)
        anchor_y.append(tgt_id)

    figure = plt.figure()
    figure.set_figwidth(20)
    figure.set_figheight(10)
    figure.tight_layout()

    ax = figure.add_subplot(1,2,1)
    ax.set_title('First-pass Alignment', pad=20, fontsize=20)
    plt.locator_params(axis="both", integer=True, tight=True)
    plt.xlabel('Source', fontsize=20, labelpad=20)
    plt.ylabel('Target', fontsize=20, labelpad=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.imshow(matrix.T,cmap='gray_r', origin='lower', interpolation='none')
    plt.plot(anchor_x,anchor_y, 'wo-', markersize=5, linewidth=2)

    matrix.fill(0.1)
    path = find_path(second_alignment, second_scores)
    src_num = src_vecs.shape[0]
    tgt_num = tgt_vecs.shape[0]
    for i in range(src_num + 1):
        for j in range(tgt_num+1):
            if not j in range(second_path[i][0], second_path[i][1] + 1):
                matrix[i][j] = 0
    for idx_score in path:
        x, y, sim_score = idx_score
        matrix[x+1][y+1] = sim_score

    ax = figure.add_subplot(1,2,2)
    ax.set_title('Second-pass Alignment', pad=20, fontsize=20)
    plt.locator_params(axis="both", integer=True, tight=True)
    plt.xlabel('Source', fontsize=20, labelpad=20)
    plt.ylabel('Target', fontsize=20, labelpad=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.imshow(matrix.T,cmap='gray_r', origin='lower', interpolation='none')
    
    plt.savefig('demo_vis.png', dpi=300)
    
def print_alignments(alignments, scores):
    for (x, y), score in zip(alignments, scores):
        print("{}:{}:{:.3f}".format(x, y, score))
      
def doc2feats(sent2line, line_embeddings, lines, num_overlaps):
  """
  Convert texts into vectors.
  Args:
      sent2line: dict. Map each sentence to its ID.
      line_embeddings: numpy array of sentence embeddings.
      lines: list. A list of sentences.
      num_overlaps: int. Maximum number of overlapping sentences allowed.
  Returns:
      vecs0: numpy array of shape (num_overlaps, num_lines, embedding_size)
             for overlapping sentence embeddings.
      vecs1: numpy array of shape (num_overlaps, num_lines)
             for overlapping sentence lengths.
  """
  lines = [preprocess_line(line) for line in lines]
  vecsize = line_embeddings.shape[1]
  vecs0 = np.empty((num_overlaps, len(lines), vecsize), dtype=np.float32)
  vecs1 = np.empty((num_overlaps, len(lines)), dtype=np.int32)
  for ii, overlap in enumerate(range(1, num_overlaps + 1)):
    for jj, out_line in enumerate(layer(lines, overlap)):
      try:
        line_id = sent2line[out_line]
      except KeyError:
        logger.warning('Failed to find overlap=%d line "%s". Will use random vector.', overlap, out_line)
        line_id = None
      if line_id is not None:
        vec = line_embeddings[line_id]
        #vec = vec / np.linalg.norm(vec) 
      else:
        vec = np.random.random(vecsize) - 0.5
        vec = vec / np.linalg.norm(vec)        
      vecs0[ii, jj, :] = vec
      vecs1[ii, jj] = len(out_line.encode("utf-8"))
  return vecs0, vecs1

def layer(lines, num_overlaps, comb=' '):
  """
  Make front-padded overlapping sentences.
  Args:
      lines: list. A list of sentences.
      num_overlaps: int. Number of overlapping sentences.
      comb: str. Symbol for sentence concatenation.
  Returns:
      out: list. Front-padded overlapping sentences.
                 Similar to n-grams for sentences.
  """
  if num_overlaps < 1:
    raise Exception('num_overlaps must be >= 1')
  out = ['PAD', ] * min(num_overlaps - 1, len(lines))
  for i in range(len(lines) - num_overlaps + 1):
    out.append(comb.join(lines[i:i + num_overlaps]))    
  return out

def preprocess_line(line):
  """
  Clean each line of the text.
  """
  line = line.strip()
  if len(line) == 0:
    line = 'BLANK_LINE'  
  return line

def read_in_embeddings(text_file, embed_file):
  """
  Read in the overlap lines and line embeddings.
  Args:
      text_file: str. Overlap file path.
      embed_file: str. Embedding file path.
  Returns:
      sent2line: dict. Map overlap sentences to line IDs.
      line_embeddings: numpy array of the shape (num_lines, embedding_size).
                       For sentence-transformers, the embedding_size is 768. 
  """
  sent2line = dict()
  with open(text_file, 'rt', encoding="utf-8") as f:
    for i, line in enumerate(f):
      sent2line[line.strip()] = i
  line_embeddings = np.fromfile(embed_file, dtype=np.float32)
  embedding_size = line_embeddings.size // len(sent2line)
  line_embeddings.resize(line_embeddings.shape[0] // embedding_size, embedding_size)
  return sent2line, line_embeddings

def find_path(alignment, score):
    path = []
    last_src, last_tgt = -1, -1
    for i, (src_idx, tgt_idx) in enumerate(alignment):
        if len(src_idx) == 0:
            src_idx = [last_src]
        elif len(tgt_idx) == 0:
            tgt_idx = [last_tgt]
        last_src, last_tgt = src_idx[-1], tgt_idx[-1]
        combined_idx = combine_idx(src_idx, tgt_idx)    
        for idx in combined_idx:
            idx.append(score[i])
        path.extend(combined_idx)
    return path
    
def combine_idx(src_idx, tgt_idx):
    combined_idx = [[i, j] for i in src_idx for j in tgt_idx]
    return combined_idx
    
if __name__ == '__main__':
    t_0 = time.time()
    main()
    print("It takes {:.3f} seconds for alignment and visualization.".format(time.time() - t_0))
