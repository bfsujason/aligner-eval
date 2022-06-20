# 2021/11/29
# bfsujason@163.com

"""
Usage:

python bin/bert_align.py \
  -s data/mac/dev/zh \
  -t data/mac/dev/en \
  -o data/mac/dev/auto \
  -m labse --max_align 8 --margin
"""

import os
import re
import time
import shutil
import argparse
import numpy as np

from corelib import *

def main():
  # User-defined parameters.
  parser = argparse.ArgumentParser('Sentence alignment using Bertalign')
  parser.add_argument('-s', '--src', type=str, required=True, help='Source texts directory.')
  parser.add_argument('-t', '--tgt', type=str, required=True, help='Target texts directory.')
  parser.add_argument('-o', '--out', type=str, required=True, help='Alignment directory.')
  parser.add_argument('-m', '--model', type=str, required=True, help='Embedding model.')
  parser.add_argument('--max_align', type=int, default=5,
            help='Maximum number of source+target sentences allowed in each alignment segment.')
  parser.add_argument('--win', type=int, default=5, help='Window size for the second-pass alignment.')
  parser.add_argument('--top_k', type=int, default=3, help='Top-k target neighbors of each source sentence.')
  parser.add_argument('--skip', type=float, default=-0.1, help='Similarity score for 0-1 and 1-0 alignment.')
  parser.add_argument('--margin', action='store_true', help='Margin-based modified cosine similarity.')
  args = parser.parse_args()
  
  # Read in source and target embeddings.
  src_overlap = os.path.join(args.src, 'overlap')
  tgt_overlap = os.path.join(args.tgt, 'overlap')

  if args.model == 'labse':
    src_embed = os.path.join(args.src, 'overlap.labse.emb')
    tgt_embed = os.path.join(args.tgt, 'overlap.labse.emb')
  elif args.model == 'laser':
    src_embed = os.path.join(args.src, 'overlap.laser.emb')
    tgt_embed = os.path.join(args.tgt, 'overlap.laser.emb')

  src_sent2line, src_line_embeddings = \
    read_in_embeddings(src_overlap, src_embed)
  tgt_sent2line, tgt_line_embeddings = \
    read_in_embeddings(tgt_overlap, tgt_embed)
  
  # Perform stentence alignment.
  make_dir(args.out)
  jobs = create_jobs(args.src, args.tgt, args.out)
  for job in jobs:
    src_file, tgt_file, out_file = job.split('\t')
    print("Aligning {} to {}".format(src_file, tgt_file))

    # Convert source and target texts into vectors.
    t_0 = time.time()
    src_lines = open(src_file, 'rt', encoding="utf-8").readlines()
    tgt_lines = open(tgt_file, 'rt', encoding="utf-8").readlines()
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

    # save alignment results
    print_alignments(second_alignment, out_file)

def print_alignments(alignments, out):
  with open(out, 'wt', encoding='utf-8') as f:
    for x, y in alignments:
      f.write("{}:{}\n".format(x, y))

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
        vec = vec / np.linalg.norm(vec)
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

def create_jobs(src_dir, tgt_dir, alignment_dir):
  """
  Create a job list of source, target and alignment file paths.
  """
  jobs = []
  for file in os.listdir(src_dir):
    if re.match(r'^\d+$', file):
      src_path = os.path.abspath(os.path.join(src_dir, file))
      tgt_path = os.path.abspath(os.path.join(tgt_dir, file))
      out_path = os.path.abspath(os.path.join(alignment_dir, file + '.align'))
      jobs.append('\t'.join([src_path, tgt_path, out_path]))  
  return jobs

def make_dir(auto_alignment_path):
  """
  Make an empty diretory for saving automatic alignment results. 
  """
  if os.path.isdir(auto_alignment_path):
    shutil.rmtree(auto_alignment_path)
  os.makedirs(auto_alignment_path, exist_ok=True)
  
if __name__ == '__main__':
  t_0 = time.time()
  main()
  print("It takes {:.3f} seconds to align all the sentences.".format(time.time() - t_0))
