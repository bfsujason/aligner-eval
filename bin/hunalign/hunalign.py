# 2021/11/27
# bfsujason@163.com

"""
Usage:
python bin/hunalign/hunalign.py \
  -s data/mac/src \
  -t data/mac/tgt \
  -o data/mac/auto \
  -m data/mac/meta_data.tsv
"""

import os
import time
import shutil
import platform
import argparse

from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(description='Sentence alignment using Hunalign')
    parser.add_argument('-s', '--src', type=str, required=True, help='Source directory.')
    parser.add_argument('-t', '--tgt', type=str, required=True, help='Target directory.')
    parser.add_argument('-o', '--out', type=str, required=True, help='Output directory.')
    parser.add_argument('-m', '--meta', type=str, required=True, help='Metadata file.')
    args = parser.parse_args()
  
    make_dir(args.out)
    job_files = create_job_files(args.meta, args.src, args.tgt, args.out)
  
    # check system OS
    OS = platform.system()
    if OS == 'Windows':
        hunalign_bin = os.path.abspath('bin/hunalign/hunalign.exe')
    elif OS == 'Linux':
        hunalign_bin = os.path.abspath('bin/hunalign/hunalign')
    
    for job in job_files:
        dic = os.path.splitext(os.path.basename(job))[0] + '.dic'
        hunalign_dic = os.path.abspath(os.path.join('bin/hunalign', dic))
        run_hunalign(hunalign_bin, hunalign_dic, job)
    
    convert_format(args.out)
  
def convert_format(dir):
    for file in sorted(os.listdir(dir)):
        fp_in = os.path.join(dir, file)
        fp_out = os.path.join(dir, file + '.align')
        alignment = _convert_format(fp_in, fp_out)
        write_alignment(alignment, fp_out)
        os.unlink(fp_in)

def _convert_format(fp_in, fp_out):
    src_id = -1
    tgt_id = -1
    alignment = []
  
    with open(fp_in, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip(' \r\n')
            items = line.split('\t');
            if not items[0] and not items[1]:
                continue
            src_seg_len, src_seg_id = _parse_seg(items[0], src_id)
            tgt_seg_len, tgt_seg_id = _parse_seg(items[1], tgt_id)
            src_id += src_seg_len
            tgt_id += tgt_seg_len
            alignment.append((src_seg_id, tgt_seg_id))
  
    return alignment

def write_alignment(alignment, fp_out):
    with open(fp_out, 'wt', encoding='utf-8') as f:
        for id in alignment:
            f.write("{}:{}\n".format(id[0], id[1]))
  
def _parse_seg(seg, id):
    seg_len = 0
    seg_id = []
    if seg:
        sents = seg.split(' ~~~ ')
        seg_len = len(sents)
        seg_id = [id + x for x in range(1, seg_len+1)]
    return seg_len, seg_id

def run_hunalign(bin, dic, job):
    cmd = "{} -text -batch {} {}".format(bin, dic, job)
    os.system(cmd)
    os.unlink(job)
   
def create_job_files(meta_file, src_dir, tgt_dir, out_dir):
    jobs = defaultdict(list)
    meta = get_meta(meta_file)
    for recs in meta:
        id, src_lang, tgt_lang = recs[0], recs[1], recs[2]
        src_path = os.path.abspath(os.path.join(src_dir, id + '.tok'))
        tgt_path = os.path.abspath(os.path.join(tgt_dir, id + '.tok'))
        out_path = os.path.abspath(os.path.join(out_dir, id))
        job = '\t'.join([src_path, tgt_path, out_path])
        if src_lang == 'zh' and tgt_lang == 'en': # Chinese to English
            jobs['ec'].append(job)
        elif src_lang == 'en' and tgt_lang == 'zh': # English to Chinese
            jobs['ce'].append(job)
    
    job_files = []
    for dic, job in jobs.items():
        job_file = os.path.abspath(os.path.join(out_dir, dic + '.job'))
        with open(job_file, 'wt', encoding='utf-8', newline='\n') as f:
            f.write('\n'.join(job))
        job_files.append(job_file)
    
    return job_files
    
def get_meta(meta_file):
    meta = []
    with open(meta_file, 'rt', encoding='utf-8') as f:
        next(f) # skip header
        for row in f:
            recs = row.strip().split('\t')
            meta_recs = (recs[0], recs[1], recs[2])
            if meta_recs not in meta:
                meta.append(meta_recs)
    return meta

def make_dir(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
  
if __name__ == '__main__':
    t_0 = time.time()
    main()
    print("It takes {:.3f} seconds to align all the sentences.".format(time.time() - t_0))
