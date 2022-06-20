# 2021/11/27
# bfsujason@163.com

"""
Usage:

python bin/bleualign/bleualign.py \
  -s data/mac/src \
  -t data/mac/tgt \
  -o data/mac/auto \
  -m data/mac/meta_data.tsv
"""

import os
import time
import shutil
import argparse

def main():
    parser = argparse.ArgumentParser(description='Sentence alignment using Bleualign')
    parser.add_argument('-s', '--src', type=str, required=True, help='Source directory.')
    parser.add_argument('-t', '--tgt', type=str, required=True, help='Target directory.')
    parser.add_argument('-o', '--out', type=str, required=True, help='Output directory.')
    parser.add_argument('-m', '--meta', type=str, required=True, help='Metadata file.')
    parser.add_argument('--bleurt', type=str, help='Use bleurt as the similarity measure.')
    args = parser.parse_args()
  
    make_dir(args.out)
    job_file = create_job_file(args.meta, args.bleurt, args.src, args.tgt, args.out)
    print(job_file)
  
    if args.bleurt:
        bleualign_bin = os.path.abspath('bin/bleualign/batch_align_bleurt.py')
        run_bleurtalign(bleualign_bin, job_file, args.bleurt)
    else:
        bleualign_bin = os.path.abspath('bin/bleualign/batch_align.py')
        run_bleualign(bleualign_bin, job_file)
    
    convert_format(args.out)
    delete_temp_files(args.src)

def delete_temp_files(dir):
    for file in os.listdir(dir):
        if file.endswith('.detok'):
            os.remove(os.path.join(dir, file))

def convert_format(dir):
    for file in os.listdir(dir):
        if file.endswith('-s'):
            file_id = file.split('.')[0]
            src = os.path.join(dir, file)
            tgt = os.path.join(dir, file_id + '.align-t')
            out = os.path.join(dir, file_id + '.align')
            _convert_format(src, tgt, out)
            os.unlink(src)
            os.unlink(tgt)

def _convert_format(src, tgt, path):
    src_align = read_alignment(src)
    tgt_align = read_alignment(tgt)
    with open(path, 'wt', encoding='utf-8') as f:
        for x, y in zip(src_align, tgt_align):
            f.write("{}:{}\n".format(x,y))

def read_alignment(file):
    alignment = []
    with open(file, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            alignment.append([int(x) for x in line.split(',')])
    return alignment
      
def run_bleualign(bin, job):
    cmd = "python {} {}".format(bin, job)
    os.system(cmd)
    os.unlink(job)

def run_bleurtalign(bin, job, checkpoint):
    cmd = "python {} {} {}".format(bin, job, checkpoint)
    os.system(cmd)
    os.unlink(job)
    
def detok_text(old_file, new_file):
    detok_text = []
    text = open(old_file, 'rt', encoding='utf-8').read()
    text = text.strip()
    
    detok_text = text.replace(' ','')
    with open(new_file, 'wt', encoding='utf-8') as f:
        f.write(detok_text)
       
def create_job_file(meta_file, is_bleurt, src_dir, tgt_dir, out_dir):
    jobs = []
    meta = get_meta(meta_file)
    for recs in meta:
        id, src_lang, tgt_lang = recs[0], recs[1], recs[2]
        src_path = os.path.abspath(os.path.join(src_dir, id))
        
        trans_path = os.path.abspath(os.path.join(src_dir, id + '.trans'))
        tgt_path = os.path.abspath(os.path.join(tgt_dir, id))

        if tgt_lang == 'zh':
            if is_bleurt:
                orig_trans_path = trans_path
                trans_path = orig_trans_path + '.detok'
                detok_text(orig_trans_path, trans_path)
            else:
                tgt_path = tgt_path + '.tok'
            
        out_path = os.path.abspath(os.path.join(out_dir, id + '.align'))
        jobs.append('\t'.join([trans_path, src_path, tgt_path, out_path]))
        
    job_file = os.path.abspath(os.path.join(out_dir, 'bleualign.job'))
    with open(job_file, 'wt', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(jobs))
    return job_file
    
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
