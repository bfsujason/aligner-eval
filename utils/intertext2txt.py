import os
import shutil
import argparse

from xml.etree.ElementTree import parse

def main():
    parser = argparse.ArgumentParser(description='Convert Intertext to plain text')
    parser.add_argument('-i', '--input', type=str, required=True, help='Input directory for Intertext alignments.')
    args = parser.parse_args()
    
    # Create output directories.
    src_dir = os.path.join(os.path.dirname(args.input), 'src')
    tgt_dir = os.path.join(os.path.dirname(args.input), 'tgt')
    gold_dir = os.path.join(os.path.dirname(args.input), 'gold')
    make_dir(src_dir)
    make_dir(tgt_dir)
    make_dir(gold_dir)
    
    # Convert Intertext files.
    input_files = get_input_files(args.input)
    for id, src_file, tgt_file, align_file in input_files:
        print("Processing {} ...".format(id))
        src_sents = get_sents(os.path.join(args.input, src_file))
        tgt_sents = get_sents(os.path.join(args.input, tgt_file))
        alignments = get_alignments(os.path.join(args.input, align_file))
        
        src_txt = os.path.join(src_dir, id)
        tgt_txt = os.path.join(tgt_dir, id)
        gold_txt = os.path.join(gold_dir, id + '.align')
        write_sents(src_sents, src_txt)
        write_sents(tgt_sents, tgt_txt)
        write_alignments(alignments, gold_txt)

def write_alignments(alignments, file):
    with open(file, 'wt', encoding='utf-8') as f:
        for x, y in alignments:
            f.write("{}:{}\n".format(x, y))

def write_sents(sents, file):
    with open(file, 'wt', encoding='utf-8') as f:
        f.write("\n".join(sents))
        
def get_alignments(file):
    doc = parse(file)
    links = []
    for link in doc.iterfind('link'):
        tgt_link, src_link = link.get('xtargets').split(';')
        src_bead = parse_link(src_link)
        tgt_bead = parse_link(tgt_link)
        links.append((src_bead, tgt_bead))
    return links
 
def parse_link(link):
    bead = []
    if len(link) > 0:
        bead = [ int(item.split(':')[1]) - 1 for item in link.split(' ')]
    return bead
    
def get_sents(file):
    doc = parse(file)
    sents = []
    for sent in doc.iterfind('p/s'):
        sents.append(sent.text)
    return sents
    
def get_input_files(dir):
    input_files = []
    for file in os.listdir(dir):
        names = file.split('.')
        if (len(names)) == 4:
            prj, src, tgt, suffix = names
            id = src.split('_')[0]
            src_file = '.'.join([prj, src, suffix])
            tgt_file = '.'.join([prj, tgt, suffix])
            input_files.append([id, src_file, tgt_file, file])
    return input_files
    
def make_dir(dir):
    if os.path.isdir(dir):
        shutil.rmtree(dir)
    os.makedirs(dir, exist_ok=True) 
    
if __name__ == '__main__':
    main()
