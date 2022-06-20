import os
import argparse
from ast import literal_eval

def main():
    parser = argparse.ArgumentParser(description='Sentence alignment evaluation')
    parser.add_argument('-t', '--test', type=str, required=True, help='Test alignment directory.')
    parser.add_argument('-g', '--gold', type=str, required=True, help='Gold alignment directory.')
    parser.add_argument('--number', type=int, help='Number of source plus target sentences.')
    parser.add_argument('--id', type=int, nargs='+', help='Text IDs.')
    args = parser.parse_args()
    
    if args.id:
        text_ids = ["{:03d}.align".format(id) for id in args.id]
        gold_list = [read_alignments(os.path.join(args.gold, x)) for x in sorted(os.listdir(args.gold)) if x in text_ids]
        test_list = [read_alignments(os.path.join(args.test, x)) for x in sorted(os.listdir(args.test)) if x in text_ids]
    else:
        gold_list = [read_alignments(os.path.join(args.gold, x)) for x in sorted(os.listdir(args.gold))]
        test_list = [read_alignments(os.path.join(args.test, x)) for x in sorted(os.listdir(args.test))]
    
    if args.number:
        gold_list = [select_by_n(list, n=args.number) for list in gold_list]
        test_list = [select_by_n(list, n=args.number) for list in test_list]

    res = score_alignments(gold_list=gold_list, test_list=test_list)
    log_final_scores(res)

def log_final_scores(res):
    print(' ----------------------- ')
    print('|             |  Strict |')
    print('| Precision   |   {precision:.3f} |'.format(**res))
    print('| Recall      |   {recall:.3f} |'.format(**res))
    print('| F1          |   {f1:.3f} |'.format(**res))
    print(' ----------------------- ')
    
def select_by_n(alignments, n=2):
    selected_alignments = []
    for src, tgt in alignments:
        if len(src) + len(tgt) == n:
            selected_alignments.append((src, tgt))
    return selected_alignments
    
def score_alignments(gold_list, test_list):
    match = 0
    pcounts = 0
    rcounts = 0
    for goldalign, testalign in zip(gold_list, test_list):
        match += match_alignments(goldalign=goldalign, testalign=testalign)
        pcounts += len(testalign)
        rcounts += len(goldalign)
    
    if pcounts > 0:
        p = match / pcounts
    else:
        p = 0
        
    if rcounts > 0:
        r = match / rcounts
    else:
        r = 0
        
    if p + r == 0:
        f1 = 0
    else:
        f1 = 2 * (p * r) / (p + r)
        
    result = dict(recall=r, precision=p, f1=f1)
    return result
    
def match_alignments(goldalign, testalign):
    match = 0
    if len(goldalign) > 0 and len(testalign) > 0:
        for (test_src, test_target) in testalign:
            if (test_src, test_target) in goldalign:
                match += 1
    return match
    
def read_alignments(fin):
    alignments = []
    with open(fin, 'rt', encoding="utf-8") as infile:
        for line in infile:
            fields = [x.strip() for x in line.split(':') if len(x.strip())]
            if len(fields) < 2:
                raise Exception('Got line "%s", which does not have at least two ":" separated fields' % line.strip())
            try:
                src = literal_eval(fields[0])
                tgt = literal_eval(fields[1])
            except:
                raise Exception('Failed to parse line "%s"' % line.strip())
            alignments.append((src, tgt))
    return alignments
    
if __name__ == '__main__':
    main()
