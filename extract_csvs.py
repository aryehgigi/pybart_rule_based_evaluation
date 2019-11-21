from collections import defaultdict
import csv
import json
import math
import os
import argparse


def list_files(names):
    for i, name in enumerate(names):
        print(i, name)


def main(names):
    rels = defaultdict(defaultdict)
    general = dict()
    for name in names:
        try:
            with open(logs_dir + name, "r") as f:
                d = json.load(f)
        except json.decoder.JSONDecodeError:
            continue
        
        is_test = 'test' in name
        name = name.replace("_dev", "")
        name = name.replace("_test", "")
    
        if name in general:
            general[name][5 if is_test else 1:-1 if is_test else 5] = [d[1], d[2], d[3], d[4]]
        else:
            general[name] = [name, float('inf'), float('inf'), float('inf'), float('inf'), d[1], d[2], d[3], d[4]] if is_test else \
                [name, d[1], d[2], d[3], d[4], float('inf'), float('inf'), float('inf'), float('inf')]
        
        for rel, scores in d[0].items():
            if (rel in rels) and (name in rels[rel].keys()):
                rels[rel][name][7 if is_test else 1:-1 if is_test else 7] = \
                    [scores['precision'], scores['recall'], scores['f1'], scores['relevant'], scores['retrieved'], scores['retrievedAndRelevant']]
            else:
                rels[rel][name] = [name, -math.inf, -math.inf, -math.inf, -math.inf, -math.inf, -math.inf, scores['precision'], scores['recall'], scores['f1'], scores['relevant'], scores['retrieved'], scores['retrievedAndRelevant']] if is_test else \
                    [name, scores['precision'], scores['recall'], scores['f1'], scores['relevant'], scores['retrieved'], scores['retrievedAndRelevant'], -math.inf, -math.inf, -math.inf, -math.inf, -math.inf, -math.inf]
    
    
    for rel, scores in rels.items():
        with open(logs_dir + "output/" + rel + ".csv", "w") as f:
            writer = csv.writer(f)
            writer.writerows(scores)
        print(rel)
        for score in scores.values():
            print("\t{: <43}{: >9.4f}{: >9.4f}{: >9.4f}{: >9}{: >9}{: >9}{: >9.4f}{: >9.4f}{: >9.4f}{: >9}{: >9}{: >9}".format(*score))
        print()
    
    
    with open(logs_dir + "output/" + "general.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(general.values())
    
    
    print("general")
    for score in general.values():
        print("\t{: <43}{: >9.4f}{: >9.4f}{: >9.4f}{: >9.4f}{: >9.4f}{: >9.4f}{: >9.4f}{: >9.4f}".format(*score))


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-a', '--action', type=str, default='ex')
    arg_parser.add_argument('-i', '--file_indices', action='append', default=None)
    args = arg_parser.parse_args()
    
    logs_dir = "/home/inbaryeh/spike/server/logs/"
    names = [f for f in os.listdir(logs_dir) if os.path.isfile(os.path.join(logs_dir, f))]
    
    if args.action == 'ex':
        if args.file_indices:
            names = [names[int(idx)] for idx in args.file_indices]
        main(names)
    elif args.action == 'ls':
        list_files(names)
