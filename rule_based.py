from typing import Tuple, List
import argparse
import errno
import os
import networkx as nx
import json
import pickle
import time

from collections import defaultdict
from spike.rest.definitions import Relation
from spike.pattern_generation.gen_pattern import PatternGenerator
from spike.pattern_generation.pattern_selectors import LabelEdgeSelector, WordNodeSelector, LemmaNodeSelector, TriggerVarNodeSelector
from spike.pattern_generation.utils import GenerationFromAnnotatedSamples
from spike.pattern_generation.compilation import spike_compiler
from spike.pattern_generation.sample_types import AnnotatedSample
from spike.pattern_generation.compilation.odinson_compiler import compile_to_odinson_rule
from spike.datamodel.dataset import FileBasedDataSet
from spike.evaluation import eval

from ud2ude_aryehgigi import converter, conllu_wrapper as cw
import spacy
from spacy.tokens import Doc

spike_relations = ["org:country_of_headquarters", "per:cause_of_death", "per:country_of_birth", "per:spouse", "org:founded", "per:children", "per:country_of_death", "per:stateorprovince_of_death", "org:founded_by", "per:cities_of_residence", "per:origin", "org:alternate_names", "org:number_of_employees_members", "per:city_of_death", "per:religion", "org:city_of_headquarters", "per:age", "per:countries_of_residence", "per:schools_attended"]
DEV_TUNED_STRATEGY_INDEX = 0
DEV_TUNED_STRATEGY_INFO = 1.0


def prevent_sentence_boundary_detection(doc):
    for token in doc:
        # This will entirely disable spaCy's sentence detection
        token.is_sent_start = False
    return doc


nlp = spacy.load("en_ud_model")
nlp.add_pipe(prevent_sentence_boundary_detection, name='prevent-sbd', before='parser')
tagger = nlp.get_pipe('tagger')
sbd_preventer = nlp.get_pipe('prevent-sbd')
parser = nlp.get_pipe('parser')


def get_odin_json(tokens, sample_, rel, tags, lemmas, entities, chunks, odin_id):
    start_offsets = []
    end_offsets = []
    offset = 0
    for tok in tokens:
        start_offsets.append(offset)
        end_offsets.append(offset + len(tok))
        offset += len(tok) + len(" ")
    
    start = min(sample_["subj_start"], sample_["obj_start"])
    end = max(sample_["subj_end"] + 1, sample_["obj_end"] + 1)
    tokens_for_text = tokens[start:end]
    
    gold = {"id": "gold_relation_{}".format(odin_id), "text": " ".join(tokens_for_text),
            'tokenInterval': {'start': start, 'end': end},
            "keep": True, "foundBy": "tacred_gold", "type": "RelationMention", "labels": [rel],
            "sentence": 0, "document": "document", "arguments":
                {"subject": [{"type": "TextBoundMention", "sentence": 0, "labels": [sample_["subj_type"]],
                              "tokenInterval": {"start": sample_["subj_start"], "end": sample_["subj_end"] + 1},
                              'id': 'subject_gold_ent_{}'.format(odin_id),
                              'text': " ".join(tokens[sample_["subj_start"]: sample_["subj_end"] + 1]),
                              'document': 'document', 'keep': True, 'foundBy': 'tacred_gold'}],
                 "object": [{"type": "TextBoundMention", "sentence": 0, "labels": [sample_["obj_type"]],
                             "tokenInterval": {"start": sample_["obj_start"], "end": sample_["obj_end"] + 1},
                             'id': 'object_gold_ent_{}'.format(odin_id),
                             'text': " ".join(tokens[sample_["obj_start"]: sample_["obj_end"] + 1]),
                             'document': 'document', 'keep': True, 'foundBy': 'tacred_gold'}]}}
    
    odin_json = {
        "documents": {
            "": {
                "id": str(odin_id),
                "text": " ".join(tokens),
                "sentences": [{"words": tokens, "raw": tokens, "tags": tags, "lemmas": lemmas, "entities": entities,
                               "chunks": chunks, "startOffsets": start_offsets, "endOffsets": end_offsets}],
                "gold_relations": [gold]
            }
        }, "mentions": []}
    
    return odin_json


def fix_entities(sample, pad):
    entities = sample['stanford_ner'] + (["O"] * (pad - len(sample['stanford_ner'])))
    
    for i in range(sample['subj_start'], sample['subj_end'] + 1):
        entities[i] = sample['subj_type']
    
    for i in range(sample['obj_start'], sample['obj_end'] + 1):
        entities[i] = sample['obj_type']
    
    return entities


def search_triggers(subj_start, subj_end, obj_start, obj_end, rel, tokens):
    trigger_toks = []
    for trigger in get_triggers(rel):
        for trigger_start, token in enumerate(tokens):
            trigger_end = trigger_start + len(trigger.split())
            if (trigger.split() == tokens[trigger_start : trigger_end]) and \
               (trigger_end <= subj_start or trigger_start >= subj_end) and \
               (trigger_end <= obj_start or trigger_start >= obj_end):
                   trigger_toks.append((trigger_start, trigger_end))
    return trigger_toks if trigger_toks else [None]


class SampleAryehAnnotator(object):
    @staticmethod
    def annotate_sample(sample_: dict, rel: str, enhance_ud: bool, enhanced_plus_plus: bool, enhanced_extra: bool, convs: int,
                        remove_eud_info: bool, remove_extra_info: bool, odin_id: int = -1, use_triggers: bool = True) -> Tuple[List[AnnotatedSample], dict]:
        doc = Doc(nlp.vocab, words=sample_['token'])
        _ = tagger(doc)
        _ = sbd_preventer(doc)
        _ = parser(doc)
        conllu_basic_out_formatted = cw.parse_spacy_doc(doc)
        
        sent, _ = converter.convert([conllu_basic_out_formatted], enhance_ud, enhanced_plus_plus, enhanced_extra, convs, remove_eud_info, remove_extra_info)
        
        assert len(sent) == 1
        sent = sent[0]
        _ = sent.pop(0)
        sent = cw.fix_sentence(sent)
        tokens = [node.get_conllu_field("form") for node in sent.values()]
        tags = [node.get_conllu_field("xpos") for node in sent.values()]
        lemmas = [node.get_conllu_field("lemma") for node in sent.values()]
        entities = fix_entities(sample_, len(tokens))
        chunks = ["O"] * len(tokens)
        
        g = nx.Graph()
        dg = nx.DiGraph()
        for node in sent.values():
            for parent, label in node.get_new_relations():
                if parent.get_conllu_field("id") == 0:
                    continue
                
                g.add_edge(parent.get_conllu_field("id") - 1, node.get_conllu_field("id") - 1, label=label)
                dg.add_edge(parent.get_conllu_field("id") - 1, node.get_conllu_field("id") - 1, label=label)
        
        odin = cw.conllu_to_odin([sent], get_odin_json(tokens, sample_, rel, tags, lemmas, entities, chunks, odin_id), False, True)
        
        ann_samples = []
        trigger_toks = search_triggers(sample_['subj_start'], sample_['subj_end'] + 1, sample_['obj_start'], sample_['obj_end'] + 1, rel, tokens) if use_triggers else [None]
        for trigger_tok in trigger_toks:
            ann_samples.append(AnnotatedSample(
                " ".join(tokens), " ".join(tokens), rel, sample_['subj_type'].title(), sample_['obj_type'].title(), tokens, tags, entities, chunks, lemmas,
                (sample_['subj_start'], sample_['subj_end'] + 1), (sample_['obj_start'], sample_['obj_end'] + 1), trigger_tok, g, dg))
        
        return ann_samples, odin


def store_pattern_stats(pattern_dict, name):
    with open("logs/pattern_stats_%s.json" % name, "w") as f:
        lens = []
        for x in range(1, 9):
            pattern_dict_x = get_percentage_strategy(pattern_dict, x * 0.1)
            lens.append(len(set(v3 for k,v in pattern_dict_x.items() for k2,v2 in v.items() for v3 in v2)))
        json.dump(lens, f)


def get_triggers(rel):
    try:
        with open("triggers/" + rel + ".xml", "r") as f:
            triggers = [l.strip() for l in f.readlines() if l.strip() != '']
        return triggers
    except FileNotFoundError:
        return []


def filter_misparsed_patterns(pattern_dict, str_rel, d):
    new_pattern_dict = dict()
    rel = Relation.fetch(id=str_rel)
    for p, samples in pattern_dict.items():
        try:
            _ = compile_to_odinson_rule(spike_compiler.from_text("\n".join([p]), rel)[0][0])
            new_pattern_dict[p] = samples
        except:
            d.append((str_rel, len(samples), p))
            continue
    return new_pattern_dict


def generate_patterns(data: List, enhance_ud: bool, enhanced_plus_plus: bool, enhanced_extra: bool, convs: int, remove_eud_info: bool, remove_extra_info: bool, use_triggers: bool):
    c = 0
    ann_samples = defaultdict(list)
    for i, sample in enumerate(data):
        if (i + 1) % 6800 == 0:
            print("finished %d/%d samples" % (i, len(data)))
        
        # store only relations that are subscribed under spike/server/resources/files_db/relations
        #   and notice to store the correct information (cammel case, correct name/label/id etc)
        rel = sample['relation'].replace('/', '_')
        if rel not in spike_relations:
            continue
        
        new_ann_samples, _ = SampleAryehAnnotator.annotate_sample(sample, rel, enhance_ud, enhanced_plus_plus, enhanced_extra, convs, remove_eud_info, remove_extra_info, use_triggers=use_triggers)
        _ = [ann_samples[ann_sample.relation].append(ann_sample) for ann_sample in new_ann_samples]
        c += 1
    
    pattern_dict_no_lemma = dict()
    pattern_dict_with_lemma = dict()
    for node_selector, pattern_dict in [(WordNodeSelector, pattern_dict_no_lemma), (LemmaNodeSelector, pattern_dict_with_lemma)]:
        total_d = 0
        errs = []
        for rel, ann_samples_per_rel in ann_samples.items():
            triggers = get_triggers(rel)
            pattern_generator_with_trigger = PatternGenerator([TriggerVarNodeSelector(triggers)], LabelEdgeSelector(), [])
            pattern_generator_no_trigger = PatternGenerator([], LabelEdgeSelector(), [node_selector()])
            pattern_dict_pre_filter, d = GenerationFromAnnotatedSamples.gen_pattern_dict(ann_samples_per_rel, pattern_generator_with_trigger, pattern_generator_no_trigger)
            pattern_dict[rel] = filter_misparsed_patterns(pattern_dict_pre_filter, rel, errs)
            total_d += sum(d.values())
        print("%d/%d patterns can’t be created for %s" % (total_d, sum([len(a) for a in ann_samples.values()]), str(node_selector)))
        for err in errs:
            print(err)
    return pattern_dict_no_lemma, pattern_dict_with_lemma


def get_link(in_port):
    host = os.environ.get("ODINSON_WRAPPER_HOST", "localhost")
    port = os.environ.get("ODINSON_WRAPPER_PORT", in_port)
    return f"http://{host}:{port}"


def eval_patterns_on_dataset(rel_to_pattern_dict, data_name, in_port, f, global_or_local):
    compute_global = True if global_or_local else False
    compute_per_pattern = False if global_or_local else True
    tot_retrieved_and_relevant = 0
    tot_relevant = 0
    tot_retrieved = 0
    stats_list = dict()
    macro_f = 0
    i = 0
    dev_tuned = dict()
    for str_rel, patterns in rel_to_pattern_dict.items():
        if str_rel not in spike_relations:
            continue
        rel = Relation.fetch(id=str_rel)
        e, labels = eval.evaluate_relation(FileBasedDataSet(data_name), spike_compiler.from_text("\n".join(patterns.keys()), rel)[0],
            rel, get_link(in_port), get_link(in_port + 90), compute_global=compute_global, compute_per_pattern=compute_per_pattern)
        dev_tuned[str_rel] = labels
        
        stats = e.get_global_stats()
        tot_retrieved_and_relevant += stats['retrievedAndRelevant']
        tot_retrieved += stats['retrieved']
        tot_relevant += stats['relevant']
        macro_f += stats['f1']
        stats_list[str_rel] = stats
        i += 1
        print("finished rel: %s %d/%d" % (str_rel, i, len(spike_relations)))
    
    # if labels:
    #     recalls = []
    #     tot_rnr = 0
    #     for rule in sorted([v2 for k,v in dev_tuned.items() for v2 in v], reverse=True):
    #         tot_rnr += rule
    #         recalls.append(tot_rnr)
    #     recalls = np.array(recalls) / tot_relevant
    #
    #     with open("recalls_by_port%d" % in_port, "w") as f:
    #         json.dump(recalls.tolist(), f)
    #     print("outputed successfully")
    #     exit()
    if compute_per_pattern:
        for str_rel, patterns in rel_to_pattern_dict.items():
            _ = [patterns.pop(pat) for pat in list(patterns.keys()) if pat not in dev_tuned[str_rel]]
    
    prec = (tot_retrieved_and_relevant / tot_retrieved) if tot_retrieved > 0 else 0
    recall = (tot_retrieved_and_relevant / tot_relevant) if tot_relevant > 0 else 0
    micro_f = ((2 * prec * recall) / (prec + recall)) if (prec + recall) > 0 else 0
    macro_f /= len(stats_list)
    json.dump([stats_list, prec, recall, micro_f, macro_f], f)
    return (prec, recall, micro_f, macro_f), stats_list


def get_percentage_strategy(pattern_dicts, percentage):
    percentage_strategy = defaultdict(defaultdict)
    
    for rel, pattern_dict in pattern_dicts.items():
        so_far = set()
        len_rel = len({vv for v in pattern_dict.values() for vv in v})
        for l, k, v in sorted([(len(v), k, v) for k, v in pattern_dict.items()], reverse=True):
            percentage_strategy[rel][k] = v
            so_far.union(v)
            if len(so_far) >= (len_rel * percentage):
                break
    
    return percentage_strategy


def get_threshold_strategy(pattern_dicts, threshold):
    threshold_strategy = defaultdict(defaultdict)
    
    for rel, pattern_dict in pattern_dicts.items():
        for k, v in pattern_dict.items():
            if len(v) >= threshold:
                threshold_strategy[rel][k] = v
    
    return threshold_strategy


def get_no_immediate_path_strategy(pattern_dicts, bla):
    no_immediate_path_strategy = defaultdict(defaultdict)
    
    for rel, pattern_dict in pattern_dicts.items():
        for k, v in pattern_dict.items():
            if ("lemma=" in k) or ("word=" in k):
                no_immediate_path_strategy[rel][k] = v
    
    return no_immediate_path_strategy


get_self_strategy = lambda x, y: x
strat_funcs = [get_self_strategy, get_threshold_strategy, get_percentage_strategy, get_no_immediate_path_strategy]


def main_annotate(strategies, dataset):
    print("started loading tacred %s" % dataset)
    with open("dataset/tacred/data/json/{}.json".format(dataset)) as f:
        data = json.load(f)
    print("finished loading tacred %s" % dataset)
    
    for i, (strat_name, enhance_ud, enhanced_plus_plus, enhanced_extra, convs, remove_eud_info, remove_extra_info) in enumerate(strategies):
        start = time.time()
        print("Started annotating %s/l" % strat_name)
        for j, sample in enumerate(data):
            filename = "resources/datasets/tacred-{}-labeled-aryeh-{}/ann/sent_{:0>5}.json".format(dataset, strat_name, j)
            filename_l = "resources/datasets/tacred-{}-labeled-aryeh-{}l/ann/sent_{:0>5}.json".format(dataset, strat_name, j)
            try:
                os.makedirs(os.path.dirname(filename))
                os.makedirs(os.path.dirname(filename_l))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
            _, odin_json = SampleAryehAnnotator.annotate_sample(
                sample, sample['relation'].replace('/', '_'), enhance_ud, enhanced_plus_plus, enhanced_extra, convs, remove_eud_info, remove_extra_info, odin_id=j)
            with open(filename, 'w') as f:
                json.dump(odin_json['documents'][''], f)
            with open(filename_l, 'w') as f:
                json.dump(odin_json['documents'][''], f)
        print("finished annotating %s/l, time:%.3f" % (strat_name, time.time() - start))


def main_eval(strats, data, use_lemma, in_port, sub_strategies, sub_infos, use_triggers):
    evals = dict()
    for i, (name, enhance_ud, enhanced_plus_plus, enhanced_extra, convs, remove_eud_info, remove_extra_info) in enumerate(strats):
        name = name + ('l' if use_lemma else '')
        trig_str = "" if use_triggers else "_no_trig"
        
        start = time.time()
        for sub_strat_idx, sub_info in zip(sub_strategies, sub_infos):
            sub_strat_idx = int(sub_strat_idx)
            sub_info = float(sub_info)
            use_tuned = True if (sub_strat_idx == DEV_TUNED_STRATEGY_INDEX) and (sub_info == DEV_TUNED_STRATEGY_INFO) and (data == 'test') else False
            tune = True if (sub_strat_idx == DEV_TUNED_STRATEGY_INDEX) and (sub_info == DEV_TUNED_STRATEGY_INFO) and (data == 'dev') else False
            
            dev_tuned_str = "_dev_tuned" if use_tuned else ""
            with open("pattern_dicts/pattern_dict_%s%s%s.pkl" % (name, trig_str, dev_tuned_str), "rb") as f:
                print("started loading patterns for strategy %s" % name)
                pattern_dict = pickle.load(f)
                print("finished loading patterns for strategy %s" % name)
            print("started calculating {}-set scores for strategy {}, sub{}_{}{}".format(data, name, sub_strat_idx, sub_info, trig_str))
            ff = open("logs/log_scores_{}_{}_sub{}_{}{}.json".format(data, name, sub_strat_idx, sub_info, trig_str), "w")
            cur_pattern_dict = strat_funcs[sub_strat_idx](pattern_dict, sub_info)
            try:
                evals[(name, data, sub_strat_idx, sub_info, trig_str)] = eval_patterns_on_dataset(cur_pattern_dict, "tacred-{}-labeled-aryeh-{}".format(data, name), in_port, ff, not tune)[0]
            except ConnectionError as e:
                raise e
            except Exception as e2:
                print(e2)
                ff.close()
                continue
            if tune:
                with open("pattern_dicts/pattern_dict_%s%s_dev_tuned.pkl" % (name, trig_str), "wb") as f:
                    pickle.dump(cur_pattern_dict, f)
            print("finished calculating %s-set scores for strategy %s, sub%d_%f%s time: %.3f" % (data, name, sub_strat_idx, sub_info, trig_str, time.time() - start))
            print("\tscores: " + str(evals[(name, data, sub_strat_idx, sub_info, trig_str)]))
            ff.close()
    print(str(evals))


def main_generate(strats, use_triggers):
    print("started loading tacred train")
    with open("dataset/tacred/data/json/train.json") as f:
        train = json.load(f)
    print("finished loading tacred train")
    
    for i, (name, enhance_ud, enhanced_plus_plus, enhanced_extra, convs, remove_eud_info, remove_extra_info) in enumerate(strats):
        print("started generating patterns for strategy %s" % name)
        start = time.time()
        pattern_dict_no_lemma, pattern_dict_with_lemma = \
            generate_patterns(train, enhance_ud, enhanced_plus_plus, enhanced_extra, convs, remove_eud_info, remove_extra_info, use_triggers)
        store_pattern_stats(pattern_dict_no_lemma, name)
        store_pattern_stats(pattern_dict_with_lemma, name)
        print("finished generating patterns for strategy %s/l, strategy-index: %d, time: %.3f" % (name, i, time.time() - start))
        with open("pattern_dicts/pattern_dict_%s%s.pkl" % (name, "" if use_triggers else "_no_trig"), "wb") as f:
            pickle.dump(pattern_dict_no_lemma, f)
        with open("pattern_dicts/pattern_dict_%sl%s.pkl" % (name, "" if use_triggers else "_no_trig"), "wb") as f:
            pickle.dump(pattern_dict_with_lemma, f)


if __name__ == "__main__":
    strategies = [
        ("n", False, False, False, 1, False, False),  # no enhancing
        ("e", True, True, False, 1, False, False),  # eUD
        ("e2", True, True, False, 1, True, True),  # eUD + no-extra-info
        ("a", True, True, True, 1, True, True),  # Aryeh enhancement + eUD
        ("ar", True, True, True, 2, True, True),  # Aryeh enhancement + eUD + 2 convs
        ("a2", False, False, True, 1, True, True),  # Aryeh enhancement + no eUD
        ("a2r", False, False, True, 2, True, True),  # Aryeh enhancement + no eUD + 2 convs
        ("a3", True, True, True, 2, False, True),
        ("a3r", True, True, True, 2, False, True)]

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-a', '--action', type=str, default='eval')
    arg_parser.add_argument('-d', '--data', type=str, default='dev')
    arg_parser.add_argument('-x', '--strat_start', type=int, default=-1)
    arg_parser.add_argument('-y', '--strat_end', type=int, default=-1)
    arg_parser.add_argument('-p', '--port', type=int, default=9000)
    arg_parser.add_argument('-s', '--sub_strats', action='append', default=None)
    arg_parser.add_argument('-i', '--sub_infos', action='append', default=None)
    arg_parser.add_argument('-l', '--use_lemma', type=int, default=1)
    arg_parser.add_argument('-t', '--use_triggers', type=int, default=1)

    args = arg_parser.parse_args()
    
    if args.strat_start >= 0:
        if args.strat_end >= 0:
            strategies = strategies[args.strat_start:args.strat_end]
        else:
            strategies = strategies[args.strat_start:args.strat_start + 1]
    
    if args.action == 'annotate':
        main_annotate(strategies, args.data)
    elif args.action == 'eval':
        main_eval(strategies, args.data, args.use_lemma, args.port, args.sub_strats, args.sub_infos, args.use_triggers)
    elif args.action == 'generate':
        main_generate(strategies, args.use_triggers)
