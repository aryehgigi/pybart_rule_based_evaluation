from typing import Tuple, List
import errno
import sys
import os
import networkx as nx
import json
import pickle
import time
import numpy as np
from collections import defaultdict
import traceback

from spike.datamodel.dataset import FileBasedDataSet
from spike.rest.definitions import Relation
from spike.pattern_generation.gen_pattern import PatternGenerator
from spike.pattern_generation.pattern_selectors import LabelEdgeSelector, WordNodeSelector, LemmaNodeSelector
from spike.pattern_generation.utils import GenerationFromAnnotatedSamples
from spike.pattern_generation.compilation import spike_compiler
from spike.pattern_generation.sample_types import AnnotatedSample
from spike.evaluation import eval

from ud2ude_aryehgigi import converter, conllu_wrapper as cw
import spacy
from spacy.tokens import Doc

THRESHOLD = 5
spike_relations = ["org:country_of_headquarters", "per:cause_of_death", "per:country_of_birth", "per:spouse", "org:founded", "per:chlidren", "per:country_of_death", "per:stateorprovince_of_death", "org:founded_by", "per:cities_of_residence", "per:origin", "org:alternate_names", "org:number_of_employees_members", "per:city_of_death", "per:religion", "org:city_of_headquarters", "per:age", "per:countries_of_residence", "per:schools_attended"]


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


def get_odin_json(tokens, sample_, tags, lemmas, entities, chunks, odin_id):
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
            "keep": True, "foundBy": "tacred_gold", "type": "RelationMention", "labels": [sample_["relation"]],
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


class SampleAryehAnnotator(object):
    @staticmethod
    def annotate_sample(sample_: dict, enhance_ud: bool, enhanced_plus_plus: bool, enhanced_extra: bool, convs: int,
                        remove_eud_info: bool, remove_extra_info: bool, odin_id: int = -1) -> Tuple[AnnotatedSample, dict]:
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
        
        odin = cw.conllu_to_odin([sent], get_odin_json(tokens, sample_, tags, lemmas, entities, chunks, odin_id), False, True)
        ann_sample = AnnotatedSample(
            " ".join(tokens), " ".join(tokens), sample_['relation'], sample_['subj_type'].title(), sample_['obj_type'].title(), tokens, tags, entities, chunks, lemmas,
            (sample_['subj_start'], sample_['subj_end'] + 1), (sample_['obj_start'], sample_['obj_end'] + 1), None, g, dg)
        
        return ann_sample, odin


def generate_patterns(data: List, enhance_ud: bool, enhanced_plus_plus: bool, enhanced_extra: bool, convs: int, remove_eud_info: bool, remove_extra_info: bool):
    ann_samples = defaultdict(list)
    for i, sample in enumerate(data):
        if (i + 1) % 6800 == 0:
            print("finished %d/%d samples" % (i, len(data)))

        # store only relations that are subscripted under spike/server/resources/files_db/relations
        #   and notice to store the correct information (cammel case, correct name/label/id etc)
        if sample['relation'] not in spike_relations:
            continue
        
        ann_sample, _ = SampleAryehAnnotator.annotate_sample(sample, enhance_ud, enhanced_plus_plus, enhanced_extra, convs, remove_eud_info, remove_extra_info)
        ann_samples[ann_sample.relation].append(ann_sample)
    
    pattern_dict_no_lemma = dict()
    pattern_dict_with_lemma = dict()
    pattern_generator_no_lemma = PatternGenerator([], LabelEdgeSelector(), [WordNodeSelector()])
    pattern_generator_with_lemma = PatternGenerator([], LabelEdgeSelector(), [LemmaNodeSelector()])
    for rel, ann_samples_per_rel in ann_samples.items():
        pattern_dict_no_lemma[rel], d = GenerationFromAnnotatedSamples.gen_pattern_dict(ann_samples_per_rel, pattern_generator_no_lemma)
        # for pattern in pattern_dict_no_lemma_rel.keys():
        #     str_rel = Relation.fetch(id=rel)
        #     try:
        #         _ = spike_compiler.from_text(pattern, str_rel)[0]
        #     except:
        #         pattern_dict_no_lemma_rel.pop(pattern)
        # pattern_dict_no_lemma[rel] = pattern_dict_no_lemma_rel
        print("%d patterns can’t be created for no-lemma" % sum(d.values()))
        pattern_dict_with_lemma[rel], d2 = GenerationFromAnnotatedSamples.gen_pattern_dict(ann_samples_per_rel, pattern_generator_with_lemma)
        # for pattern in pattern_dict_with_lemma_rel.keys():
        #     str_rel = Relation.fetch(id=rel)
        #     try:
        #         _ = spike_compiler.from_text(pattern, str_rel)[0]
        #     except:
        #         pattern_dict_no_lemma_rel.pop(pattern)
        # pattern_dict_with_lemma[rel] = pattern_dict_with_lemma_rel
        print("%d patterns can’t be created for with-lemma" % sum(d2.values()))
    
    return pattern_dict_no_lemma, pattern_dict_with_lemma


def get_link(in_port):
    host = os.environ.get("ODINSON_WRAPPER_HOST", "localhost")
    port = os.environ.get("ODINSON_WRAPPER_PORT", in_port)
    return f"http://{host}:{port}"


def eval_patterns_on_dataset(rel_to_pattern_dict, data_name, in_port, f, name, sub_strategy):
    tot_retrieved_and_relevant = 0
    tot_relevant = 0
    tot_retrieved = 0
    stats_list = dict()
    macro_f = 0
    for str_rel, patterns in rel_to_pattern_dict.items():
        if str_rel not in spike_relations:
            continue
        rel = Relation.fetch(id=str_rel)
        attempts = 0
        while attempts < 5:
            try:
                e = eval.evaluate_relation(FileBasedDataSet(data_name), spike_compiler.from_text("\n".join(patterns.keys()), rel)[0], rel, get_link(in_port), get_link(in_port + 90))
                break
            except ConnectionError:
                # os.system('setterm -background white -foreground black')
                # print("\n"*100)
                # import pdb;pdb.set_trace()
                # os.system('setterm -background black -foreground white')
                attempts += 1
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stdout)
                print("finished attempt %d" % attempts)
            except:
                # os.system('setterm -background white -foreground black')
                # print("\n"*100)
                # import pdb;pdb.set_trace()
                # os.system('setterm -background black -foreground white')
                new_patterns = {}
                for pattern, vvv in patterns.items():
                    new_pattern = pattern
                    if "nmod:vs." in pattern:
                        new_pattern = pattern.replace("nmod:vs.", "nmod:vs")
                    if ("nmod:@" in pattern):
                        new_pattern = pattern.replace("nmod:@", "nmod:at_sign")
                    elif "nmod:'s" in pattern:
                        new_pattern = pattern.replace("nmod:'s", "nmod:poss")
                    new_patterns[new_pattern] = vvv
                patterns = new_patterns
                attempts += 1
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stdout)
                print("finished attempt %d" % attempts)
        if attempts == 5:
            import pdb;pdb.set_trace()
         
        stats = e.get_global_stats()
        tot_retrieved_and_relevant += stats['retrievedAndRelevant']
        tot_retrieved += stats['retrieved']
        tot_relevant += stats['relevant']
        macro_f += stats['f1']
        f.write(name + str(sub_strategy) + str_rel + str(stats))
        stats_list[str_rel] = stats
        print("finished rel: %s" % str_rel)
    
    prec = (tot_retrieved_and_relevant / tot_retrieved) if tot_retrieved > 0 else 0
    recall = (tot_retrieved_and_relevant / tot_relevant) if tot_relevant > 0 else 0
    micro_f = ((2 * prec * recall) / (prec + recall)) if (prec + recall) > 0 else 0
    macro_f /= len(stats_list)
    return (prec, recall, micro_f, macro_f), stats_list


def get_percentage_strategy(pattern_dicts):
    percentage_strategy = defaultdict(defaultdict)
    
    for rel, pattern_dict in pattern_dicts.items():
        so_far = 0
        len_rel = sum(len(v) for v in pattern_dict.values())
        for l, k, v in sorted([(len(v), k, v) for k, v in pattern_dict.items()], reverse=True):
            percentage_strategy[rel][k] = v
            so_far += l
            if so_far >= (len_rel * 0.8):
                break
    
    return percentage_strategy


def get_threshold_strategy(pattern_dicts):
    threshold_strategy = defaultdict(defaultdict)
    
    for rel, pattern_dict in pattern_dicts.items():
        for k, v in pattern_dict.items():
            if len(v) >= THRESHOLD:
                threshold_strategy[rel][k] = v
    
    return threshold_strategy


get_self_stratagey = lambda x: x


def main_dev_test(strategies, name):
    print("started loading tacred %s" % name)
    with open("dataset/tacred/data/json/{}.json".format(name)) as f:
        data = json.load(f)
    print("finished loading tacred %s" % name)
    
    # annotate dev
    for i, (strat_name, enhance_ud, enhanced_plus_plus, enhanced_extra, convs, remove_eud_info, remove_extra_info,
            use_lemma) in enumerate(strategies):
        start = time.time()
        if not use_lemma:
            print("Started annotating %s/l" % strat_name)
            for j, sample in enumerate(data):
                filename = "resources/datasets/tacred-{}-labeled-aryeh-{}/ann/sent_{:0>5}.json".format(name, strat_name, j)
                filename_l = "resources/datasets/tacred-{}-labeled-aryeh-{}l/ann/sent_{:0>5}.json".format(name, strat_name, j)
                try:
                    os.makedirs(os.path.dirname(filename))
                    os.makedirs(os.path.dirname(filename_l))
                except OSError as exc:  # Guard against race condition
                    if exc.errno != errno.EEXIST:
                        raise
                _, odin_json = SampleAryehAnnotator.annotate_sample(
                    sample, enhance_ud, enhanced_plus_plus, enhanced_extra, convs, remove_eud_info, remove_extra_info, odin_id=j)
                with open(filename, 'w') as f:
                    json.dump(odin_json['documents'][''], f)
                with open(filename_l, 'w') as f:
                    json.dump(odin_json['documents'][''], f)
            print("finished annotating %s/l, time:%.3f" % (strat_name, time.time() - start))


def test_set_eval(name, sub_strategy, in_port):
    ff = open("log_scores_test_%s.pkl" % name, "w")
    print("started calculating test-set scores for strategy %s" % name)
    start = time.time()
    with open("pattern_dicts/pattern_dict_%s.pkl" % name, "rb") as f:
        pattern_dict = pickle.load(f)
    chosen_strategy = sub_strategy(pattern_dict)
    (p, r, micro_f, macro_f), _ = eval_patterns_on_dataset(chosen_strategy, "tacred-test-labeled-aryeh-{}".format(name), in_port, ff, name, sub_strategy)
    print("Final scores %s (time:%.3f):\tprec: %.3f\n\trecall: %.3f\n\tmicro_f1: %.3f\n\tmacro_f1: %.3f" % (name, time.time() - start, p, r, micro_f, macro_f))


def main_train(strats, in_port, no_eval, strat_chooser_start=None, strat_chooser_end=None):
    n_scores = dict()
    e_scores = dict()
    a_scores = dict()
    
    print("started loading tacred train")
    with open("dataset/tacred/data/json/train.json") as f:
        train = json.load(f)
    print("finished loading tacred train")
    
    if strat_chooser_start is not None:
        if strat_chooser_end is not None:
            strats = strats[strat_chooser_start:strat_chooser_end]
        else:
            strats = strats[strat_chooser_start:strat_chooser_start + 1]
    
    # annotate train and generate patterns
    for i, (name, enhance_ud, enhanced_plus_plus, enhanced_extra, convs, remove_eud_info, remove_extra_info, use_lemma) in enumerate(strats):
        ff = open("log_scores_%s.pkl" % name, "w")
        # generate patterns for each strategy
        try:
            with open("pattern_dicts/pattern_dict_%s.pkl" % name, "rb") as f:
                print("started loading patterns for strategy %s" % name)
                pattern_dict = pickle.load(f)
                print("finished loading patterns for strategy %s" % name)
        except FileNotFoundError:
            print("started generating patterns for strategy %s" % name)
            start = time.time()
            if not use_lemma:
                pattern_dict_no_lemma, pattern_dict_with_lemma = \
                    generate_patterns(train, enhance_ud, enhanced_plus_plus, enhanced_extra, convs, remove_eud_info, remove_extra_info)
                pattern_dict = pattern_dict_no_lemma
            else:
                pattern_dict = pattern_dict_with_lemma
            print("finished generating patterns for strategy %s, strategy-index: %d, time: %.3f" % (name, i, time.time() - start))
            with open("pattern_dicts/pattern_dict_%s.pkl" % name, "wb") as f:
                pickle.dump(pattern_dict, f)
        
        if no_eval:
            continue
        
        print("started calculating dev-set scores for strategy %s" % name)
        # split to sub strategies and test each against dev set.
        # store scores separately for baseline:'n'/baseline:'e'/new-awesome-SOTA:'a'
        scores = a_scores if name.startswith("a") else (n_scores if name.startswith("n") else e_scores)
        start = time.time()
        for sub_strategy in [get_threshold_strategy]:  # [get_self_stratagey, get_threshold_strategy, get_percentage_strategy]:
            cur_pattern_dict = sub_strategy(pattern_dict)
            print("working on: tacred-dev-labeled-aryeh-{}".format(name))
            scores[(name, sub_strategy)] = eval_patterns_on_dataset(cur_pattern_dict, "tacred-dev-labeled-aryeh-{}".format(name), in_port, ff, name, sub_strategy)
            print(scores[(name, sub_strategy)][0])
            import pdb;pdb.set_trace() # need to change the odinson thats up to fit port
        print("finished calculating dev-set scores for strategy %s, strategy-index: %d, time: %.3f" % (name, i, time.time() - start))
        ff.close()
    
    if no_eval:
        return
    
    # choose best strategies and evaluate them on test set
    chosen_strategies = (n_scores[np.argmax(list(zip(*n_scores.values()))[2])],
                         e_scores[np.argmax(list(zip(*e_scores.values()))[2])],
                         a_scores[np.argmax(list(zip(*a_scores.values()))[2])])
    for name, sub_strategy, port in chosen_strategies:
        test_set_eval(name, sub_strategy, port)


if __name__ == "__main__":
    strategies = [
        ("n", False, False, False, 1, False, False, False),  # no enhancing
        ("nl", False, False, False, 1, False, False, True),  # no enhancing + lemma
        ("e", True, True, False, 1, False, False, False),  # eUD
        ("el", True, True, False, 1, False, False, True),  # eUD + lemma
        ("a", True, True, True, 1, True, True, False),  # Aryeh enhancement + eUD
        ("al", True, True, True, 1, True, True, True),  # Aryeh enhancement + eUD + lemma
        ("a2", False, False, True, 1, True, True, False),  # Aryeh enhancement + no eUD
        ("a2l", False, False, True, 1, True, True, True),  # Aryeh enhancement + no eUD + lemma
        ("ar", True, True, True, 2, True, True, False),  # Aryeh enhancement + eUD + 2 convs
        ("arl", True, True, True, 2, True, True, True),  # Aryeh enhancement + eUD + 2 convs + lemma
        ("a2r", False, False, True, 2, True, True, False),  # Aryeh enhancement + no eUD + 2 convs
        ("a2rl", False, False, True, 2, True, True, True),  # Aryeh enhancement + no eUD + 2 convs + lemma
        ("e2", True, True, False, 1, True, True, False),  # eUD + no-extra-info
        ("e2l", True, True, False, 1, True, True, True),  # eUD + no-extra-info + lemma
        ("a3r", True, True, True, 2, False, True, False),
        ("a3rl", True, True, True, 2, False, True, True)]
    
    if sys.argv[1] == 'train':
        if len(sys.argv) == 2:
            main_dev_test(strategies, 'train')
        elif len(sys.argv) == 3:
            main_train(strategies, int(sys.argv[2]), False)
        elif len(sys.argv) == 4:
            if sys.argv[3] == 'no_eval':
                main_train(strategies, int(sys.argv[2]), True)
            else:
                main_train(strategies, int(sys.argv[2]), False, int(sys.argv[3]))
        elif len(sys.argv) == 5:
            main_train(strategies, int(sys.argv[2]), False, int(sys.argv[3]), int(sys.argv[4]))
    elif sys.argv[1] == 'dev':
        main_dev_test(strategies, 'dev')
    elif sys.argv[1] == 'test':
        main_dev_test(strategies, 'test')
    else:
        strat_funcs = [get_self_stratagey, get_threshold_strategy, get_percentage_strategy]
        test_set_eval(sys.argv[1], strat_funcs[int(sys.argv[2])], int(sys.argv[3]))

