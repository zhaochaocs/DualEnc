"""
Laura Perez

networkx documentation:
https://networkx.github.io/documentation/networkx-1.10/reference/classes.multidigraph.html#networkx.MultiDiGraph
"""
import copy
import json
import networkx as nx
import re
import sys
import getopt
import random
import _pickle as p

from collections import defaultdict
from benchmark_reader import Benchmark
from webnlg_baseline_input import delexicalisation2, select_files, relexicalise
#import webnlg_baseline_input as BASE_PREPROCESS


UNSEEN_CATEGORIES = ['Athlete', 'Artist', 'MeanOfTransportation', 'CelestialBody', 'Politician']
SEEN_CATEGORIES = ['Astronaut', 'Building', 'Monument', 'University', 'SportsTeam',
                   'WrittenWork', 'Food', 'ComicsCharacter', 'Airport', 'City']



if sys.version_info[0] < 3:
    raise Exception("Python 3 or a more recent version is required.")

def tokenize(str_mention):
    str_mention = str_mention.replace('_', ' ').replace('"', '').strip()
    return ' '.join(re.split('(\W)', str_mention))

delex_dict = {}
with open('../../webnlg_eval_scripts/delex_dict2.json', encoding='utf-8') as data_file:
    data = json.load(data_file)
    for key, values in data.items():
        for value in values:
            delex_dict[tokenize(value)] = key.upper()
with open('../../webnlg_eval_scripts/delex_dict.json', encoding='utf-8') as data_file:
    data = json.load(data_file)
    for key, values in data.items():
        for value in values:
            delex_dict[tokenize(value)] = key.upper()

def get_cpt(str_mention, default):
    str_mention_dense = "".join(str_mention.split())
    if re.match(r'^(")?\d{4}-\d{2}-\d{2}(")?$', str_mention_dense, flags=0):
        return "DATE"
    elif re.match(r"^[-+]?\d+$", str_mention_dense, flags=0):
        return "INT"
    elif re.match(r"^[-+]?\d*\.?\d+$", str_mention_dense, flags=0):
        return "FLOAT"
    elif str_mention in delex_dict and not delex_dict[str_mention] == "THING":
        return delex_dict[str_mention]
    else:
        return default.upper()

def camel_to_list(camel_format):
    # print(camel_format)
    # The camel name might contain blank space: 5th RunwayName
    # if "identifier" in camel_format.lower():
    #     print("hit")
    camel_format = camel_format.replace('_', ' ').replace('"', '').strip()
    if not camel_format == camel_format.lower() and not camel_format == camel_format.upper():
        underline_format = ''
        if isinstance(camel_format, str):
            for i, _s_ in enumerate(camel_format):
                if _s_.islower():
                    underline_format += _s_
                elif i+1 < len(camel_format) and camel_format[i+1].islower():
                    underline_format += ' ' + _s_.lower()
                else:
                    underline_format += _s_.lower()
        return [s for s in underline_format.split() if len(s)]
    elif " " in camel_format:
        return [s for s in camel_format.split() if len(s)]
    else:
        return [camel_format]


def buildGraph2(srgGraph):  #, uniqueRelation=False):

    DG = nx.MultiDiGraph()
    for t in srgGraph.split("< TSP >"):
        t = t.strip().split(" | ")  # triple list
        DG.add_edge(t[0],t[2], label=t[1]) # edge label is the property


    srcNodes = []   # s, p, o
    srcEdgesLabels = []
    srcEdgesNode1 = []
    srcEdgesNode2 = []

    for eTriple in DG.edges(data='label'):
        rel = "_".join([x.strip() for x in eTriple[2].split()]) #eTriple[2].replace(" ", "_")
        subj = "_".join([x.strip() for x in eTriple[0].split()]) #eTriple[0].replace(" ", "_")
        obj = "_".join([x.strip() for x in eTriple[1].split()]) #eTriple[1].replace(" ", "_")

        relIdx = -1
        if not subj in srcNodes:
            srcNodes.append(subj)
        #if (uniqueRelation and not rel in srcNodes) or not uniqueRelation:
        srcNodes.append(rel)
        relIdx = len(srcNodes) - 1
        if not obj in srcNodes:
            srcNodes.append(obj)

        #srcEdges.append("|".join(["A0", str(srcNodes.index(subj)), str(srcNodes.index(rel))]))
        srcEdgesLabels.append("A0")
        srcEdgesNode1.append(str(srcNodes.index(subj)))
        srcEdgesNode2.append(str(relIdx))
        #srcEdges.append("|".join(["A1", str(srcNodes.index(obj)), str(srcNodes.index(rel))]))
        srcEdgesLabels.append("A1")
        srcEdgesNode1.append(str(srcNodes.index(obj)))
        srcEdgesNode2.append(str(relIdx))

    return " ".join(srcNodes), (" ".join(srcEdgesLabels), " ".join(srcEdgesNode1), " ".join(srcEdgesNode2))


def buildGraph(srgGraph):  #, uniqueRelation=False):

    DG = nx.MultiDiGraph()
    for t in srgGraph.split("< TSP >"):
        t = t.strip().split(" | ")  # triple list
        DG.add_edge(t[0],t[2], label=t[1]) # edge label is the property


    srcNodes = []   # s, p, o
    srcEdgesLabels = []
    srcEdgesNode1 = []
    srcEdgesNode2 = []

    for eTriple in DG.edges(data='label'):
        rel = "_".join([x.strip() for x in eTriple[2].split()]) #eTriple[2].replace(" ", "_")
        subj = "_".join([x.strip() for x in eTriple[0].split()]) #eTriple[0].replace(" ", "_")
        obj = "_".join([x.strip() for x in eTriple[1].split()]) #eTriple[1].replace(" ", "_")

        relIdx = -1
        if not subj in srcNodes:
            srcNodes.append(subj)
        #if (uniqueRelation and not rel in srcNodes) or not uniqueRelation:
        srcNodes.append(rel)
        relIdx = len(srcNodes) - 1
        if not obj in srcNodes:
            srcNodes.append(obj)

        #srcEdges.append("|".join(["A0", str(srcNodes.index(subj)), str(srcNodes.index(rel))]))
        srcEdgesLabels.append("A0")
        srcEdgesNode1.append(str(srcNodes.index(subj)))
        srcEdgesNode2.append(str(relIdx))
        #srcEdges.append("|".join(["A1", str(srcNodes.index(obj)), str(srcNodes.index(rel))]))
        srcEdgesLabels.append("A1")
        srcEdgesNode1.append(str(srcNodes.index(obj)))
        srcEdgesNode2.append(str(relIdx))

    return " ".join(srcNodes), (" ".join(srcEdgesLabels), " ".join(srcEdgesNode1), " ".join(srcEdgesNode2))



def buildGraphWithNE2(srgGraph, lowercase, is_sep=True):  #, uniqueRelation=False):

    DG = []
    triples = srgGraph.split("< TSP >")

    for t in triples:
        if lowercase:
            t = t.lower()
        t = t.strip().split(" | ")
        assert len(t) == 3
        t = [ti.strip() for ti in t]
        DG.append((t[0], t[1].lower(), t[2])) # edge label is the property, is "_" joined list

    srcNodes = []
    srcNodes_rnn = []
    srcEdgesLabels = []
    srcEdgesNode1 = []
    srcEdgesNode2 = []

    nodes_start_idx = {}

    for eTriple_idx, eTriple in enumerate(DG):
        subj = [x.strip() for x in eTriple[0].split()]
        rel = [x.strip() for x in eTriple[1].split()]
        obj = [x.strip() for x in eTriple[2].split()]
        srcNodes_rnn += ['S|'] + subj + ['P|'] + rel + ['O|'] + obj

        subj_ne, obj_ne = True, True

        # generate srcNodes
        for i, (node_mention, node_mention_list) in enumerate(zip([eTriple[0], eTriple[1], eTriple[2]], [subj, rel, obj])):
            node_mention_ = "_".join(node_mention.split())
            if i == 1:
                node_mention_ = "|".join(["_".join(e.split()) for e in eTriple])
                nodes_start_idx[node_mention] = len(srcNodes)
                srcNodes += [node_mention_]
                srcNodes += node_mention_list if is_sep else "_".join(node_mention.split())
            else:
                if not node_mention in nodes_start_idx:
                    nodes_start_idx[node_mention] = len(srcNodes)
                    srcNodes += [node_mention_]
                    srcNodes += node_mention_list if is_sep else "_".join(node_mention.split())
                else:
                    if i == 0:
                        subj_ne = False
                    if i == 2:
                        obj_ne = False

        # add A0 and A1 edges
        relIdx = nodes_start_idx[eTriple[1]]
        subjIdx, objIdx = nodes_start_idx[eTriple[0]], nodes_start_idx[eTriple[2]]
        srcEdgesLabels.append("A0")
        srcEdgesNode1.append(str(subjIdx))
        srcEdgesNode2.append(str(relIdx))
        srcEdgesLabels.append("A1")
        srcEdgesNode1.append(str(objIdx))
        srcEdgesNode2.append(str(relIdx))

        # add NE edges
        if subj and subj_ne:
            for i, neNode in enumerate(subj):
                nodeIdx = subjIdx + i + 1
                srcEdgesLabels.append("NE")
                srcEdgesNode1.append(str(nodeIdx))
                srcEdgesNode2.append(str(subjIdx))

        if rel:
            for i, neNode in enumerate(rel):
                nodeIdx = relIdx + i + 1
                srcEdgesLabels.append("NE")
                srcEdgesNode1.append(str(nodeIdx))
                srcEdgesNode2.append(str(relIdx))

        if obj and obj_ne:
            for i, neNode in enumerate(obj):
                nodeIdx = objIdx + i + 1
                srcEdgesLabels.append("NE")
                srcEdgesNode1.append(str(nodeIdx))
                srcEdgesNode2.append(str(objIdx))

    return " ".join(srcNodes), \
           (" ".join(srcEdgesLabels), " ".join(srcEdgesNode1), " ".join(srcEdgesNode2)), \
           srcNodes_rnn


def create_source_target(b, options, dataset, delex=True, relex=False, doCategory=[],
                         negraph=False, lowercased=True, is_train=True):
    """
    Write target and source files, and reference files for BLEU.
    :param b: instance of Benchmark class
    :param options: string "delex" or "delex2" or "notdelex" to label files
    :param dataset: dataset part: train, dev, test
    :param delex: boolean; perform delexicalisation or not
    TODO:update parapms
    :return: if delex True, return list of replacement dictionaries for each example
    """
    lex_ids = []
    source_out = []
    source_out_rnn = []
    source_out_orders = []
    source_nodes_out = []
    source_edges_out_labels = []
    source_edges_out_node1 = []
    source_edges_out_node2 = []
    source_out_split = []

    target_out = []
    rplc_list = []  # store the dict of replacements for each example
    for entr in b.entries:
        lexics = entr.lexs
        category = entr.category
        if doCategory and not category in doCategory:
        #if not category in UNSEEN_CATEGORIES:
            continue
        for lex in lexics:
            entry_tripleset = copy.deepcopy(entr.modifiedtripleset)
            lex_ordered_tripleset = copy.deepcopy(lex.orderedtripleset)
            # During eval, we do not distinguish the individual lexes
            lex_id = "{}_{}_{}_{}".format(entr.id, entr.size, category, lex.id)
            if not lex_ordered_tripleset.size:
                if is_train:
                    continue
                # print("No ordered tripleset found in {}! Replaced with modified tripleset".format(lex_id))
                lex_ordered_tripleset = entry_tripleset

            elif is_train and not lex_ordered_tripleset.size == entry_tripleset.size:   # replace tripleset as the incomplete version
                entry_tripleset = copy.deepcopy(lex.orderedtripleset)

            entry_tripleset.shuffle_triples()
            golden_order = entry_tripleset.get_order(lex_ordered_tripleset)
            assert len(golden_order) and len(golden_order) <= entry_tripleset.size
            source_out_orders.append(" ".join(golden_order))

            lex_ids.append(lex_id)
            triples = ''
            properties_objects = {}
            tripleSep = ""
            for i, triple in enumerate(entry_tripleset.triples):
                if is_train and str(i) not in golden_order:
                    continue
                triples += tripleSep + triple.s + '|' + "_".join(camel_to_list(triple.p)) + '|' + triple.o + ' '
                tripleSep = "<TSP>"
                properties_objects[triple.p] = triple.o
            triples = triples.replace('_', ' ').replace('"', '')
            # separate punct signs from text
            out_src = ' '.join(re.split('(\W)', triples))   # 'Aarhus   Airport | city   served | Aarhus ,    Denmark   '
            out_src = ' '.join(out_src.split())
            out_trg = ' '.join(re.split('(\W)', lex.lex))   # 'The   Aarhus   is   the   airport   of   Aarhus ,    Denmark . '
            out_trg = ' '.join(out_trg.split())
            if delex:
                out_src, out_trg, rplc_dict = delexicalisation2(out_src, entr, lex)
                out_trg = ' '.join(re.split('([^\w\-])', out_trg))
                out_trg = ' '.join(out_trg.split())
                rplc_list.append(rplc_dict)
            if negraph:
                source_nodes, source_edges, source_nodes_rnn = \
                    buildGraphWithNE2(out_src, lowercase=True if lowercased else False)
                source_out_rnn.append(' '.join(source_nodes_rnn))
            else:
                source_nodes, source_edges = buildGraph(out_src)

            source_split = []
            for set_len in lex.tripleset_split:
                set_split = [0 for _ in range(set_len)]
                set_split[-1] = 1
                source_split += set_split

            source_nodes_out.append(source_nodes)
            source_edges_out_labels.append(source_edges[0])
            source_edges_out_node1.append(source_edges[1])
            source_edges_out_node2.append(source_edges[2])
            source_out.append(' '.join(out_src.split()))
            target_out.append(' '.join(out_trg.split()))
            source_out_split.append(' '.join(map(str, source_split)))

    if relex:
        return rplc_list

    #TODO: we could add a '-src-features.txt' if we want to attach features to nodes
    if not relex:
        #we do not need to re-generate GCN input files when doing relexicalisation.. check this works ok
        with open(dataset + '-webnlg-' + options + '-src-nodes.txt', 'w+', encoding='utf8') as f:
            f.write('\n'.join(source_nodes_out))
        with open(dataset + '-webnlg-' + options + '-src-labels.txt', 'w+', encoding='utf8') as f:
            f.write('\n'.join(source_edges_out_labels))
        with open(dataset + '-webnlg-' + options + '-src-node1.txt', 'w+', encoding='utf8') as f:
            f.write('\n'.join(source_edges_out_node1))
        with open(dataset + '-webnlg-' + options + '-src-node2.txt', 'w+', encoding='utf8') as f:
            f.write('\n'.join(source_edges_out_node2))
        with open(dataset + '-webnlg-' + options + '-src-rel-order.txt', 'w+', encoding='utf8') as f:
            f.write('\n'.join(source_out_orders))
        with open(dataset + '-webnlg-' + options + '-src-split.txt', 'w+', encoding='utf8') as f:
            f.write('\n'.join(source_out_split))
        with open(dataset + '-webnlg-' + options + '-tgt.txt', 'w+', encoding='utf8') as f:
            f.write('\n'.join(target_out).lower() if lowercased else '\n'.join(target_out))


    with open(dataset + '-webnlg-' + options + '.triple', 'w+', encoding='utf8') as f:
        f.write('\n'.join(source_out).lower() if lowercased else '\n'.join(source_out))
    with open(dataset + '-webnlg-' + options + '-translate.lexid.txt', 'w+', encoding='utf8') as f:
        f.write('\n'.join(lex_ids))
    with open(dataset + '-webnlg-' + options + '-src.txt', 'w+', encoding='utf8') as f:
        if not source_out_rnn:
            f.write('\n'.join(source_out).replace(" < TSP > ", " ").replace(" | ", " ").lower() if lowercased \
                else '\n'.join(source_out).replace(" < TSP > ", " ").replace(" | ", " "))
        else:
            f.write('\n'.join(source_out_rnn))
    with open(dataset + '-webnlg-' + options + '.lex', 'w+', encoding='utf8') as f:
        f.write('\n'.join(target_out).lower() if lowercased  else '\n'.join(target_out))

    # create separate files with references for multi-bleu.pl for dev set
    scr_refs = defaultdict(list)
    if (dataset == 'dev' or dataset.startswith('test')) and not delex:
        ##TODO: I think that taking only the nodes part is enough for BLEU scripts, see if we really nead the whole graph here in the src part
        assert len(lex_ids) == len(source_out) == len(target_out)
        for lex_id, src, trg in zip(lex_ids, source_out, target_out):
            entry_id = lex_id.rsplit("_", 1)[0]
            scr_refs[entry_id].append(trg)
        # length of the value with max elements
        max_refs = sorted(scr_refs.values(), key=len)[-1]
        # keys = [key for (key, value) in sorted(scr_refs.items())]
        # values = [value for (key, value) in sorted(scr_refs.items())]
        keys = [key for (key, value) in sorted(scr_refs.items(), key=lambda kv: int(kv[0].split("_")[0][2:]))]
        values = [scr_refs[key] for key in keys]

        # write the source file not delex
        with open(dataset + "-" + options + '-eval.lexid.txt', 'w+', encoding='utf8') as f:
            f.write('\n'.join(keys))
        # write references files
        for j in range(0, len(max_refs)):
            with open(dataset + "-" + options + '-reference' + str(j) + '.lex', 'w+', encoding='utf8') as f:
                out = ''
                for ref in values:
                    try:
                        out += ref[j].lower()  + '\n' if lowercased else ref[j] + '\n'
                    except:
                        out += '\n'
                f.write(out)
                f.close()
        # write references files for ter metric

        with open(dataset + "-" + options + '-reference-ter.lex', 'w+', encoding='utf8') as f:
            out = ''
            for key, ref in zip(keys, values):
                for i in range(3):
                    try:
                        out += ref[i].lower() + "  ({})\n".format(key) if lowercased \
                            else ref[i] + "  ({})\n".format(key)
                    except:
                        out += '\n'
            f.write(out)
        # write references files for meteor metric

        with open(dataset + "-" + options + '-reference-meteor.lex', 'w+', encoding='utf8') as f:
            out = ''
            for key, ref in zip(keys, values):
                for i in range(3):
                    try:
                        out += ref[i].lower() + '\n' if lowercased \
                            else ref[i] + '\n'
                    except:
                        out += '\n'
            f.write(out)


        #write reference files for E2E evaluation metrics
        with open(dataset + "-" + options + '-conc.txt', 'w+', encoding='utf8') as f:
            for ref in values:
                for j in range(len(ref)):
                    f.write( ref[j].lower()  + '\n' if lowercased else ref[j] + '\n')
                f.write("\n")
            f.close()

    return rplc_list

def input_files(path, filepath=None, relex=False, parts=['train', 'dev'],
                doCategory=[],
                options = None,
                negraph=True,
                lowercased=True,
                fileid=None):
    """
    Read the corpus, write train and dev files.
    :param path: directory with the WebNLG benchmark
    :param filepath: path to the prediction file with sentences (for relexicalisation)
    :param relex: boolean; do relexicalisation or not
    :param parts: partition to process
    :param negraph: whether to add edges for multi-word entitites
    :param lowercased: whether to do all lowercased for the notdelex version of the files
    :return:
    """

    rplc_list_relex = None
    # options = ['all-delex', 'all-notdelex']  # generate files with/without delexicalisation
    if options is None:
        options = ['all-delex', 'all-notdelex']  # generate files with/without delexicalisation
    else:
        assert isinstance(options, list)
        assert len(set(options) - {'all-notdelex', 'all-delex'}) == 0

    for part in parts:
        for option in options:
            files = select_files(path + part, size=(1, 8))
            b = Benchmark()
            b.fill_benchmark(files)
            if option == 'all-delex':
                rplc_list = create_source_target(
                    b, option, part, delex=True, relex=relex, doCategory=doCategory, negraph=negraph,
                    lowercased=lowercased, is_train=(part=="train"))
                print('Total of {} instances processed in {} with {} mode'.format(len(rplc_list), part, option))
            elif option == 'all-notdelex':
                rplc_list = create_source_target(
                    b, option, part, delex=False, relex=relex, doCategory=doCategory, negraph=negraph,
                    lowercased=lowercased, is_train=(part=="train"))
                print('Total of {} instances processed in {} with {} mode'.format(len(rplc_list), part, option))
            if option == 'all-delex':
                rplc_list_relex = rplc_list

    if relex and rplc_list_relex:
        relexicalise(filepath, rplc_list_relex, fileid, part, lowercased=lowercased, doCategory=doCategory)
    print('Files necessary for training/evaluating are written on disc.')


def main(argv):
    usage = 'usage:\npython3 webnlg_gcnonmt_input.py -i <data-directory> [-p PARTITION] [-c CATEGORIES] [-e NEGRAPH]' \
           '\ndata-directory is the directory where you unzipped the archive with data'\
           '\nPARTITION which partition to process, by default test/devel will be done.'\
           '\n-c is seen or unseen if we want to filter the test seen per category.' \
           '\n-l generate all source/target files in lowercase.'
    try:
        opts, args = getopt.getopt(argv, 'i:p:c:el', ['inputdir=','partition=', 'category=', 'negraph=', 'lowercased='])
    except getopt.GetoptError:
        print(usage)
        sys.exit(2)
    input_data = False
    ngraph = False
    partition = None
    category = None
    lowercased = False
    for opt, arg in opts:
        if opt in ('-i', '--inputdir'):
            inputdir = arg
            input_data = True
        elif opt in ('-p', '--partition'):
            partition = arg
        elif opt in ('-c', '--category'):
            category = arg
        elif opt in ('-e', '--negraph'):
            ngraph = True
        elif opt in ('-l', '--lowercased'):
            lowercased = True
        else:
            print(usage)
            sys.exit()
    if not input_data:
        print(usage)
        sys.exit(2)
    print('Input directory is {}, NE={}, lowercased={}'.format(inputdir, ngraph, lowercased))
    if partition:
        if category=='seen':
            input_files(inputdir, parts=[partition], doCategory=SEEN_CATEGORIES,
                        negraph=ngraph, lowercased=lowercased)
        else:
            input_files(inputdir, parts=[partition], negraph=ngraph, lowercased=lowercased)
    else:
        input_files(inputdir, parts=['train', 'dev'], negraph=ngraph, lowercased=lowercased)


if __name__ == "__main__":
    main(sys.argv[1:])

