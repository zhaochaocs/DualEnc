import os
import random
import re
import json
import string
import sys
import getopt
from operator import itemgetter
from collections import defaultdict
from benchmark_reader import Benchmark, normalize
import random

#TODO: I comment out the corpus shuffling here, BUT need to refactor better for gcn

pronoun_set = set()
with open('../meta/pronoun.txt', 'r', encoding='utf-8') as fr:
    for line in fr:
        line = line.strip()
        if not len(line):
            continue
        pronoun_set.add(line)


def select_files(topdir, category='', size=(1, 8)):
    """
    Collect all xml files from a benchmark directory.
    :param topdir: directory with benchmark
    :param category: specify DBPedia category to retrieve texts for a specific category (default: retrieve all)
    :param size: specify size to retrieve texts of specific size (default: retrieve all)
    :return: list of tuples (full path, filename)
    """
    if size==0:
        finaldirs = [topdir]
    else:
        finaldirs = [topdir+'/'+str(item)+'triples' for item in range(size[0], size[1])]

    finalfiles = []
    for item in finaldirs:
        finalfiles += [(item, filename) for filename in os.listdir(item) if filename.endswith('xml')]
    if category:
        finalfiles = []
        for item in finaldirs:
            finalfiles += [(item, filename) for filename in os.listdir(item) if category in filename and filename.endswith('xml')]
    # return finalfiles
    return sorted(finalfiles, key=itemgetter(0, 1))
    #return sorted(finalfiles,key=itemgetter(1)) TODO: uncoment this after getting BLEU scores




def is_similar(str1, str2):

    def lcs(X, Y):
        # find the length of the strings
        m = len(X)
        n = len(Y)

        # declaring the array for storing the dp values
        L = [[None] * (n + 1) for i in range(m + 1)]

        for i in range(m + 1):
            for j in range(n + 1):
                if i == 0 or j == 0:
                    L[i][j] = 0
                elif X[i - 1] == Y[j - 1]:
                    L[i][j] = L[i - 1][j - 1] + 1
                else:
                    L[i][j] = max(L[i - 1][j], L[i][j - 1])

                    # L[m][n] contains the length of LCS of X[0..n-1] & Y[0..m-1]
        return L[m][n]
        # end of function lcs

    str2 = normalize(str2)

    if str2 in pronoun_set:
        return False
    return True



def delexicalisation2(out_src, entry, lex):
    """
    Perform delexicalisation based on the v1.5 file.
    :param out_src: source string
    :param out_trg: target string
    :param category: DBPedia category
    :param properties_objects: dictionary mapping properties to objects
    :return: delexicalised strings of the source and target; dictionary containing mappings of the replacements made
    """

    triples = [[e.strip() for e in triple.split('|')] for triple in out_src.split('< TSP >')]
    for i in range(len(triples)):
        triples[i][0] = entry.entity_agent_map[normalize(triples[i][0])]
        triples[i][2] = entry.entity_agent_map[normalize(triples[i][2])]
    delex_triples = [" | ".join(triple) for triple in triples]
    delex_triples = ' < TSP > '.join(delex_triples)

    delex_trg = lex.template
    if not delex_trg:
        return delex_triples, lex.lex, entry.agent_entity_map_relex
    pattern = r'(?:AGENT|PATIENT|BRIDGE)-\d'
    groups = list(re.finditer(pattern, delex_trg))
    if not len(groups):
        return delex_triples, delex_trg, entry.agent_entity_map_relex
    if not len(groups) == len(lex.refs):
        return delex_triples, delex_trg, entry.agent_entity_map_relex
    new_delex_trg = ''
    new_delex_trg += delex_trg[0:groups[0].span()[0]]
    for idx, (agent, ref) in enumerate(zip(groups, lex.refs)):
        assert agent.group() in ref['tag']
        if is_similar(ref['entity'], ref['text']):
            new_delex_trg += agent.group()
        else:
            new_delex_trg += ref['text']
        if idx <= len(groups) - 2:
            new_delex_trg += delex_trg[groups[idx].span()[1] : groups[idx+1].span()[0]]
        else:
            new_delex_trg += delex_trg[groups[-1].span()[1]:]

    return delex_triples, new_delex_trg, entry.agent_entity_map_relex


def create_source_target(b, options, dataset, delex=True):
    """
    Write target and source files, and reference files for BLEU.
    :param b: instance of Benchmark class
    :param options: string "delex" or "notdelex" to label files
    :param dataset: dataset part: train, dev, test
    :param delex: boolean; perform delexicalisation or not
    :return: if delex True, return list of replacement dictionaries for each example
    """
    source_out = []
    target_out = []
    rplc_list = []  # store the dict of replacements for each example
    for entr in b.entries:
        tripleset = entr.modifiedtripleset
        # triples = ''
        triples = []
        properties_objects = {}
        for triple in tripleset.triples:
            # triples += triple.s + ' ' + triple.p + ' ' + triple.o + ' '
            triples.append(triple.s + ' ' + triple.p + ' ' + triple.o + ' ')
            properties_objects[triple.p] = triple.o
        #random.shuffle(triples)

        #print(triples)
        triples.reverse()
        #print(triples)

        triples = ' '.join(triples)
        triples = triples.replace('_', ' ').replace('"', '')
        lexics = entr.lexs
        category = entr.category
        out_src = ' '.join(re.split('(\W)', triples))
        for lex in lexics:
            # separate punct signs from text
            out_trg = ' '.join(re.split('(\W)', lex.lex))
            if delex:
                out_src, out_trg, rplc_dict = delexicalisation(out_src, out_trg, category, properties_objects)
                rplc_list.append(rplc_dict)
            # delete white spaces
            source_out.append(' '.join(out_src.split()))
            target_out.append(' '.join(out_trg.split()))

    # shuffle two lists in the same way
    random.seed(10)
    if delex:
        corpus = list(zip(source_out, target_out, rplc_list))
        random.shuffle(corpus)
        source_out, target_out, rplc_list = zip(*corpus)
    else:
        corpus = list(zip(source_out, target_out))
        random.shuffle(corpus)
        source_out, target_out = zip(*corpus)

    with open(dataset + '-webnlg-' + options + '.triple', 'w+', encoding='utf8') as f:
        f.write('\n'.join(source_out))
    with open(dataset + '-webnlg-' + options + '.lex', 'w+', encoding='utf8') as f:
        f.write('\n'.join(target_out))

    # create separate files with references for multi-bleu.pl for dev set
    scr_refs = defaultdict(list)
    if dataset == 'dev' and not delex:
        for src, trg in zip(source_out, target_out):
            scr_refs[src].append(trg)

        # length of the value with max elements
        max_refs = sorted(scr_refs.values(), key=len)[-1]
        keys = [key for (key, value) in sorted(scr_refs.items())]
        values = [value for (key, value) in sorted(scr_refs.items())]
        # write the source file not delex
        with open(options + '-source.triple', 'w+', encoding='utf8') as f:
            f.write('\n'.join(keys))
        # write references files
        for j in range(0, len(max_refs)):
            with open(options + '-reference' + str(j) + '.lex', 'w+', encoding='utf8') as f:
                out = ''
                for ref in values:
                    try:
                        out += ref[j] + '\n'
                    except:
                        out += '\n'
                f.write(out)

    return rplc_list

def realize_date(date):
    year, month, day = date

    month = int(month)
    if month == 1:
        month = 'January'
    elif month == 2:
        month = 'February'
    elif month == 3:
        month = 'March'
    elif month == 4:
        month = 'April'
    elif month == 5:
        month = 'May'
    elif month == 6:
        month = 'June'
    elif month == 7:
        month = 'July'
    elif month == 8:
        month = 'August'
    elif month == 9:
        month = 'September'
    elif month == 10:
        month = 'October'
    elif month == 11:
        month = 'November'
    elif month == 12:
        month = 'December'

    return '{0} {1} , {2}'.format(month, str(int(day)), str(int(year)))


def realize_dates(sentence):
    regex='([0-9]{4})\s-\s([0-9]{2})\s-\s([0-9]{2})'
    dates = re.findall(regex, sentence)
    if len(dates) > 0:
        for date in dates:
            date_realization = realize_date(date)
            sentence = sentence.replace(' - '.join(date), date_realization)
    return sentence


def relexicalise(predfile, rplc_list, fileid, part='dev', lowercased=True, doCategory=None):
    """
    Take a file from seq2seq output and write a relexicalised version of it.
    :param rplc_list: list of dictionaries of replacements for each example (UPPER:not delex item)
    :return: list of predicted sentences
    """
    relex_predictions = []

    with open(part + '-webnlg-all-notdelex-translate.lexid.txt', 'r') as f:
        dev_lex_ids = [line.strip() for line in f]
    dev_lex_ids = [lex_id.rsplit("_", 1)[0] for lex_id in dev_lex_ids]
    dev_lex_ids = [(i, lex_id) for i, lex_id in enumerate(dev_lex_ids)
                      if lex_id.rsplit("_", 1)[1] in doCategory or not doCategory]
    category_idxes, dev_lex_ids = zip(*dev_lex_ids)
    category_idxes = set(category_idxes)

    with open(predfile, 'r') as f:
        predictions = [line for i, line in enumerate(f) if i in category_idxes]
    if rplc_list:
        for i, pred in enumerate(predictions):
            # replace each item in the corresponding example
            rplc_dict = rplc_list[i]
            relex_pred = pred

            for key in sorted(rplc_dict):
                relex_pred = relex_pred.replace(key + ' ', rplc_dict[key] + ' ')

            relex_pred = realize_dates(relex_pred)
            relex_predictions.append(relex_pred)
    else:
        relex_predictions = predictions

    with open(predfile.replace('delex', 'relex'), 'w', encoding='utf-8') as f:
        relexoutput = ''.join(relex_predictions)
        if lowercased:
            relexoutput = relexoutput.lower()
        f.write(relexoutput)

    # create a mapping between not delex triples and relexicalised sents

    src_gens = {}
    for src, gen in zip(dev_lex_ids, relex_predictions):
        src_gens[src] = gen  # need only one lex, because they are the same for a given triple

    # write generated sents to a file in the same order as triples are written in the source file
    with open(part+'-all-notdelex-eval.lexid.txt', 'r') as f:
        eval_lex_ids = [line.strip() for line in f if line.strip().split('_')[-1] in doCategory or not doCategory]
    # outfileName =   'relexicalised_predictions.txt'
    # if fileid:
    #     outfileName = 'relexicalised_predictions'+str(fileid)+'.txt'
    # with open(predfile.replace('delex', 'relex'), 'w+', encoding='utf8') as f:
    with open(predfile[:-4]+'-relex.txt', 'w+', encoding='utf8') as f, \
        open(predfile[:-4] + '-relex-ter.txt', 'w+', encoding='utf8') as f2:
        for lex_id in eval_lex_ids:
            relexoutput = src_gens.get(lex_id, "\n")
            if lowercased:
                relexoutput = relexoutput.lower()
            relexoutput = relexoutput.replace(" ##", "")
            f.write(relexoutput)
            f2.write(relexoutput[:-1] + "  ({})\n".format(lex_id))

    return relex_predictions



def input_files(path, filepath=None, relex=False):
    """
    Read the corpus, write train and dev files.
    :param path: directory with the WebNLG benchmark
    :param filepath: path to the prediction file with sentences (for relexicalisation)
    :param relex: boolean; do relexicalisation or not
    :return:
    """
    rplc_list_dev_delex = None
    parts = ['train', 'dev']
    options = ['all-delex', 'all-notdelex']  # generate files with/without delexicalisation
    for part in parts:
        for option in options:
            files = select_files(path + part, size=(1, 8))
            b = Benchmark()
            b.fill_benchmark(files)
            if option == 'all-delex':
                rplc_list = create_source_target(b, option, part, delex=True)
                print('Total of {} files processed in {} with {} mode'.format(len(files), part, option))
            elif option == 'all-notdelex':
                rplc_list = create_source_target(b, option, part, delex=False)
                print('Total of {} files processed in {} with {} mode'.format(len(files), part, option))
            if part == 'dev' and option == 'all-delex':
                rplc_list_dev_delex =rplc_list

    if relex and rplc_list_dev_delex:
        relexicalise(filepath, rplc_list_dev_delex)
    print('Files necessary for training/evaluating are written on disc.')


def main(argv):
    usage = 'usage:\npython3 webnlg_baseline_input.py -i <data-directory> [-s]' \
           '\ndata-directory is the directory where you unzipped the archive with data'
    try:
        opts, args = getopt.getopt(argv, 'i:s', ['inputdir=','shuffle'])
    except getopt.GetoptError:
        print(usage)
        sys.exit(2)
    input_data = False
    shuffleTripleSet = False
    for opt, arg in opts:
        if opt in ('-i', '--inputdir'):
            inputdir = arg
            input_data = True
        elif opt in ('-s', '--shuffle'):
            shuffleTripleSet = True
        else:
            print(usage)
            sys.exit()
    if not input_data:
        print(usage)
        sys.exit(2)
    print('Input directory is ', inputdir)
    input_files(inputdir)


if __name__ == "__main__":
    main(sys.argv[1:])

