import random
import re
import string
import xml.etree.ElementTree as Et
from collections import defaultdict
import _pickle as pickle


punc_regex = re.compile('[%s]' % re.escape(string.punctuation))
def remove_punc(s):  # From Vinko's solution, with fix.
    return punc_regex.sub('', s)

def normalize(p, split=False):
    p = punc_regex.sub('', p.lower().replace('_', ' ').replace('"', ''))
    if split:
        return ' '.join(p.split())
    else:
        return ''.join(p.split())

def normalize2(p, punc=False, split=False, lower=True):
    p = p.replace('_', ' ').replace('"', '')
    if lower:
        p = p.lower()
    if not punc:
        p = punc_regex.sub("", p)
    else:
        p = ' '.join(re.split('(\W)', p))
    if split:
        return ' '.join(p.split())
    else:
        return ''.join(p.split())


class Triple:

    def __init__(self, s, p, o):
        self.s = s
        self.o = o
        self.p = p

    def __eq__(self, other):
        if not isinstance(other, Triple):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return remove_punc(self.s) == remove_punc(other.s) and \
               remove_punc(self.p) == remove_punc(other.p) and \
               remove_punc(self.o) == remove_punc(other.o)

class Tripleset:

    def __init__(self):
        self.triples = []

    @property
    def size(self):
        return len(self.triples)

    def fill_tripleset(self, t):
        for xml_triple in t:
            s, p, o = xml_triple.text.split(' | ')
            triple = Triple(s, p, o)
            self.triples.append(triple)

    def fill_tripleset2(self, t):
        for xml_triple in t:
            s, p, o = xml_triple.split(' | ')
            triple = Triple(s, p, o)
            self.triples.append(triple)

    def shuffle_triples(self):
        pass
        # random.shuffle(self.triples)

    def get_order(self, tripleset2):
        order = []
        for triple in tripleset2.triples:
            idx = self.triples.index(triple)
            order.append(idx)
        assert len(set(order)) == len(order)
        return list(map(str, order))


class Lexicalisation:

    def __init__(self, lex, comment, lid, orderedtripleset=None, refs=None, template=None, tripleset_split=[]):
        self.lex = lex
        self.comment = comment
        self.id = lid
        self.orderedtripleset = orderedtripleset
        self.refs = refs
        self.template = template
        self.tripleset_split = tripleset_split
        self.good = True


class Entry:

    def __init__(self, category, size, eid):
        self.originaltripleset = []
        self.modifiedtripleset = Tripleset()
        self.lexs = []
        self.category = category
        self.size = size
        self.id = eid
        self.agent_entity_map = {}
        self.agent_entity_map_relex = {}
        self.entity_agent_map = {}

    def fill_originaltriple(self, xml_t):
        otripleset = Tripleset()
        self.originaltripleset.append(otripleset)   # multiple originaltriplesets for one entry
        otripleset.fill_tripleset(xml_t)

    def fill_modifiedtriple(self, xml_t):
        self.modifiedtripleset.fill_tripleset(xml_t)
        self.modifiedtripleset.shuffle_triples()

    def create_lex(self, xml_lex, size):
        comment = xml_lex.attrib['comment']
        lid = xml_lex.attrib['lid']
        tripleset, lex_text, refs, lex_template = None, "", [], ''
        for child in xml_lex:
            if child.tag == "sortedtripleset":
                sents = []
                for sentence in child:
                    if len(sents) and not len(sents[-1]):
                        pass
                    else:
                        sents.append([])
                    for striple in sentence:
                        sents[-1].append(striple)
                sents_len = [len(subsents) for subsents in sents if len(subsents)]
                sents = [sent for subsents in sents for sent in subsents]

                tripleset = Tripleset()
                tripleset.fill_tripleset(sents)
            elif child.tag == "text":
                lex_text = child.text
            elif child.tag == "template":
                lex_template = child.text
            elif child.tag == 'references':
                for ref in child:
                    ref_info = {'entity': ref.attrib['entity'], 'tag': ref.attrib['tag'], 'text': ref.text}
                    refs.append(ref_info)
        lex = Lexicalisation(lex_text, comment, lid, tripleset, refs, lex_template, sents_len)
        if tripleset is not None and lex_text is not None: # and tripleset.size == size:
            self.lexs.append(lex)

    def count_lexs(self):
        return len(self.lexs)


class Benchmark:

    def __init__(self):
        self.entries = []

    def fill_benchmark(self, fileslist):
        cnt = 0
        for file in fileslist:
            tree = Et.parse(file[0] + '/' + file[1])
            root = tree.getroot()
            for xml_entry in root.iter('entry'):
                # ignore triples with no lexicalisations
                lexfound = False
                for child in xml_entry:
                    if child.tag == "lex":
                        lexfound = True
                        break
                if lexfound is False:
                    continue

                entry_id = xml_entry.attrib['eid']
                category = xml_entry.attrib['category']
                size = xml_entry.attrib['size']
                entry = Entry(category, size, entry_id)
                for child in xml_entry:
                    if child.tag == 'originaltripleset':
                        entry.fill_originaltriple(child)
                    elif child.tag == 'modifiedtripleset':
                        entry.fill_modifiedtriple(child)
                    elif child.tag == 'lex':
                        entry.create_lex(child, int(size))
                    elif child.tag == 'entitymap':
                        for entity_map in child:
                            agent, entity = entity_map.text.split(' | ')
                            agent = agent.strip()
                            entity = entity.replace('_', ' ').replace('"', '').strip()
                            entity = ' '.join(re.split('(\W)', entity))
                            assert agent not in entry.agent_entity_map
                            entry.agent_entity_map[agent] = normalize(entity)
                            entry.agent_entity_map_relex[agent.lower()] = normalize2(entity, punc=True, lower=True, split=True)
                        entry.entity_agent_map = {e:a for a, e in entry.agent_entity_map.items()}
                for lex in entry.lexs:  # check the size
                    # assert int(size) == len(lex.orderedtripleset.triples)
                    cnt += 1
                self.entries.append(entry)
        print(" ** Reading {} lex entries **".format(cnt))


    def total_lexcount(self):
        count = [entry.count_lexs() for entry in self.entries]
        return sum(count)

    def unique_p(self):
        properties = [triple.p for entry in self.entries for triple in entry.modifiedtripleset.triples]
        return len(set(properties))

    def entry_count(self, size=None, cat=None):
        """
        calculate the number of entries in benchmark
        :param size: size (should be string)
        :param cat: category
        :return: entry count
        """
        if not size and cat:
            entries = [entry for entry in self.entries if entry.category == cat]
        elif not cat and size:
            entries = [entry for entry in self.entries if entry.size == size]
        elif not size and not cat:
            return len(self.entries)
        else:
            entries = [entry for entry in self.entries if entry.category == cat and entry.size == size]
        return len(entries)

    def lexcount_size_category(self, size='', cat=''):
        count = [entry.count_lexs() for entry in self.entries if entry.category == cat and entry.size == size]
        return len(count)

    def property_map(self):
        mprop_oprop = defaultdict(set)
        for entry in self.entries:
            for tripleset in entry.originaltripleset:
                for i, triple in enumerate(tripleset.triples):
                    mprop_oprop[entry.modifiedtripleset.triples[i].p].add(triple.p)
        return mprop_oprop

    # def order_tripleset(self, ordered_dataset):
    #     with open(ordered_dataset, 'rb') as fr:
    #         lexEntry_orderedTripleset = pickle.load(fr)
    #
    #     for entry in self.entries:
    #         for lex in entry.lexs:
    #             entry_id = "{}_{}_{}_{}".format(entry.id, entry.size, entry.category, lex.id)
    #             try:
    #                 ordered_tripleset = Tripleset()
    #                 orderedtripleset_str = lexEntry_orderedTripleset[entry_id]["ordered_source_out"]
    #                 for triple in orderedtripleset_str.split(" < TSP > "):
    #                     s, p, o = triple.split(" | ")
    #                     ordered_tripleset.triples.append(Triple(s, p, o))
    #                 lex.orderedtripleset = ordered_tripleset
    #             except:
    #                 # print("Fail to match the ordered tripleset of {} ...".format(entry_id))
    #                 lex.orderedtripleset = entry.modifiedtripleset


