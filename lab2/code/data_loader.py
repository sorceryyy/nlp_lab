import spacy
from copy import copy
from collections import Counter

nlp = spacy.load("en_core_web_sm")

seps=[',', '-LRB-', '-RRB-']
stopwords = [',','.','the','was','-LRB-','-RRB-']

def split(txt):
    default_sep = seps[0]
    # we skip seps[0] because that's the default separator
    for sep in seps[1:]:
        txt = txt.replace(sep, default_sep)
    splited_list = [i.strip() for i in txt.split(default_sep)]
    index = [0]
    splited_num = [len(i.split())+1 for i in splited_list]
    for i, num in enumerate(splited_num):
        index.append(index[i]+num)
    
    return splited_list, index

def add_adj(token):
    prepend = ''
    for child in token.children:
        if child.pos_ != 'ADV':
            continue
        prepend += child.text + ' '
    return prepend + token.text

def add_word(descriptive_term:set, token):
    if token.text not in stopwords:
        descriptive_term.add(token.text)
        # if token.pos_ == 'ADJ' or token.pos_ == 'VERB':
        #     descriptive_term.add(token.text)

def find_range(numbers, target):
    for i in range(len(numbers) - 1):
        if numbers[i] <= target < numbers[i+1]:
            return numbers[i], numbers[i+1]
    return 0 ,numbers[-1]
    
    return None


class Data:
    def __init__(self,file_path,train=True) -> None:
        '''parameter: file_path, train(default: true)'''
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        texts = []
        targets = []
        labels = []
        targets_indexes = []
        range_indexes = []
        for i in range(0, len(lines), 3):
            _, sep_indexes = split(lines[i].strip())
            #text = [setence for setence in sep_setence if '$T$' in setence][0]
            text = lines[i]
            begin = text.split().index('$T$')
            end = begin + len(lines[i+1].split())

            range_b,range_e = find_range(sep_indexes,begin)
            range_indexes.append( (range_b, range_e+len(lines[i+1].split())-1) )

            texts.append(text.strip().replace('$T$', lines[i+1].strip()))
            targets.append(lines[i+1].strip())
            if train:
                labels.append(int(lines[i+2].strip()))
            targets_indexes.append((begin,end))

        self.texts, self.targets, self.labels = \
              texts, targets, labels
        self.old_texts = self.texts #never changed
        self.targets_indexes = targets_indexes
        self.range_indexes = range_indexes

        if train:
            x = Counter(self.labels)
            print("labels: ",x)
            

    def extract_context(self):
        sentences = self.old_texts
        aspects = []
        indexes = []
        c_len = 3
        for i, sentence in enumerate(sentences):
            words = sentence.split()
            begin = max(self.range_indexes[i][0],self.targets_indexes[i][0]-c_len)
            end = min(self.range_indexes[i][1], self.targets_indexes[i][1]+c_len)
            aspect = copy(words[begin:end])
            aspects.append(aspect)
            indexes.append((begin,end)) # 左闭右开！

        return aspects,indexes


    def extract_dataset(self):
        '''return train, test dataset of target'''
        nlp = spacy.load("en_core_web_sm")
        sentences = self.texts
        contextes = []
        relateds = []
        _, indexes = self.extract_context()
        for i, sentence in enumerate(sentences):
            doc = nlp(sentence)
            context_term = set()
            related_term = set()
            target = self.targets[i]
            for j in range(indexes[i][0],min(len(doc),indexes[i][1])):
                token = doc[j]
                add_word(context_term,token)
                for ancestor in token.ancestors:
                    add_word(related_term,ancestor)
                for child in token.children:
                    add_word(related_term,child)
            if not context_term:
                context_term.add("None")
            context = " ".join(context_term)
            contextes.append(context)

            related = " ".join(related_term)
            relateds.append(related)

        self.texts = contextes
        self.relateds = relateds

# descriptive_term = set()
#             target = self.targets[i]
#             for token in doc:
#                 in_token = False
#                 for child in token.children:
#                     if child.text == target:
#                         in_token = True
#                         break
#                 if in_token:
#                     for child in token.children:
#                         if child.pos_ == 'ADJ':
#                             descriptive_term.add(add_adj(child))
#                         for child_child in child.children:
#                             if child_child.pos_ == 'ADJ':
#                                 descriptive_term.add(add_adj(child_child))


# to grep the adj of (adj n) 
            # chunks = []
            # for chunk in doc.noun_chunks:
            #     out = {}
            #     root = chunk.root
            #     out[root.pos_] = root
            #     for tok in chunk:
            #         if tok != root:
            #             out[tok.pos_] = tok
            #     chunks.append(out)

            # grep the adj after a noun
            # descriptive_term = ''
            # for token in doc:
            #     child_text = [child.text for child in token.children]
            #     print(child_text)
            #     if token.dep_ == 'nsubj' and token.pos_ == 'NOUN':
            #         target = token.text
                    
            #     if token.pos_ == 'ADJ':
            #         prepend = ''
            #         for child in token.children:
            #             if child.pos_ != 'ADV':
            #                 continue
            #             prepend += child.text + ' '
            #         descriptive_term = prepend + token.text    