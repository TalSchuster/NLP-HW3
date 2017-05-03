import os
import re

MIN_FREQ = 3

# Categories for rare words (shouldn't overlap)
REGS = {
    'oneDigitPosNum' : '^[+]?[0-9]$',
    'oneDigitNegNum' : '^-[0-9]$',
    'twoDigitPosNum' : '^[+]?[0-9]{2}$',
    'twoDigitNegNum' : '^-[0-9]{2}$',
    'threeDigitPosNum' : '^[+]?[0-9]{3}$',
    'threeDigitNegNum' : '^-[0-9]{3}$',
    'fourDigitPosNum': '^[+]?[0-9]{4}$',
    'fourDigitNegNum': '^-[0-9]{4}$',
    'containsDigitAndDash': '^[0-9]+\-[0-9]+(\-[0-9]+)?$',
    'containsDigitAndSlash': '^[0-9]+\/[0-9]+(\/[0-9]+)?$',
    'containsDigitAndComma': '^[0-9]+\,[0-9]+$',
    'containsDigitAndPeriod': '^[0-9]+\.[0-9]+$',
    'allCaps': '^[A-Z]+$',
    'capPeriod': '^[A-Z]\.$',
    'initCap': '^[A-Z][a-z]+$',
    'lowerCase': '^[a-z]+$',
}

NUM_REG = '^[-+]?[0-9]+$'

def invert_dict(d):
    res = {}
    for k, v in d.iteritems():
        res[v] = k
    return res

def read_conll_pos_file(path):
    """
        Takes a path to a file and returns a list of word/tag pairs
    """
    sents = []
    with open(path, "r") as f:
        curr = []
        for line in f:
            line = line.strip()
            if line == "":
                sents.append(curr)
                curr = []
            else:
                tokens = line.strip().split("\t")
                curr.append((tokens[1],tokens[3]))
    return sents

def increment_count(count_dict, key):
    """
        Puts the key in the dictionary if does not exist or adds one if it does.
        Args:
            count_dict: a dictionary mapping a string to an integer
            key: a string
    """
    if key in count_dict:
        count_dict[key] += 1
    else:
        count_dict[key] = 1

def compute_vocab_count(sents):
    """
        Takes a corpus and computes all words and the number of times they appear
    """
    vocab = {}
    for sent in sents:
        for token in sent:
            increment_count(vocab, token[0])
    return vocab

def replace_word(word):
    """
        Replaces rare words with ctegories (numbers, dates, etc...)
    """
    for category in REGS.keys():
        p = re.compile(REGS[category])
        if p.match(word):
            return category

    p = re.compile(NUM_REG)
    if p.match(word):
        return "otherNum"

    return "UNK"

def preprocess_sent(vocab, sents):
    """
        return a sentence, where every word that is not frequent enough is replaced
    """
    res = []
    total, replaced = 0, 0
    for sent in sents:
        new_sent = []
        for token in sent:
            if token[0] in vocab and vocab[token[0]] >= MIN_FREQ:
                new_sent.append(token)
            else:
                new_sent.append((replace_word(token[0]), token[1]))
                replaced += 1
            total += 1
        res.append(new_sent)
    print "replaced: " + str(float(replaced)/total)
    return res


def test_replace_word():
    assert(replace_word('2') == 'oneDigitPosNum')
    assert (replace_word('-2') == 'oneDigitNegNum')
    assert (replace_word('23') == 'twoDigitPosNum')
    assert (replace_word('-23') == 'twoDigitNegNum')
    assert(replace_word('165') == 'threeDigitPosNum')
    assert (replace_word('-895') == 'threeDigitNegNum')
    assert(replace_word('7098') == 'fourDigitPosNum')
    assert (replace_word('-1897') == 'fourDigitNegNum')
    assert (replace_word('22-11') == 'containsDigitAndDash')
    assert (replace_word('22-11-') == 'UNK')
    assert (replace_word('22-11-897') == 'containsDigitAndDash')
    assert (replace_word('22/11/1897') == 'containsDigitAndSlash')
    assert (replace_word('1922/11/8') == 'containsDigitAndSlash')
    assert (replace_word('11/8') == 'containsDigitAndSlash')
    assert (replace_word('/11/8') == 'UNK')
    assert (replace_word('ABDFSADA') == 'allCaps')
    assert (replace_word('T.') == 'capPeriod')
    assert (replace_word('Mandatory') == 'initCap')
    assert (replace_word('mandatory') == 'lowerCase')


if __name__ == "__main__":
    test_replace_word()




