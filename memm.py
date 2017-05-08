from data import *
from sklearn.feature_extraction import DictVectorizer
from sklearn import linear_model
import numpy as np
import time
import pickle
import os
import timeit

MODEL_FILE_NAME = 'saved_model.pickle' # Use empty string to disable storing.
GREEDY_FILE_NAME = 'greedy_tags.pickle'
VITERBI_FILE_NAME = 'viterbi_tags.pickle'

class q_values_cache:
    """
    Holds a cache for computed q probabilities
    """

    def __init__(self, logreg):
        self.logreg = logreg
        self.cache = {}

    def get_prob(self, tag_index, features):
        if features in self.cache:
            return self.cache[features][tag_index]

        self.cache[features] = self.logreg.predict_proba(features)[0]
        return self.cache[features][tag_index]


def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

def hasUpper(inputString):
    return any(char.isupper() for char in inputString)

def hasHyphen(inputString):
    return any(char=='-' for char in inputString)

def extract_features_base(curr_word, next_word, prev_word, prevprev_word, prev_tag, prevprev_tag):
    """
        Receives: a word's local information
        Rerutns: The word's features.
    """
    features = {}
    features['word'] = curr_word
    features['word_prev'] = prev_word
    features['word_prevprev'] = prevprev_word
    features['word_next'] = next_word
    features['prev_tag'] = prev_tag
    features['prev_tags'] = '%s,%s' % (prevprev_tag, prev_tag)
    for i in xrange(1,min(5,len(curr_word))):
        prefix_str = 'prefix_%s' % (i)
        suffix_str = 'suffix_%s' % (i)
        features[prefix_str] = curr_word[:i]
        features[suffix_str] = curr_word[-i:]

    features['contains_number'] = hasNumbers(curr_word)
    features['contains_upper'] = hasUpper(curr_word)
    features['contains_hyphen'] = hasHyphen(curr_word)

    return features

def extract_features_base_test():
    pred = extract_features_base('walked', 'fast', 'he', '*', 'A', 'B')
    ans = {
        'word': 'walked',
        'word_prev': 'he',
        'word_prevprev': '*',
        'word_next': 'fast',
        'prev_tag': 'A',
        'prev_tags': 'B,A',
        'prefix_1': 'w',
        'prefix_2': 'wa',
        'prefix_3': 'wal',
        'prefix_4': 'walk',
        'prefix_5': 'walke',
        'suffix_5': 'alked',
        'suffix_4': 'lked',
        'suffix_3': 'ked',
        'suffix_2': 'ed',
        'suffix_1': 'd',
        'contains_number': False,
        'contains_upper': False,
        'contains_hyphen': False,
    }
    for key in pred.keys():
        if pred[key] != ans[key]:
            raise ValueError('feature %s not equal: %s != %s' %(key, pred[key], ans[key]))

    pred = extract_features_base('Walked', 'fast', 'he', '*', 'A', 'B')
    if not pred['contains_upper']:
        raise ValueError('feature contains_upper should be true but got false')

    pred = extract_features_base('walk3d', 'fast', 'he', '*', 'A', 'B')
    if not pred['contains_number']:
        raise ValueError('feature contains_number should be true but got false')

    pred = extract_features_base('wal-ked', 'fast', 'he', '*', 'A', 'B')
    if not pred['contains_hyphen']:
        raise ValueError('feature contains_upper should be true but got false')

def extract_features(sentence, i):
    curr_word = sentence[i][0]
    prev_token = sentence[i - 1] if i > 0 else ('<s>', '*')
    prevprev_token = sentence[i - 2] if i > 1 else ('<s>', '*')
    next_token = sentence[i + 1] if i < (len(sentence) - 1) else ('</s>', 'STOP')
    return extract_features_base(curr_word, next_token[0], prev_token[0], prevprev_token[0], prev_token[1], prevprev_token[1])

def vectorize_features(vec, features):
    """
        Receives: feature dictionary
        Rerutns: feature vector

        Note: use this function only if you chose to use the sklearn solver!
        This function prepares the feature vector for the sklearn solver,
        use it for tags prediction.
    """
    example = [features]
    return vec.transform(example)


def create_examples(sents):
    print "building examples"
    examples = []
    labels = []
    num_of_sents = 0
    for sent in sents:
        num_of_sents += 1
        for i in xrange(len(sent)):
            features = extract_features(sent, i)
            examples.append(features)
            labels.append(tagset[sent[i][1]])
    return examples, labels
    print "done"


def memm_greedy(sent, logreg):
    """
        Receives: a sentence to tag and the parameters learned by hmm
        Rerutns: predicted tags for the sentence
    """
    return list(logreg.predict(sent))

def possible_tag_set(tag_set, curr_index, for_prev_prev=True):
    if curr_index > 2 or (curr_index == 2 and not for_prev_prev):
        return tag_set
    return ["*"]

def memm_viterbi(sent, logreg):
    """
        Receives: a sentence to tag and the parameters learned by hmm
        Rerutns: predicted tags for the sentence
    """
    predicted_tags = [""] * sent.shape[0]
    q_values = q_values_cache(logreg)
    possible_tags = index_to_tag_dict.keys()
    s = len(possible_tags)
    n = sent.shape[0]

    # Preparing the table, it starts with zeros for pi(0,*,*)
    table = np.zeros((n + 1, s + 1, s + 1), np.float64)
    table[0][s][s] = 1.0
    bp = np.zeros((n + 1, s + 1, s + 1), np.int8)

    # Phase 1: Filling the table
    for index, features in enumerate(sent):
        k = index + 1
        for v_index, v in enumerate(possible_tags):
            for u_index, u in enumerate(possible_tag_set(possible_tags, k, False)):
                if u == "*":
                    u_index = s
                max_val, max_bp = 0, 0
                for t_index, t in enumerate(possible_tag_set(possible_tags, k, True)):
                    if t == "*":
                        t_index = s
                    q = q_values.get_prob(v, features)
                    val = table[k-1][t_index][u_index] * q
                    if val > max_val:
                        max_val = val
                        max_bp = t_index
                # if max_val == 0: print "prob val"
                table[k][u_index][v_index] = max_val
                bp[k][u_index][v_index] = max_bp

    # Phase 2: Finding maximal assignment for y_(n-1), y_n according to the table
    max_val, max_u, max_v = 0.0, 0, 0
    for u_index, u in enumerate(possible_tags):
        for v_index, v in enumerate(possible_tags):
            curr_val = table[n][u_index][v_index]
            if curr_val > max_val:
                max_val = curr_val
                max_u, max_v = u_index, v_index
    predicted_tags[n-1], y_k_2 = possible_tags[max_v], max_v
    predicted_tags[n-2], y_k_1 = possible_tags[max_u], max_u

    # Phase 3: Finding maximal assignment iteratively for all other words
    for k in xrange(n-2, 0, -1):
        predicted_tag_index = bp[k+2][y_k_1][y_k_2]
        predicted_tags[k-1] = possible_tags[predicted_tag_index]
        y_k_2 = y_k_1
        y_k_1 = predicted_tag_index

    return predicted_tags


def memm_eval(test_data, vectorized_dev_data, logreg):
    """
    Receives: test data set and the parameters learned by hmm
    Returns an evaluation of the accuracy of hmm & greedy hmm
    """
    num_of_words, num_of_correct_in_greedy, num_of_correct_in_viterbi = 0, 0, 0

    start_time = timeit.default_timer()
    all_viterbi_tags = []
    all_greedy_tags = []
    for i, sent in enumerate(test_data):
        if i % 10 == 0 and i != 0:
            stop_time = timeit.default_timer()
            print "Evaluated %i sentences. Time: %.1f seconds. Greedy Acc: %.2f. Viterbi Acc: %.2f" \
                  % (i, stop_time - start_time, float(num_of_correct_in_greedy) / num_of_words * 100 \
                    , float(num_of_correct_in_viterbi) / num_of_words * 100)

        sent_vecrtorized = vectorized_dev_data[num_of_words:num_of_words+len(sent)]
        viterbi_tags = memm_viterbi(sent_vecrtorized, logreg)
        greedy_tags = memm_greedy(sent_vecrtorized, logreg)
        all_viterbi_tags.append(viterbi_tags)
        all_greedy_tags.append(greedy_tags)

        for index, (word, tag) in enumerate(sent):
            num_of_words += 1
            tag_index = tagset[tag]

            if viterbi_tags[index] == tag_index:
                num_of_correct_in_viterbi += 1

            if greedy_tags[index] == tag_index:
                num_of_correct_in_greedy += 1

    if VITERBI_FILE_NAME != '':
        print 'saving viterbi labels to file: ' + VITERBI_FILE_NAME
        pickle.dump(all_viterbi_tags, open(VITERBI_FILE_NAME, 'wb'))

    if GREEDY_FILE_NAME != '':
        print 'saving greedy labels to file: ' + GREEDY_FILE_NAME
        pickle.dump(all_greedy_tags, open(GREEDY_FILE_NAME, 'wb'))

    return float(num_of_correct_in_viterbi)/num_of_words, float(num_of_correct_in_greedy)/num_of_words


if __name__ == "__main__":
    extract_features_base_test()

    train_sents = read_conll_pos_file("Penn_Treebank/train.gold.conll")
    dev_sents = read_conll_pos_file("Penn_Treebank/dev.gold.conll")

    vocab = compute_vocab_count(train_sents)
    train_sents = preprocess_sent(vocab, train_sents)
    dev_sents = preprocess_sent(vocab, dev_sents)

    #The log-linear model training.
    #NOTE: this part of the code is just a suggestion! You can change it as you wish!
    curr_tag_index = 0
    tagset = {}
    for train_sent in train_sents:
        for token in train_sent:
            tag = token[1]
            if tag not in tagset:
                tagset[tag] = curr_tag_index
                curr_tag_index += 1
    index_to_tag_dict = invert_dict(tagset)
    vec = DictVectorizer()
    print "Create train examples"
    train_examples, train_labels = create_examples(train_sents)
    num_train_examples = len(train_examples)
    print "#example: " + str(num_train_examples)
    print "Done"

    print "Create dev examples"
    dev_examples, dev_labels = create_examples(dev_sents)
    num_dev_examples = len(dev_examples)
    print "#example: " + str(num_dev_examples)
    print "Done"

    all_examples = train_examples
    all_examples.extend(dev_examples)

    print "Vectorize examples"
    #vec = vectorize_features(vec, all_examples)
    all_examples_vectorized = vec.fit_transform(all_examples)
    train_examples_vectorized = all_examples_vectorized[:num_train_examples]
    dev_examples_vectorized = all_examples_vectorized[num_train_examples:]
    print "Done"

    logreg = linear_model.LogisticRegression(
        multi_class='multinomial', max_iter=128, solver='lbfgs', C=100000, verbose=1)
    if MODEL_FILE_NAME != '' and os.path.exists(MODEL_FILE_NAME):
        print 'loading existing model from file'
        logreg = pickle.load(open(MODEL_FILE_NAME, 'rb'))
        print 'Done'
    else:
        print "Fitting..."
        start = time.time()
        logreg.fit(train_examples_vectorized, train_labels)
        end = time.time()
        print "done, " + str(end - start) + " sec"
        if MODEL_FILE_NAME != '':
            print 'saving model to file: ' + MODEL_FILE_NAME
            pickle.dump(logreg, open(MODEL_FILE_NAME, 'wb'))
    #End of log linear model training

    acc_viterbi, acc_greedy = memm_eval(dev_sents, dev_examples_vectorized, logreg)
    print "dev: acc memm greedy: " + str(acc_greedy)
    print "dev: acc memm viterbi: " + str(acc_viterbi)
    if os.path.exists('Penn_Treebank/test.gold.conll'):
        test_sents = read_conll_pos_file("Penn_Treebank/test.gold.conll")
        test_sents = preprocess_sent(vocab, test_sents)
        acc_viterbi, acc_greedy = memm_eval(test_sents, logreg, vec)
        print "test: acc memmm greedy: " + str(acc_greedy)
        print "test: acc memmm viterbi: " + str(acc_viterbi)