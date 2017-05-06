from data import *
import numpy as np
import timeit

STOP = ("STOP", "STOP")
START = ("*", "*")


def hmm_train(sents):
    """
        sents: list of tagged sentences
        sents contain sentences which are arrays of touple 
        Rerutns: the q-counts and e-counts of the sentences' tags
    """
    print 'collecting counts'
    start_time = timeit.default_timer()
    total_tokens = 0
    q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts = {}, {}, {}, {}

    for sent in sents:
        sent.append(STOP)
        total_tokens += len(sent)
        sent.insert(0, START)
        sent.insert(0, START)
        for i in xrange(1,len(sent)):
            uni = sent[i][1]
            if uni not in q_uni_counts:
                q_uni_counts[uni] = 0
            q_uni_counts[uni] += 1

            bi = (sent[i-1][1], sent[i][1])
            if bi not in q_bi_counts:
                q_bi_counts[bi] = 0
            q_bi_counts[bi] += 1

            if i >= 2:
                tri = (sent[i-2][1], sent[i-1][1], sent[i])
                if tri not in q_tri_counts:
                    q_tri_counts[tri] = 0
                q_tri_counts[tri] += 1

                word_tag = sent[i]
                if word_tag not in e_word_tag_counts:
                    e_word_tag_counts[word_tag] = 0
                e_word_tag_counts[word_tag] += 1

    e_tag_counts = dict(q_uni_counts)
    del e_tag_counts[START[1]]

    stop_time = timeit.default_timer()
    print 'Time for computing counts: %.2f seconds' % (stop_time - start_time)

    # q_uni_counts, e_tag_counts is the same
    return total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts

def get_pruned_emmisions(e_word_tag_counts, e_tag_counts, factor = 0.1):
    """
        Gets the emission probabilities and returns a pruned e_word_tag_counts the occurrences that have lower 
        probability by a factor from the maximal one.
    """
    words = {}
    for word, tag in e_word_tag_counts.keys():
        if word not in words:
            words[word] = []

        words[word].append(e_prob(word, tag, e_word_tag_counts, e_tag_counts))

    max_prob_word = {}
    for word in words:
        max_prob_word[word] = max(words[word])

    e_word_tag_counts_pruned = {}
    for word, tag in e_word_tag_counts.keys():
        if e_prob(word, tag, e_word_tag_counts, e_tag_counts) >= factor * max_prob_word[word]:
            e_word_tag_counts_pruned[(word,tag)] = e_word_tag_counts[(word,tag)]

    return e_word_tag_counts_pruned


def q_prob(prev_prev_tag, prev_tag, curr_tag, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts):
    q = 0.0
    trigram = (prev_prev_tag, prev_tag, curr_tag)
    if trigram in q_tri_counts:
        q += (lambda1 * q_tri_counts[trigram]) / q_bi_counts[(prev_prev_tag, prev_tag)]

    bigram = (prev_tag, curr_tag)
    if bigram in q_bi_counts:
        q += (lambda2 * q_bi_counts[bigram]) / q_uni_counts[prev_tag]

    if curr_tag in q_uni_counts:
        q += (lambda3 * q_uni_counts[curr_tag])/total_tokens

    return q


def e_prob(word, pos, e_word_tag_counts, e_tag_counts):
    word_pos = (word, pos)
    if word_pos not in e_word_tag_counts:
        return 0.0
    return float(e_word_tag_counts[word_pos]) / e_tag_counts[pos]


def possible_tag_set(tag_set, curr_index, for_prev_prev=True):
    if curr_index > 2 or (curr_index == 2 and not for_prev_prev):
        return tag_set
    return [START[1]]


def hmm_viterbi(sent, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts):
    """
        Receives: a sentence to tag and the parameters learned by hmm
        Rerutns: predicted tags for the sentence
    """
    predicted_tags = [""] * (len(sent))
    possible_tags = e_tag_counts.keys()
    s = len(possible_tags)
    n = len(sent)

    # e_word_tag_counts_pruned = get_pruned_emmisions(e_word_tag_counts, e_tag_counts)
    e_word_tag_counts_pruned = e_word_tag_counts

    # Preparing the table, it starts with zeros for pi(0,*,*) and for entries not filled because of prunning policy
    table = np.zeros((n + 1, s + 1, s + 1), np.float64)
    table[0][s][s] = 1.0
    bp = np.zeros((n + 1, s + 1, s + 1), np.int8)

    # Phase 1: Filling the table
    for index, (word, _) in enumerate(sent):
        k = index + 1
        for v_index, v in enumerate(possible_tags):
            e = e_prob(word, v, e_word_tag_counts_pruned, e_tag_counts)
            if e == 0:
                continue
            for u_index, u in enumerate(possible_tag_set(possible_tags, k, False)):
                if u == "*":
                    u_index = s
                max_val, max_bp = 0, 0
                for w_index, w in enumerate(possible_tag_set(possible_tags, k, True)):
                    if w == "*":
                        w_index = s
                    q = q_prob(w, u, v, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts)
                    val = table[k-1][w_index][u_index] * q * e
                    if val > max_val:
                        max_val = val
                        max_bp = w_index
                # if max_val == 0: print "prob val"
                table[k][u_index][v_index] = max_val
                bp[k][u_index][v_index] = max_bp

    # Phase 2: Finding maximal assignment for y_(n-1), y_n according to the table
    max_val, max_u, max_v = 0.0, 0, 0
    for u_index, u in enumerate(possible_tags):
        for v_index, v in enumerate(possible_tags):
            curr_val = table[n][u_index][v_index] \
                       * q_prob(u, v, "STOP", total_tokens, q_tri_counts, q_bi_counts, q_uni_counts)
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


def hmm_eval(test_data, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts,e_tag_counts):
    """
    Receives: test data set and the parameters learned by hmm
    Returns an evaluation of the accuracy of hmm
    """
    num_of_words, num_of_correct_tags = 0, 0
    for i, sent in enumerate(test_data):
        if i % 100 == 0 and i != 0:
            print i, float(num_of_correct_tags)/num_of_words
        prediction = hmm_viterbi(sent, total_tokens, q_tri_counts, q_bi_counts,
                                 q_uni_counts, e_word_tag_counts, e_tag_counts)
        for i, (word, pos) in enumerate(sent):
            num_of_words += 1
            if pos == prediction[i]:
                num_of_correct_tags += 1
    return str(float(num_of_correct_tags)/num_of_words)

if __name__ == "__main__":
    train_sents = read_conll_pos_file("Penn_Treebank/train.gold.conll")
    dev_sents = read_conll_pos_file("Penn_Treebank/dev.gold.conll")
    vocab = compute_vocab_count(train_sents)

    train_sents = preprocess_sent(vocab, train_sents)
    dev_sents = preprocess_sent(vocab, dev_sents)

    lambda1, lambda2 = 0.3, 0.3
    lambda3 = 1 - lambda1 - lambda2

    total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts = hmm_train(train_sents)
    e_word_tag_counts_prunned = get_pruned_emmisions(e_word_tag_counts, e_tag_counts, 0.01)
    acc_viterbi = hmm_eval(dev_sents, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts,
                           e_word_tag_counts_prunned ,e_tag_counts)
    print "dev: acc hmm viterbi: " + acc_viterbi

    if os.path.exists("Penn_Treebank/test.gold.conll"):
        test_sents = read_conll_pos_file("Penn_Treebank/test.gold.conll")
        test_sents = preprocess_sent(vocab, test_sents)
        acc_viterbi = hmm_eval(test_sents, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts,
                                           e_word_tag_counts, e_tag_counts)
        print "test: acc hmm viterbi: " + acc_viterbi