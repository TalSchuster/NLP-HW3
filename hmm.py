from data import *
import numpy as np

STOP = "STOP"

def hmm_train(sents):
    """
        sents: list of tagged sentences
        sents contain sentences which are arrays of touple 
        Rerutns: the q-counts and e-counts of the sentences' tags
    """
    total_tokens = 0
    q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts = {}, {}, {}, {}, {}

    for sent in sents:
        total_tokens += len(sent)
        for i in xrange(len(sent)):
            word_tag = sent[i]
            if word_tag not in e_word_tag_counts:
                e_word_tag_counts[word_tag] = 0
            e_word_tag_counts[word_tag] += 1

            uni = sent[i][1]
            if uni not in q_uni_counts:
                q_uni_counts[uni] = 0
            q_uni_counts[uni] += 1

            if i >= 1:
                bi = (sent[i-1][1], sent[i][1])
                if bi not in q_bi_counts:
                    q_bi_counts[bi] = 0
                q_bi_counts[bi] += 1

            if i >= 2:
                tri = (sent[i-2][1], sent[i-1][1], sent[i])
                if tri not in q_tri_counts:
                    q_tri_counts[tri] = 0
                q_tri_counts[tri] += 1

        # Add STOP to end of sentence for the viterbi run
        if STOP not in q_uni_counts:
            q_uni_counts[STOP] = 0
        q_uni_counts[STOP] += 1

        if len(sent) >= 1:
            bi = (sent[-1][1], STOP)
            if bi not in q_bi_counts:
                q_bi_counts[bi] = 0
            q_bi_counts[bi] += 1

        if len(sent) >= 2:
            tri = (sent[-2][1], sent[-1][1], STOP)
            if tri not in q_tri_counts:
                q_tri_counts[tri] = 0
            q_tri_counts[tri] += 1


    # q_uni_counts, e_tag_counts is the same
    return total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, q_uni_counts


def prunning(e_word_tag_counts, e_tag_counts, factor = 0.01):
    """
        Gets the emission probabilities and prunes the occurrences that have lower probability by a factor from the
         maximal one.
         NOT FINISHED
    """
    return True


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


def hmm_viterbi(sent, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts):
    """
        Receives: a sentence to tag and the parameters learned by hmm
        Rerutns: predicted tags for the sentence
    """
    predicted_tags = [""] * (len(sent))
    possible_tags = e_tag_counts.keys()
    s = len(possible_tags)
    n = len(sent)

    # Preparing the table, it starts with zeros for pi(0,*,*) and for entries not filled because of prunning policy
    table = np.zeros((n + 1, s, s), np.float32)
    bp = np.zeros((n + 1, s, s), np.int8)

    # Phase 1: Filling the table
    for index, (word, _) in enumerate(sent):
        k = index + 1
        for v_index, v in enumerate(possible_tags):
            e = e_prob(word, v, e_word_tag_counts, e_tag_counts)
            if e == 0:
                continue
            for u_index, u in enumerate(possible_tags):
                if prunning_policy(word, u, v):
                    max_val, max_bp = 0, 0
                    for w_index, w in enumerate(possible_tags):
                        val = table[k-1][w_index][u_index] \
                              * q_prob(w, u, v, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts) * e
                        if val > max_val:
                            max_val = val
                            max_bp = w_index
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
    if max_val == 0.0:
        print "problem"
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
    for sent in test_data:
        prediction = hmm_viterbi(sent, total_tokens, q_tri_counts, q_bi_counts,
                                 q_uni_counts, e_word_tag_counts, e_tag_counts)
        for i, (word, pos) in enumerate(sent):
            num_of_words += 1
            if pos == prediction[i]:
                num_of_correct_tags += 1
    return str(float(num_of_correct_tags)/num_of_words)


    return acc_viterbi

if __name__ == "__main__":
    train_sents = read_conll_pos_file("Penn_Treebank/train.gold.conll")
    dev_sents = read_conll_pos_file("Penn_Treebank/dev.gold.conll")
    vocab = compute_vocab_count(train_sents)

    train_sents = preprocess_sent(vocab, train_sents)
    dev_sents = preprocess_sent(vocab, dev_sents)

    lambda1, lambda2 = 0.3, 0.3
    lambda3 = 1 - lambda1 - lambda2

    total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts = hmm_train(train_sents)
    acc_viterbi = hmm_eval(dev_sents, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts,e_tag_counts)
    print "dev: acc hmm viterbi: " + acc_viterbi

    if os.path.exists("Penn_Treebank/test.gold.conll"):
        test_sents = read_conll_pos_file("Penn_Treebank/test.gold.conll")
        test_sents = preprocess_sent(vocab, test_sents)
        acc_viterbi = hmm_eval(test_sents, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts,
                                           e_word_tag_counts, e_tag_counts)
        print "test: acc hmm viterbi: " + acc_viterbi