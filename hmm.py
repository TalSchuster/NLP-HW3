from data import *
import numpy as np

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
                tri = (sent[i-2], sent[i-1], sent[i])
                if tri not in q_tri_counts:
                    q_tri_counts[tri] = 0
                q_tri_counts[tri] += 1

    # q_uni_counts, e_tag_counts is the same
    return total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, q_uni_counts


def prunning_policy(k, u, v):
    """
        Gets a word and candidate POS tags for it and for the precious one.
        Returns True if tere's need to calculate pi(k,u,v)
    """
    return True


def hmm_viterbi(sent, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts,e_tag_counts):
    """
        Receives: a sentence to tag and the parameters learned by hmm
        Rerutns: predicted tags for the sentence
    """
    predicted_tags = [""] * (len(sent))

    table = np.ones((len(sent), total_tokens, total_tokens))

    return predicted_tags

def hmm_eval(test_data, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts,e_tag_counts):
    """
    Receives: test data set and the parameters learned by hmm
    Returns an evaluation of the accuracy of hmm
    """
    acc_viterbi = 0.0
    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE
    return acc_viterbi

if __name__ == "__main__":
    train_sents = read_conll_pos_file("Penn_Treebank/train.gold.conll")
    dev_sents = read_conll_pos_file("Penn_Treebank/dev.gold.conll")
    vocab = compute_vocab_count(train_sents)

    train_sents = preprocess_sent(vocab, train_sents)
    dev_sents = preprocess_sent(vocab, dev_sents)

    total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts = hmm_train(train_sents)
    acc_viterbi = hmm_eval(dev_sents, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts,e_tag_counts)
    print "dev: acc hmm viterbi: " + acc_viterbi

    if os.path.exists("Penn_Treebank/test.gold.conll"):
        test_sents = read_conll_pos_file("Penn_Treebank/test.gold.conll")
        test_sents = preprocess_sent(vocab, test_sents)
        acc_viterbi = hmm_eval(test_sents, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts,
                                           e_word_tag_counts, e_tag_counts)
        print "test: acc hmm viterbi: " + acc_viterbi