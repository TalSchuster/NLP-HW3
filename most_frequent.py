from data import *


def most_frequent_train(train_data):
    """
    Gets training data that includes tagged sentences.
    Returns a dictionary that maps every word in the training set to its most frequent tag.
    """
    # Filling a dictionary from words and pos tags to their counters
    tokens_to_tags_counts = {}
    for sent in train_data:
        for word, pos in sent:
            if word not in tokens_to_tags_counts:
                tokens_to_tags_counts[word] = {}
            curr_dict = tokens_to_tags_counts[word]
            if pos in curr_dict:
                curr_dict[pos] += 1
            else:
                curr_dict[pos] = 1

    # Now extracting the most frequent tag for each word
    most_frequent_dict = {}
    for word in tokens_to_tags_counts:
        curr_dict = tokens_to_tags_counts[word]
        most_freq_count, most_freq_pos = 0, ''
        for pos in curr_dict:
            if curr_dict[pos] > most_freq_count:
                most_freq_count = curr_dict[pos]
                most_freq_pos = pos
        most_frequent_dict[word] = most_freq_pos
    return most_frequent_dict


def most_frequent_eval(test_set, pred_tags):
    """
    Gets test data and tag prediction map.
    Returns an evaluation of the accuracy of the most frequent tagger.
    """
    num_of_words, num_of_correct_tags = 0, 0
    for sent in test_set:
        for word, pos in sent:
            num_of_words += 1
            if pos == pred_tags[word]:
                num_of_correct_tags += 1
    return str(float(num_of_correct_tags)/num_of_words)


if __name__ == "__main__":
    train_sents = read_conll_pos_file("Penn_Treebank/train.gold.conll")
    dev_sents = read_conll_pos_file("Penn_Treebank/dev.gold.conll")
    vocab = compute_vocab_count(train_sents)
    train_sents = preprocess_sent(vocab, train_sents)
    dev_sents = preprocess_sent(vocab, dev_sents)

    model = most_frequent_train(train_sents)
    print "dev: most frequent acc: " + most_frequent_eval(dev_sents, model)

    if os.path.exists('Penn_Treebank/test.gold.conll'):
        test_sents = read_conll_pos_file("Penn_Treebank/test.gold.conll")
        test_sents = preprocess_sent(vocab, test_sents)
        print "test: most frequent acc: " + most_frequent_eval(test_sents, model)