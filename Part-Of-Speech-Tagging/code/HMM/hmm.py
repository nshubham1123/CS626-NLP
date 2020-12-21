import copy

class HMM:
    def __init__(self):
        self.model_name = "HMM POS tagger"

        self.tagged_sentences = []

        self.transition_matrix = {}
        self.emission_matrix = {}

        self.dict_word_tag = {} # tags for each word
        self.tag_set=[] #tag_set is same as transition probablity keys

        self.default_emission_prob = 0.0001
        self.default_transition_prob = 0.0001

    def calc_frequencies(self, tagged_sentences):
        # return emission and transition frequencies
        emission_matrix = {}
        transition_matrix = {}

        for sent in tagged_sentences:
            # prepend ^ to each sentence to make analysis easier.
            sent.insert(0, ("^", "^"))

            prev_tag = "" # for use in transition matrix calculation.
            for word, tag in sent:
                # increment count of this word occuring with this tag by 1.
                emission_matrix[tag][word] = emission_matrix.setdefault(tag, {}) \
                                             .setdefault(word, 0) + 1

                if prev_tag:
                    transition_matrix[prev_tag][tag] = transition_matrix.setdefault(prev_tag, {}) \
                                                       .setdefault(tag, 0) + 1
                prev_tag = tag

        return transition_matrix, emission_matrix

    def calc_probabilities(self, frequency_matrix):
        # return modified matrix.
        for key, val in frequency_matrix.items():
            total = 0.0
            for key1 in val:
                total += val[key1];
            for key2 in val:
                val[key2] /= total
        return frequency_matrix

    def make_dict_word_tag(self, emission_matrix):
        # dict to store tags for particular word
        dict_word_tag = {}
        for tag in emission_matrix.keys():
            for word in emission_matrix[tag].keys():
                try:
                    dict_word_tag[word][tag] = 1
                except:
                    dict_word_tag[word] = {tag: 1}

        return dict_word_tag

    def prepend_start_symbol(self, sequences):
        # adding ^ as first word/tag in each sentence
        for sequence in sequences:
            sequence.insert(0, '^')

        return sequences

    def viterbi_inference(self, query_sentence):
        # return list of inferred pos tags for this query sentence.
        prob = {} # nested dict to store cummilative probablities
        assigned = {} # nested dict to store previous word tag which give max probablity for current word tag
        words = query_sentence

        dict_word_tag=self.dict_word_tag
        emission=self.emission_matrix
        transition=self.transition_matrix

        # for simplicity, we prepend a "^" to the sentence.
        # This elegantly captures the probability of a tag being the first tag in a sentence.
        # setting up intial probablities for first word of sentence.
        first_word = "^"
        for tag in dict_word_tag[first_word].keys():
            try:
                prob[0][tag] = emission[tag][first_word]
            except:
                prob[0] = {tag: emission[tag][first_word]}

        for i in range(1, len(words)): # loop from second word to last

            word_prev = words[i - 1]
            word_cur = words[i]
            cur, prev = i, i - 1

            # handling if word not in train data
            if word_cur not in dict_word_tag.keys():
                for tag in self.emission_matrix.keys():
                    self.emission_matrix[tag][word_cur] = self.default_emission_prob
                # but '^' should not emit it.
                self.emission_matrix['^'][word_cur] = 0.0
                self.dict_word_tag[word_cur] = {}
                for tag in self.tag_set:
                    self.dict_word_tag[word_cur][tag] = 1
                # also remove '^' from list of possible tags. not really required, but still.
                del self.dict_word_tag[word_cur]['^']


            for tag_cur in dict_word_tag[word_cur].keys(): # loop for current word tags

                prob_max = -1
                tag_assigned = None

                for tag_prev in dict_word_tag[word_prev].keys(): # loop for previous word tags
                    # get the transition prob for this pair of tags. If tag_prev has no transitions at all,
                    # set an entry for it in transition matrix and then return the default probability for
                    # all tag_curs. The entry is set because otherwise, each time, a search would take
                    # O(#tags) time, but now, since it doesn't have to exhaust all keys, the time required
                    # possibly decreases.
                    trans_prob = self.transition_matrix.setdefault(tag_prev, {}) \
                                 .get(tag_cur, self.default_transition_prob)

                    prob_cur = prob[prev][tag_prev] * trans_prob
                    if (prob_cur > prob_max):
                        prob_max = prob_cur
                        tag_assigned = tag_prev
                try:
                    prob[cur][tag_cur] = prob_max * emission[tag_cur][word_cur] # prob is dict to store cummilative probablities till current word
                    assigned[cur][tag_cur] = tag_assigned # assigned is dict to store previous word tag for which current words's tag has highest probablity
                except:
                    prob[cur] = {tag_cur: prob_max * emission[tag_cur][word_cur]}
                    assigned[cur] = {tag_cur: tag_assigned}

        predicted_tags = []

        max_prob, last_tag = -1, None
        last_index = len(words) -1

        for tag in prob[last_index].keys(): # last tag calculated on basis of maximum probablity
            if (prob[last_index][tag] > max_prob):
                max_prob = prob[last_index][tag]
                last_tag = tag
        predicted_tags.append(last_tag)

        for i in range(last_index, 1, -1): # backtracking from last word using dict assigned
            last_tag = assigned[i][last_tag]
            predicted_tags.append(last_tag)

        predicted_tags.reverse()
        return predicted_tags

    def train(self, tagged_sentences, default_emission_prob = None,
              default_transition_prob = None):
        self.tagged_sentences = copy.deepcopy(list(tagged_sentences))
        if default_transition_prob is not None:
            self.default_transition_prob = default_transition_prob
        if default_emission_prob is not None:
            self.default_emission_prob = default_emission_prob

        self.transition_matrix, self.emission_matrix = self.calc_frequencies(self.tagged_sentences)
        self.transition_matrix = self.calc_probabilities(self.transition_matrix)
        self.emission_matrix = self.calc_probabilities(self.emission_matrix)

        self.dict_word_tag = self.make_dict_word_tag(self.emission_matrix)
        self.tag_set = [ key for key in self.transition_matrix.keys() ]

    def predict(self, test_sentences):
        # use current model to predict pos tags for given sentences.

        test_sentences = self.prepend_start_symbol(test_sentences)

        # use viterbi algo for each instance.
        predictions = [self.viterbi_inference(sentence) for sentence in test_sentences]
        return predictions
