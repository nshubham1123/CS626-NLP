import numpy as np
import utils

class ExperimentSetup:
    def __init__(self, total_tagged_sentences, tags_list, fold_k = 5):
        self.total_tagged_sentences = total_tagged_sentences
        self.fold_k = 5
        self.crossval_indices = self.calc_crossvalidation_indices(self.fold_k, self.total_tagged_sentences)
        self.tags_list = tags_list
        self.n_tags = len(self.tags_list)

    def calc_crossvalidation_indices(self, fold_k, total_tagged_sentences):
        # calculate the ending indices of different folds in the k-fold crossvalidation.
        incr = int(len(total_tagged_sentences) / fold_k) # will yield an integer.
                                                    #  Need to handle the remaining samples accordingly.
        crossval_indices = []
        for i in range(fold_k + 1):
            crossval_indices.append(i * incr)

        return crossval_indices

    def split_test_data(self, test_data_unsplit):
        test_data = [[word for (word, _) in sentence] for sentence in test_data_unsplit]
        true_tag_sequences = [[tag for (_, tag) in sentence] for sentence in test_data_unsplit]

        return test_data, true_tag_sequences

    def get_confusion_matrix(self, true_tag_sequences, predicted_tag_sequences):
        # return a matrix with predicted tags along columns and true tags along rows.
        # count of joint occurences is recorded.

        # sanity check.
        if (len(true_tag_sequences) != len(predicted_tag_sequences)):
            raise ValueError("Lists must have same number of sentences.")

        n_tags = self.n_tags
        conf_mat = np.zeros((n_tags, n_tags), dtype=int)

        # set up inverse dictionary.
        tag_index = {}
        for idx, tag in enumerate(self.tags_list):
            tag_index[tag] = idx
        self.tag_index = tag_index

        for i in range(len(true_tag_sequences)):
            true_seq, pred_seq = true_tag_sequences[i], predicted_tag_sequences[i]
            for j in range(len(true_seq)):
                conf_mat[tag_index[true_seq[j]]][tag_index[pred_seq[j]]] += 1

        return conf_mat

    def get_overall_accuracy(self, conf_mat):
        n_tags = self.n_tags
        tps = [conf_mat[i][i] for i in range(n_tags)]
        n_tp = sum(tps)

        n_total = np.sum(conf_mat)

        accuracy = n_tp / n_total

        return accuracy

    def get_tag_accuracy(self, idx, conf_mat):
        # calculate accuracy for tag at index idx.

        n_tp = conf_mat[idx][idx]

        n_tn = 0
        n_tags = self.n_tags
        for i in range(n_tags):
            for j in range(n_tags):
                if i != idx and j != idx:
                    # true negatives.
                    n_tn += conf_mat[i][j]

        n_total = np.sum(conf_mat)

        accuracy = 1.0 * (n_tp + n_tn) / n_total

        return accuracy

    def get_per_tag_accuracies(self, conf_mat):
        # returns per tag accuracies for all tags in tagset.
        n_tags = self.n_tags
        per_tag_accuracies = np.zeros((n_tags))

        for i in range(n_tags):
            per_tag_accuracies[i] = self.get_tag_accuracy(i, conf_mat)

        return per_tag_accuracies

    def run(self, model, flag_printaccs=True):
        overall_accuracy_arr = np.zeros((self.fold_k))
        pertag_accuracies_arr = np.zeros((self.fold_k, len(self.tags_list)))
        confusion_matrices = [] # initialized as list to avoid storing k*t^2 entries unnecessarily until used.

        for i in range(self.fold_k):
            # get train and test data for this instance of crossvalidation.
            test_data_unsplit = self.total_tagged_sentences[self.crossval_indices[i] : self.crossval_indices[i + 1]]
            train_data = self.total_tagged_sentences[0 : self.crossval_indices[i]] \
                         + self.total_tagged_sentences[self.crossval_indices[i + 1] : ]
            test_data, true_tag_sequences = self.split_test_data(test_data_unsplit)

            # initialize the model.
            predictor = model()

            # train the model.
            predictor.train(train_data)

            # test the model.
            predicted_tag_sequences = predictor.predict(test_data)

            # calculate accuracy. confusion matrix. per-tag accuracy
            confusion_matrices.append(self.get_confusion_matrix(true_tag_sequences, predicted_tag_sequences))
            overall_accuracy_arr[i] = self.get_overall_accuracy(confusion_matrices[-1])
            pertag_accuracies_arr[i] = self.get_per_tag_accuracies(confusion_matrices[-1])

            if flag_printaccs:
                utils.print_metrics_one(predictor.model_name, confusion_matrices[-1],
                    overall_accuracy_arr[i], pertag_accuracies_arr[i], i, self.tags_list)


        confusion_matrices = np.array(confusion_matrices)

        # return all results.
        return confusion_matrices, overall_accuracy_arr, pertag_accuracies_arr
