from nltk.corpus import brown
from HMM.hmm import HMM
from SVM.svm import SVM
from Bi_LSTM.bilstm import BiLSTM
from experiment_setup import ExperimentSetup as experiment
import utils
import exceptions
import numpy as np
import argparse

def process_results(name, tags, confusion_matrices, overall_accuracies_arr, per_tag_accuracies_arr, fold_k = 5):
    for i in range(fold_k):
        utils.section_separate()

        print("Model {}: Fold {}\n".format(name, str(i)))

        print("Confusion matrix\n")
        print(confusion_matrices[i])
        print("\n")

        print("Overall accuracy: % 5.2f %% \n" %(overall_accuracies_arr[i] * 100))

        print("Per tag accuracies:\n")
        for tag, acc in zip(tags, per_tag_accuracies_arr[i].tolist()):
            print("%s\t: % 5.2f %%" %(tag, acc * 100))
        print("\n")

    # average numerical results over 5 folds.
    utils.section_separate()

    avg_overall_acc = np.mean(overall_accuracies_arr)
    print("Average overall accuracy: % 5.2f %%\n" %(avg_overall_acc * 100))

    for tag, acc in zip(tags, np.mean(per_tag_accuracies_arr, axis = 0)):
        print("%s\t: % 5.2f %%"  %(tag, acc * 100))



def main():
    defined_models = {"hmm":HMM, "bilstm":BiLSTM, "svm":SVM}

    parser = argparse.ArgumentParser(description="Perform pos-tagging using various models.")
    parser.add_argument("model", help="Pos tagger to use: one of {hmm, svm, bilstm}")

    args = parser.parse_args()

    if args.model not in defined_models:
        raise exceptions.ModelNotDefinedError("Error: Unknown model.")

    # get sentences with universal tagset tags.
    tagged_sentences = brown.tagged_sents(tagset='universal')

    # create a list of tags to supply to the experiment.
    tags = set()
    for sent in tagged_sentences:
        for _, tag in sent:
            tags.add(tag)
    tags = list(tags)
    tags.sort()

    # setup experiment.
    expt = experiment(tagged_sentences, tags)

    # run experiment
    model_to_use = defined_models[args.model]
    results = expt.run(model_to_use)

    # process results.
    # process_results("HMM POS tagger", tags, *results)

    return 0

if __name__ == "__main__":
    main()
