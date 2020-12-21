def section_separate(c = '_', n = 80):
    print(c * n)

def print_metrics_one(model_name, confmat, overall_acc, per_tag_acc, fold_i, tags):
    # print all evaluation metrics
    section_separate()

    print("Model {}: Fold {}\n".format(model_name, str(fold_i)))

    print("Confusion matrix\n")
    print(confmat)
    print("\n")

    print("Overall accuracy: % 5.2f %% \n" %(overall_acc * 100))

    print("Per tag accuracies:\n")
    for tag, acc in zip(tags, per_tag_acc.tolist()):
        print("%s\t: % 5.2f %%" %(tag, acc * 100))
    print("\n")
