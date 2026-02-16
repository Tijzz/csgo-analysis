from ml.tactic_classifier.classifier_trainer.single_split_trainer import train_classifier
from ml.tactic_classifier.classifier_trainer.kfold_trainer import train_classifier_kfold

if __name__ == "__main__":
    # model, dataset, history = train_classifier(
    #     graph_root_dir = "data/preprocessed/de_dust2",
    #     tactics_json_path = "data/tactic_labels/de_dust2_tactics.json",
    #     num_epochs=50,
    #     batch_size=32,
    #     learning_rate=0.001
    # )
    model, dataset, fold_results, summary, test_metrics = train_classifier_kfold(
    graph_root_dir="data/preprocessed/de_dust2",
    tactics_json_path="data/tactic_labels/de_dust2_tactics.json",
    num_epochs=50,
    k_folds=5,
    output_dir="data/results"
)
