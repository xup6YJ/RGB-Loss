import pandas as pd
import shutil
import os

def CreateCSVFile(disease_labels, csv_path = None):
    disease_columns = disease_labels
    if not os.path.exists("result"):
        os.makedirs("result")

    df = pd.DataFrame(columns=disease_columns)
    df.to_csv(csv_path + "/train_single_acc.csv", index=False)
    df.to_csv(csv_path + "/val_single_acc.csv", index=False)
    df.to_csv(csv_path + "/train_single_f1.csv", index=False)
    df.to_csv(csv_path + "/val_single_f1.csv", index=False)
    df.to_csv(csv_path + "/train_single_recall.csv", index=False)
    df.to_csv(csv_path + "/val_single_recall.csv", index=False)
    df.to_csv(csv_path + "/train_single_precision.csv", index=False)
    df.to_csv(csv_path + "/val_single_precision.csv", index=False)
    df.to_csv(csv_path + "/train_AUC.csv", index=False)
    df.to_csv(csv_path + "/val_AUC.csv", index=False)

    columns = ["acc", "f1", "recall", "precision", "mAUC", "train_loss"]
    df = pd.DataFrame(columns=columns)
    df.to_csv(csv_path + "/train_all.csv", index=False)

    columns = ["acc", "f1", "recall", "precision", "mAUC", "val_loss"]
    df = pd.DataFrame(columns=columns)
    df.to_csv(csv_path + "/val_all.csv", index=False)

    columns = ["tp", "tn", "fp", "fn"] + disease_columns
    df = pd.DataFrame(columns=columns)
    df.to_csv(csv_path + "/train_true&false.csv", index=False)
    df.to_csv(csv_path + "/val_true&false.csv", index=False)

    columns = ['miF1', 'maF1']
    df = pd.DataFrame(columns=columns)
    df.to_csv(csv_path + "/val_mi&maF1.csv", index=False)

    df = pd.DataFrame(columns=['']+disease_columns+['mean'])
    df.to_csv(csv_path + "/summary_test.csv", index=False) # for test


def MoveCSVFile(move_path):    
    os.system(f"mv result {move_path}/result")
