import numpy as np
import sys
import pandas as pd
import os
from sklearn.metrics import *
from pprint import pformat

def main():
    # Fixed input file names
    input_files = sys.argv[1:]
    output_file = "../results/output.csv"

    # Initialize a DataFrame to store averages
    results_list = []

    for file in input_files:
        # Read the CSV file
        df = pd.read_csv(file)

        # Drop the 'target' column before averaging
        df_without_target = df.drop(columns=["target"])

        # Add the filename (without .csv) as the 'modeltag' column
        modeltag = file.replace(".csv", "")

        # Prepare predictions and true labels for F1 score calculation
        predictions = df_without_target.idxmax(axis=1).apply(lambda x: int(x.split('_')[-1]))  # Extract predicted class
        true_labels = df["target"]  # True class labels

        # Calculate F1 scores
        f1_micro = f1_score(true_labels, predictions, average="micro")
        f1_macro = f1_score(true_labels, predictions, average="macro")

        # True positives
        cm = confusion_matrix(true_labels, predictions)

        with open(f"../results/" + os.path.basename(modeltag) + "_cm.txt", "w") as f:
            f.write(
                pformat(cm)
            )

        results_list.append(
            {"modeltag": modeltag, "F1scoremicro": f1_micro, "F1scoremacro": f1_macro}
        )
        for i in range(5):
            results_list[-1][f"TP_{i}"] = cm[i, i] / np.sum(cm, axis=1, keepdims=True)[i]


    # Combine the averages into a single DataFrame
    results_df = pd.DataFrame.from_dict(results_list)

    # Write the output to a new CSV file
    results_df.to_csv(output_file, index=False)
    print(f"Averaged CSV saved to {output_file}, with F1 scores included.")

if __name__ == "__main__":
    main()
