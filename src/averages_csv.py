import pandas as pd
from sklearn.metrics import f1_score

def main():
    # Fixed input file names
    input_files = ["predictions1.csv", "predictions2.csv", "predictions3.csv"]
    output_file = "../results/output.csv"

    # Initialize a DataFrame to store averages
    averages_list = []

    for file in input_files:
        # Read the CSV file
        df = pd.read_csv(file)

        # Drop the 'target' column before averaging
        df_without_target = df.drop(columns=["target"])

        # Calculate the mean for each column
        averages = df_without_target.mean()

        # Add the filename (without .csv) as the 'modeltag' column
        modeltag = file.replace(".csv", "")

        # Prepare predictions and true labels for F1 score calculation
        predictions = df_without_target.idxmax(axis=1).apply(lambda x: int(x.split('_')[-1]))  # Extract predicted class
        true_labels = df["target"]  # True class labels

        # Calculate F1 scores
        f1_micro = f1_score(true_labels, predictions, average="micro")
        f1_macro = f1_score(true_labels, predictions, average="macro")

        # Add F1 scores and modeltag
        averages = pd.concat(
            [pd.Series({"modeltag": modeltag, "F1scoremicro": f1_micro, "F1scoremacro": f1_macro}), averages]
        )

        # Append to the list
        averages_list.append(averages)

    # Combine the averages into a single DataFrame
    averages_df = pd.DataFrame(averages_list)

    # Rename prediction columns to include 'TP_' prefix
    column_names = ["modeltag", "F1scoremicro", "F1scoremacro"] + [f"TP_{i}" for i in range(averages_df.shape[1] - 3)]
    averages_df.columns = column_names

    # Write the output to a new CSV file
    averages_df.to_csv(output_file, index=False)
    print(f"Averaged CSV saved to {output_file}, with F1 scores included.")

if __name__ == "__main__":
    main()
