import pandas as pd


def save_result(input_text, task_type, age_group, output_text, save_result_option):
    # If the example was not helpful, then return it
    if save_result_option == 1:
        return

    # Add the instance to the CSV file
    df = pd.read_csv("data.csv")
    new_row = {
        "query": input_text,
        "task_type": task_type,
        "age_group": age_group,
        "answer": output_text
    }
    new_row_df = pd.DataFrame([new_row])
    df = pd.concat([df, new_row_df], ignore_index=True)
    df.to_csv("data.csv", index=False)
