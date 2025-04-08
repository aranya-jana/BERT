import pandas as pd
import os

def load_datasets():
    # Load datasets
    base_dir = os.path.dirname(__file__)
    cyberbullying_df = pd.read_csv(os.path.join(base_dir,"cleaned_cyberbullying_tweets.csv"))
    hatespeech_df = pd.read_csv(os.path.join(base_dir,"cleaned_x_hates.csv"))

    # Label
    cyberbullying_df['label'] = 1 - cyberbullying_df['not_cyberbullying']
    hatespeech_df['label'] = ((hatespeech_df['hatespeech'] + hatespeech_df['offensive']) > 0).astype(int)

    # Keep relevant columns
    cyberbullying_df = cyberbullying_df[['text', 'label']]
    hatespeech_df = hatespeech_df[['text', 'label']]

    # Merge
    merged_df = pd.concat([cyberbullying_df, hatespeech_df], ignore_index=True)
    merged_df.dropna(subset=['text'], inplace=True)
    merged_df = merged_df[merged_df['text'].str.strip() != ""]

    # Return Merged Dataframe
    return merged_df