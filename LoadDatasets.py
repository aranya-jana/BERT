import pandas as pd
import os

def load_datasets():
    # Load datasets
    base_dir = os.path.dirname(__file__)
    cyberbullying_df = pd.read_csv(os.path.join(base_dir,"cleaned/cleaned_cyberbullying_tweets.csv"))
    hatespeech_df = pd.read_csv(os.path.join(base_dir,"cleaned_x_hates.csv"))

    # Return Merged Dataframe
    return {
        "cyberbullying_df": cyberbullying_df,
        # "hatespeech_df": hatespeech_df
    }
