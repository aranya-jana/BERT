import pandas as pd
import os

def load_datasets():
    # Load datasets
    base_dir = os.path.dirname(__file__)
    cyberbullying_df = pd.read_csv(os.path.join(base_dir,"cleaned/cleaned_cyberbullying_tweets.csv"))
<<<<<<< HEAD
    hatespeech_df = pd.read_csv(os.path.join(base_dir,"cleaned/cleaned_x_hates.csv"))
=======
    hatespeech_df = pd.read_csv(os.path.join(base_dir,"cleaned_x_hates.csv"))
>>>>>>> 5c0bcbf959e5b4534d8fa89999e1533868090bf4

    # Return Merged Dataframe
    return {
        "cyberbullying_df": cyberbullying_df,
<<<<<<< HEAD
        "hatespeech_df": hatespeech_df
    }

if __name__ == "__main__":
    data_context = load_datasets()
=======
        # "hatespeech_df": hatespeech_df
    }
>>>>>>> 5c0bcbf959e5b4534d8fa89999e1533868090bf4
