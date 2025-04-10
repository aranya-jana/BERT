import pandas as pd
import os

base_dir = os.path.dirname(__file__)
df = pd.read_csv(os.path.join(base_dir,"cleaned_cyberbullying_tweets.csv"))

# Convert all text in the 'text' column to lowercase
df['text'] = df['text'].str.lower()

# Remove all URLs and HTML tags from the 'text' column
df['text'] = df['text'].str.replace(r'http\S+|www.\S+', '', regex=True)  # Remove URLs
df['text'] = df['text'].str.replace(r'<.*?>', '', regex=True)  # Remove HTML tags

# Remove usernames/mentions from the 'text' column
df['text'] = df['text'].str.replace(r'@\w+', '', regex=True)

# Remove whitespaces before and after the text in the 'text' column, and replace multiple spaces with a single space
df['text'] = df['text'].str.replace(r'\s+', ' ', regex=True).str.strip()

# Remove emojis from the 'text' column
df['text'] = df['text'].str.replace(r'[^\w\s,]', '', regex=True)

# # Analyze the dataset
# # Display basic information about the dataset
# print("Dataset Info:")
# print(df.info())

# # Display summary statistics for numerical columns
# print("\nSummary Statistics:")
# print(df.describe())

# # Check for missing values in the dataset
# print("\nMissing Values:")
# print(df.isnull().sum())

# # Display the distribution of the target variable (if applicable)
# if 'target' in df.columns:
#     print("\nTarget Variable Distribution:")
#     print(df['target'].value_counts())

# # Display the number of unique values in each column
# print("\nUnique Values in Each Column:")
# print(df.nunique())

# # Display unique values in each column and their counts if occurrences are more than 2
# print("\nUnique Values with Counts (Occurrences > 2):")
# for column in df.columns:
#     value_counts = df[column].value_counts()
#     filtered_counts = value_counts[value_counts > 2]
#     if not filtered_counts.empty:
#         print(f"\nColumn: {column}")
#         print(filtered_counts)

print(df.head())

# Ensure the 'cleaned' folder exists
cleaned_folder = os.path.join(base_dir, "cleaned")
os.makedirs(cleaned_folder, exist_ok=True)

# Export the DataFrame to a CSV file in the 'cleaned' folder
output_file = os.path.join(cleaned_folder, "cleaned_cyberbullying_tweets.csv")
df.to_csv(output_file, index=False)