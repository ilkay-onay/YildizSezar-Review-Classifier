import pandas as pd
from sklearn.model_selection import train_test_split
import html
import re
import emoji
from tqdm import tqdm

# --- Data Loading and Initial Checks ---

def load_data(filepath):
    """Loads data, checks for nulls, and ensures 'star_rating' is integer."""
    try:
        df = pd.read_csv(filepath, names=["star_rating", "review_text"], header=0)
        df['star_rating'] = df['star_rating'].astype(int)  # Convert to integer
        print("Initial Data Shape:", df.shape)
        print("Null Counts:\n", df.isnull().sum())
        print("Class Distribution:\n", df['star_rating'].value_counts())
        return df
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        exit()

# --- Data Cleaning Function (Revised for BERT) ---

def clean_text_for_bert(text):
    """Cleans text more suitable for BERT, preserving numbers and some punctuation."""
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = re.sub(r'http\S+|www\S+|mailto:\S+|\S+@\S+', '', text)
    text = emoji.demojize(text)
    text = re.sub(r'[^\w\sğüşıöçĞÜŞİÖÇ!.,?]', '', text)  # Keep numbers and some punctuation
    text = ' '.join(text.split()).lower()
    return text

# --- Data Splitting and Saving ---

def split_and_save_data(df, cleaned=False):
    """Splits data into train, validation, and test sets, applies cleaning if needed, and saves to CSV."""
    X = df['review_text']
    y = df['star_rating']

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    if cleaned:
        tqdm.pandas(desc="Cleaning Data")
        X_train = X_train.progress_apply(clean_text_for_bert)
        X_val = X_val.progress_apply(clean_text_for_bert)
        X_test = X_test.progress_apply(clean_text_for_bert)

    for name, X_data, y_data in zip(['train', 'val', 'test'], [X_train, X_val, X_test], [y_train, y_val, y_test]):
        data = pd.DataFrame({'star_rating': y_data, 'review_text': X_data})

        if cleaned:
            data = data[data['review_text'] != '']  # Remove empty strings
            data.to_csv(f"{name}_data_cleaned.csv", index=False)
            print(f"Saved {name}_data_cleaned.csv (Empty Strings: {len(data[data['review_text'] == ''])})")
        else:
            data.to_csv(f"{name}_data_uncleaned.csv", index=False)
            print(f"Saved {name}_data_uncleaned.csv")

# --- Main Execution ---

if __name__ == "__main__":
    df = load_data("combined_reviews.csv")
    split_and_save_data(df.copy())  # Process uncleaned data
    split_and_save_data(df.copy(), cleaned=True)  # Process cleaned data