import pandas as pd

# Step 1: Load both datasets
df1 = pd.read_csv('hb.csv')
df2 = pd.read_csv('ecommerce_review_dataset.csv')

# Step 2: Preprocess datasets
# Drop rows with missing values in relevant fields
df1 = df1.dropna(subset=['Review', 'Rating (Star)'])
df2 = df2.dropna(subset=['review', 'star'])

# Standardize column names for merging
df1 = df1.rename(columns={"Review": "review", "Rating (Star)": "star"})

# Ensure there are no URL columns in either dataset
if 'URL' in df1.columns:
    df1 = df1.drop(columns=['URL'])

if 'URL' in df2.columns:
    df2 = df2.drop(columns=['URL'])

df2 = df2[['review', 'star']]  # Select only relevant columns

# Combine the datasets
combined_df = pd.concat([df1, df2], ignore_index=True)

# Remove extra spaces in the 'review' column
combined_df['review'] = combined_df['review'].str.strip()

# Ensure 'star' ratings are integers
combined_df['star'] = combined_df['star'].astype(int)

# Step 3: Save the combined dataset to a CSV file
output_file = "combined_reviews.csv"
combined_df.to_csv(output_file, index=False)

print(f"Combined dataset saved to {output_file}")
