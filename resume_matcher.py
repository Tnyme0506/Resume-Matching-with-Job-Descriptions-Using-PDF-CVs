import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Data Preprocessing

# Load the CSV file into a DataFrame
data = pd.read_csv('resume.csv')
job_descriptions = pd.read_csv('job_description.csv', encoding='ISO-8859-1')

# Check for missing values and handle them
data.dropna(inplace=True)  # Remove rows with missing data

# Ensure the category column is formatted appropriately
data['category'] = data['category'].str.lower()

# Step 2: Text Preprocessing

# Tokenization
data['resume_content'] = data['resume_content'].apply(lambda x: word_tokenize(x.lower()))

# Remove stopwords
stop_words = set(stopwords.words('english'))
data['resume_content'] = data['resume_content'].apply(lambda x: [word for word in x if word not in stop_words])

# Stemming (or you can use lemmatization)
stemmer = PorterStemmer()
data['resume_content'] = data['resume_content'].apply(lambda x: [stemmer.stem(word) for word in x])

# Step 3: Job Descriptions

# Filter out rows with missing job descriptions
job_descriptions = job_descriptions.dropna(subset=['job_description'])

# Load a pretrained NLP model (e.g., DistilBERT)
model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

# Step 4: Candidate-Job Matching

# Embed Resume Contents
resume_embeddings = model.encode(data['resume_content'].apply(lambda x: ' '.join(x)).tolist(), convert_to_tensor=True)

# Initialize a dictionary to store top 5 matches for each job description
top_5_matches = {}

# Calculate Cosine Similarity and store top 5 matches
for i, row in job_descriptions.iterrows():
    job_description_embedding = model.encode(row['job_description'], convert_to_tensor=True)

    # Calculate cosine similarity manually
    cosine_scores = cosine_similarity(resume_embeddings, job_description_embedding.reshape(1, -1))
    cosine_scores = cosine_scores.flatten()

    # Get the indices of the top 5 matches
    top_indices = cosine_scores.argsort()[-5:][::-1]
    top_matches = [{
        'Resume_ID': data.iloc[idx]['ID'],
        'Similarity_Score': cosine_scores[idx]
    } for idx in top_indices]

    top_5_matches[(row['company_name'], row['position_title'])] = top_matches

# Step 5: Output Results to CSV

# Create a list to store the results
results_list = []

# Prepare the results data
for key, value in top_5_matches.items():
    for match in value:
        results_list.append({
            'Company_Name': key[0],
            'Position_Title': key[1],
            'Resume_ID': match['Resume_ID'],
            'Similarity_Score': match['Similarity_Score']
        })

# Create a DataFrame from the results
result_df = pd.DataFrame(results_list)

# Save the results to a CSV file
result_df.to_csv('matching_results.csv', index=False)
