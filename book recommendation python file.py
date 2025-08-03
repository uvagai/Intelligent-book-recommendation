#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df1=pd.read_csv(r"D:/DS projects/Book recommendation/Audible_Catlog.csv")
df2=pd.read_csv(r"D:/DS projects/Book recommendation/Audible_Catlog_Advanced_Features.csv")


# In[2]:


df1.info()


# In[3]:


df2.info()


# In[4]:


df1.head()


# In[5]:


df2.head()


# In[6]:


# Merge datasets on 'Book Name' and 'Author'
merged_df = pd.merge(df1, df2, on=['Book Name', 'Author'], suffixes=('_df1', '_df2'))


# In[7]:


merged_df.head()


# In[8]:


#  Drop duplicate columns post-merge by keeping one of the duplicates
# keep the 'Rating_df1', 'Number of Reviews_df1', and 'Price_df1' and drop the others
merged_df.drop(['Rating_df2', 'Number of Reviews_df2', 'Price_df2'], axis=1, inplace=True)


# In[9]:


# Rename kept columns for clarity
merged_df.rename(columns={
    'Rating_df1': 'Rating',
    'Number of Reviews_df1': 'Number of Reviews',
    'Price_df1': 'Price'
}, inplace=True)


# In[10]:


# Drop records with missing critical fields
merged_df.dropna(subset=['Rating', 'Number of Reviews', 'Price', 'Description'], inplace=True)


# In[11]:


merged_df


# In[12]:


# Convert Listening Time to total minutes
def convert_time_to_minutes(time_str):
    try:
        parts = time_str.lower().replace('minutes', '').replace('minute', '').replace('hours', '').replace('hour', '').split('and')
        hours = int(parts[0].strip()) if len(parts) > 0 else 0
        minutes = int(parts[1].strip()) if len(parts) > 1 else 0
        return hours * 60 + minutes
    except:
        return None


# In[13]:


merged_df['Listening Time (minutes)'] = merged_df['Listening Time'].apply(convert_time_to_minutes)


# In[14]:


merged_df['Listening Time (minutes)']


# In[15]:


#  Drop  missing converted time values
merged_df.dropna(subset=['Listening Time (minutes)'], inplace=True)


# In[16]:


# Remove duplicates
merged_df.drop_duplicates(subset=['Book Name', 'Author'], inplace=True)


# In[17]:


# Final cleanup: drop unnecessary original columns
merged_df = merged_df.drop(columns=['Listening Time'])


# In[18]:


merged_df.info()


# In[19]:


merged_df # Final cleanup: drop unnecessary original columns
merged_df = merged_df.drop(columns=['Listening Time'])
fo()


# In[41]:


# Show some sample values from the original 'Ranks and Genre' column to inspect the format
sample_ranks_genre = merged_df['Ranks and Genre'].dropna().unique()[:10]
sample_ranks_genre


# In[20]:


import re

# Function to extract exact genres excluding general categories
def extract_exact_genre(rank_str):
    try:
        genres = re.findall(r'#\d+\s+in\s+([^(,]+)', rank_str)
        # Filter out generic or platform-specific categories
        filtered = [g.strip() for g in genres if "Audible Audiobooks & Originals" not in g]
        return filtered[0] if filtered else None
    except:
        return None

# Apply extraction to each row
merged_df['Genre'] = merged_df['Ranks and Genre'].apply(extract_exact_genre)

# Check value counts for genres extracted
genre_counts = merged_df['Genre'].value_counts()
merged_df[['Book Name', 'Genre']].head(), genre_counts


# In[22]:


import re

def extract_best_rank(rank_str):
    ranks = re.findall(r'#(\d+)\s+in\s+([^(,]+)', rank_str)
    if not ranks:
        return None
    # Find the minimum rank value and corresponding genre
    best = min(ranks, key=lambda x: int(x[0]))
    return int(best[0])  # Return rank number

merged_df['Rank'] = merged_df['Ranks and Genre'].apply(extract_best_rank)


# In[24]:


merged_df


# In[26]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Step 1: Create a subset with known genres and descriptions
known_genres = merged_df[merged_df['Genre'].notnull()]
unknown_genres = merged_df[merged_df['Genre'].isnull()]

# Step 2: Vectorize descriptions using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_known = vectorizer.fit_transform(known_genres['Description'])
tfidf_unknown = vectorizer.transform(unknown_genres['Description'])

# Step 3: Use Nearest Neighbors to find closest match by description
nn = NearestNeighbors(n_neighbors=1, metric='cosine')
nn.fit(tfidf_known)

# Step 4: Predict genres for missing entries
distances, indices = nn.kneighbors(tfidf_unknown)
predicted_genres = known_genres.iloc[indices.flatten()]['Genre'].values

# Step 5: Fill in the predicted genres
merged_df.loc[merged_df['Genre'].isnull(), 'Genre'] = predicted_genres

# Check genre fill stats
filled_genre_counts = merged_df['Genre'].value_counts().head(10)
merged_df[['Book Name', 'Genre']].head(), filled_genre_counts


# In[28]:


merged_df.info()


# In[30]:


# Option 2: Fill missing Rank values with a default high number (e.g., 999) to indicate unranked books
merged_df['Rank'] = merged_df['Rank'].fillna(999)

# Confirm the update
merged_df['Rank'].isnull().sum(), merged_df['Rank'].value_counts().head()


# In[32]:


# Convert 'Number of Reviews' and 'Rank' to integer type
merged_df['Number of Reviews'] = merged_df['Number of Reviews'].astype(int)
merged_df['Rank'] = merged_df['Rank'].astype(int)


# In[34]:


# Convert 'Genre' and 'Ranks and Genre' to category dtype
merged_df['Genre'] = merged_df['Genre'].astype('category')
merged_df['Ranks and Genre'] = merged_df['Ranks and Genre'].astype('category')


# In[36]:


# Check for any missing data
print(merged_df.isnull().sum())


# In[38]:


merged_df = merged_df.drop(columns=['Ranks and Genre'])


# In[40]:


merged_df.head(2)


# In[42]:


merged_df.info()


# In[44]:


merged_df.to_csv(r"D:/DS projects/Book recommendation/cleaned_dataset.csv",index=False)


# **EDA**

# **Distribution of rating across geners**

# In[50]:


import pandas as pd 
df=pd.read_csv(r"D:/DS projects/Book recommendation/cleaned_dataset.csv")


# In[52]:


df.info()


# In[54]:


import matplotlib.pyplot as plt
import seaborn as sns

# Calculate the distribution of ratings across genres
rating_distribution = df.groupby('Genre')['Rating'].mean().reset_index()

# Sort genres by average rating
rating_distribution = rating_distribution.sort_values(by='Rating', ascending=False)

# Select top 10 genres (you can adjust this number based on your dataset)
top_genres = rating_distribution.head(10)

# Plot the distribution using a bar plot
plt.figure(figsize=(12, 6))
sns.barplot(x='Rating', y='Genre', data=top_genres, palette='viridis')

# Rotate the genre labels for better readability
plt.xticks(rotation=45, ha="right")

# Add labels and title
plt.title('Top 10 Average Rating Distribution Across Genres')
plt.xlabel('Average Rating')
plt.ylabel('Genre')

# Show the plot
plt.tight_layout()  # Ensures everything fits within the figure area
plt.show()


# **The common most games**

# In[61]:


# Count the frequency of each genre
genre_counts = df['Genre'].value_counts().head(10).reset_index()

# Rename columns for clarity
genre_counts.columns = ['Genre', 'Count']

# Plot the bar chart for popular genres
plt.figure(figsize=(12, 6))
sns.barplot(x='Count', y='Genre', data=genre_counts, palette='mako')


# Rotate the genre labels for better readability
plt.xticks(rotation=45, ha="right")

# Add labels and title
plt.title('Popular Genres in the Dataset')
plt.xlabel('Count')
plt.ylabel('Genre')

# Show the plot
plt.tight_layout()
plt.show()


# **The common most authors**

# In[66]:


# Count the frequency of each author and select top 10 authors
author_counts = df['Author'].value_counts().head(10).reset_index()

# Rename columns for clarity
author_counts.columns = ['Author', 'Count']

# Plot the most common authors (top 10)
plt.figure(figsize=(12, 6))
sns.barplot(x='Count', y='Author', data=author_counts, palette='viridis')

# Rotate the author labels for better readability
plt.xticks(rotation=45, ha="right")

# Add labels and title
plt.title('Top 10 Most Common Authors')
plt.xlabel('Count')
plt.ylabel('Author')

# Show the plot
plt.tight_layout()
plt.show()


# **Relationship between book ratings and review counts**

# In[73]:


# Select relevant columns for correlation
corr_data = df[['Rating', 'Number of Reviews', 'Price', 'Listening Time (minutes)']]

# Calculate the correlation matrix
corr_matrix = corr_data.corr()

# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)

# Add title and labels
plt.title('Correlation Heatmap of Ratings, Reviews, Price, and Listening Time')

# Show the plot
plt.tight_layout()
plt.show()


# **most popular geners in the datset**

# In[76]:


# Count the frequency of each genre
genre_counts = df['Genre'].value_counts().reset_index()

# Rename columns for clarity
genre_counts.columns = ['Genre', 'Count']

# Show the top 10 most popular genres
top_genres = genre_counts.head(10)
print(top_genres)


# **Authors have the highest rated books**

# In[81]:


# Group the data by 'Author' and calculate the average rating for each author
author_ratings = df.groupby('Author')['Rating'].mean().reset_index()

# Sort the authors by their average rating in descending order
highest_rated_authors = author_ratings.sort_values(by='Rating', ascending=False).head(10)

# Display the top 10 authors with the highest-rated books
print(highest_rated_authors)


# **Average rating distribution across books**

# In[88]:


# Plotting the histogram and density plot for the Rating distribution
plt.figure(figsize=(10, 6))

# Histogram with a kernel density estimate (KDE)
sns.histplot(df['Rating'], kde=True, color='green', bins=20)

# Adding title and labels
plt.title('Distribution of Book Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')

# Show the plot
plt.tight_layout()
plt.show()


# **Ratings vary between books with different review counts**

# In[91]:


# Creating bins for the number of reviews
bins = [0, 50, 200, 500, 1000, 5000, 20000, df['Number of Reviews'].max()]
labels = ['0-50', '51-200', '201-500', '501-1000', '1001-5000', '5001-20000', '20001+']

# Assigning each book to a bin based on its number of reviews
df['Review Count Range'] = pd.cut(df['Number of Reviews'], bins=bins, labels=labels)

# Box plot to show how ratings vary across different review count ranges
plt.figure(figsize=(12, 6))
sns.boxplot(x='Review Count Range', y='Rating', data=df, palette='Set2')

# Adding title and labels
plt.title('Variation of Ratings with Number of Reviews')
plt.xlabel('Review Count Range')
plt.ylabel('Rating')

# Show the plot
plt.tight_layout()
plt.show()


# **NLP Clustering**

# In[94]:


print(df.duplicated().sum())
print(df.info())


# Convert to lowercase
# 
#  Remove punctuation and stopwords
# 
#  Apply lemmatization
# 
#  Combine Book Name and Description into a single text field
# 
# 
# 

# In[96]:


import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# In[98]:


nltk.download('stopwords')
nltk.download('wordnet')


# In[100]:


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


# In[102]:


stop_words


# In[104]:


lemmatizer


# In[106]:


def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)
    


# In[108]:


df['combined_text'] = df['Book Name'] + ' ' + df['Description']
df['cleaned_text'] = df['combined_text'].apply(preprocess_text)
df['cleaned_text'].head()


# **Feature extraction using TF-IDF**

# In[111]:


#tranform text into numerical vectors
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=1000)
X = tfidf.fit_transform(df['cleaned_text'])


# In[113]:


tfidf


# **Clustering books with K-means**

# In[118]:


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# In[120]:


# Elbow method to choose optimal k
inertia = []
k_range = range(2, 11)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X)
    inertia.append(km.inertia_)


# In[122]:


plt.plot(k_range, inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal k')
plt.show()


# In[124]:


# Fit model with chosen k
kmeans = KMeans(n_clusters=4, random_state=42)
df['cluster'] = kmeans.fit_predict(X)


# In[126]:


df['cluster'].value_counts()


# In[ ]:


#visualizing rhe cluster


# In[128]:


from sklearn.decomposition import PCA

pca = PCA(n_components=2)
components = pca.fit_transform(X.toarray())

plt.figure(figsize=(10, 6))
plt.scatter(components[:, 0], components[:, 1], c=df['cluster'], cmap='tab10')
plt.title('Book Clusters Based on Text Similarity')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.colorbar(label='Cluster ID')
plt.show()


# In[151]:


df.to_csv("D:/DS projects/Book recommendation/clustered_books.csv",index=False)


# **Content based recommendation**

# In[135]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib


# In[137]:


# Step 1: TF-IDF Vectorization on 'cleaned_text'
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['cleaned_text'])


# In[139]:


# Step 2: Compute Cosine Similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


# In[141]:


# Step 2: Compute Cosine Similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


# In[143]:


# Step 3: Create a mapping from book name to index
book_indices = pd.Series(df.index, index=df['Book Name']).drop_duplicates()


# In[149]:


# Save TF-IDF and cosine similarity for deployment
joblib.dump(tfidf, 'D:/DS projects/Book recommendation/tfidf_vectorizer.pkl')
joblib.dump(cosine_sim, 'D:/DS projects/Book recommendation/cosine_similarity_matrix.pkl')


# In[153]:


# Recommendation function
def get_content_based_recommendations(title, top_n=5):
    idx = book_indices.get(title)
    if idx is None:
        return "Book not found."
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    book_indices_top = [i[0] for i in sim_scores]
    return df[['Book Name', 'Author', 'Genre']].iloc[book_indices_top]


# In[155]:


# Example usage
recommendations = get_content_based_recommendations("Think Like a Monk: The Secret of How to Harness the Power of Positivity and Be Happy Now")
recommendations


# **Clustering-based recommendation**

# In[158]:


# Create mapping of book names to their cluster
cluster_map = df.set_index('Book Name')['cluster'].to_dict()


# In[160]:


# Define recommendation function
def get_cluster_based_recommendations(book_title, top_n=5):
    if book_title not in cluster_map:
        return "Book not found."
    
    cluster_id = cluster_map[book_title]
    cluster_books = df[df['cluster'] == cluster_id]
    
    # Exclude the input book from recommendations
    recommendations = cluster_books[cluster_books['Book Name'] != book_title]
    
    return recommendations[['Book Name', 'Author', 'Genre']].head(top_n)


# In[162]:


# Example usage
book_title = "Think Like a Monk: The Secret of How to Harness the Power of Positivity and Be Happy Now"
get_cluster_based_recommendations(book_title)


# **Hybrid based recommendation**

# In[167]:


def get_hybrid_recommendations(title, top_n=5):
    idx = book_indices.get(title)
    if idx is None:
        return "Book not found."
    
    # Step 1: Content-based similarity
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:]  # Skip self

    # Step 2: Fetch books from same cluster
    input_cluster = df.loc[idx, 'cluster']

    # Step 3: Score boost based on cluster match and rating
    recommendations = []
    for i, sim in sim_scores:
        cluster_score = 1.2 if df.loc[i, 'cluster'] == input_cluster else 1.0
        rating_score = df.loc[i, 'Rating']
        final_score = sim * cluster_score * rating_score
        recommendations.append((i, final_score))

    # Step 4: Sort and return top N
    top_recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)[:top_n]
    top_indices = [i[0] for i in top_recommendations]
    return df[['Book Name', 'Author', 'Genre', 'Rating']].iloc[top_indices]


# In[169]:


# Example usage
get_hybrid_recommendations("Think Like a Monk: The Secret of How to Harness the Power of Positivity and Be Happy Now")


# In[ ]:




