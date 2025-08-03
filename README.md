# 📚 Audible Insights: Intelligent Book Recommendations

A personalized book recommendation system built using machine learning, NLP, and clustering techniques. The system processes Audible catalog datasets, builds multiple recommendation models, and deploys a user-friendly application using Streamlit and AWS.

---

## 🧠 Project Overview

**Objective**  
To design and develop an intelligent recommendation system that suggests books to users based on their preferences. The solution processes and cleans data from two Audible datasets, extracts features using NLP, clusters books using similarity metrics, and applies content-based and hybrid recommendation techniques. The final system is deployed using Streamlit and hosted on AWS.

---

## 🔧 Skills & Technologies

- Python Scripting  
- Data Cleaning & Preprocessing  
- Exploratory Data Analysis (EDA)  
- Natural Language Processing (NLP)  
- Clustering (KMeans, DBSCAN)  
- Machine Learning (Scikit-learn, Surprise)  
- Recommendation Systems  
- Streamlit (UI Development)  
---

## 🗃️ Datasets

### 📁 Dataset 1: `Audible_Catalog.csv`
Contains detailed book information:
- Book Name, Author, Rating, Number of Reviews
- Price, Description, Listening Time
- Genre and Rankings

### 📁 Dataset 2: `Audible_Catalog_Advanced_Features.csv`
Provides complementary attributes:
- Book Name, Author, Rating
- Reviews, Price, Additional metadata

---

## 🧪 Project Pipeline

### 1. Data Preparation
- Merged two datasets on common keys (Book Name, Author)
- Cleaned missing values and duplicates
- Standardized columns (genre, rating, etc.)

### 2. Exploratory Data Analysis (EDA)
- Visualized rating distributions, genre frequency, author popularity
- Correlation heatmaps and trend analysis over years

### 3. NLP & Clustering
- Applied NLP on book titles/descriptions
- Vectorized text using TF-IDF
- Grouped similar books using clustering algorithms (KMeans, DBSCAN)

### 4. Recommendation System
- **Content-Based Filtering**: Based on book metadata and user preferences
- **Clustering-Based Recommender**: Based on grouped similarities
- **Hybrid Model**: Combined multiple features for improved suggestions

### 5. Application Development
- Built using **Streamlit**
- Features:
  - Input favorite genres/authors
  - Get personalized recommendations
  - Visual insights from EDA


---

## 🎯 Business Use Cases

- Personalized reading experiences  
- Data-driven book suggestions for libraries and bookstores  
- Insights for publishers and authors  
- Higher reader engagement through smart recommendations

---

## 📊 Key Visualizations

- Most popular genres (bar chart)  
- Ratings vs reviews (scatter plot)  
- Publication trends (line plot)  
- Heatmaps showing feature correlations

---

## 📌 Sample Questions Answered

- What are the top genres and authors?
- Which books are highly rated but under-reviewed?
- How can user preferences influence recommendations?
- Which books cluster together based on text similarity?

---

## ✅ Results

- Cleaned and structured merged dataset  
- Multiple recommendation models implemented  
- Streamlit web app created and deployed  
- High user engagement and relevant recommendations  
- System ready for real-world use in libraries or platforms

---


## 🚀 Installation & Running Locally

```bash
# Clone the repository
git clone https://github.com/your-username/audible-insights.git
cd audible-insights

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
