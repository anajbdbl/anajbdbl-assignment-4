from flask import Flask, render_template, request, jsonify
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

app = Flask(__name__)


# TODO: Fetch dataset, initialize vectorizer and LSA here

# Fetch the dataset
newsgroups = fetch_20newsgroups(subset='all')
documents = newsgroups.data

# Preprocess and create the term-document matrix using TF-IDF
stop_words = stopwords.words('english')
vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=5000)  # Limit features to avoid high memory usage
X = vectorizer.fit_transform(documents)

# Perform LSA using TruncatedSVD
svd = TruncatedSVD(n_components=100)  # Reduce dimensionality to 100 components
X_reduced = svd.fit_transform(X)


def search_engine(query):
    """
    Function to search for top 5 similar documents given a query
    Input: query (str)
    Output: documents (list), similarities (list), indices (list)
    """
    # TODO: Implement search engine here
    # return documents, similarities, indices 

    # Transform the query using the same vectorizer and project it into the reduced space
    query_vec = vectorizer.transform([query])
    query_reduced = svd.transform(query_vec)

    # Compute cosine similarity between the query and all documents
    similarities = cosine_similarity(query_reduced, X_reduced)[0]

    # Get the indices of the top 5 most similar documents
    top_indices = np.argsort(similarities)[::-1][:5]

    # Retrieve the corresponding documents and their similarity scores
    top_documents = [documents[i] for i in top_indices]
    top_similarities = [float(similarities[i]) for i in top_indices]  # Convert to float
    top_indices = top_indices.tolist()  # Convert ndarray to list

    return top_documents, top_similarities, top_indices


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    documents, similarities, indices = search_engine(query)
    return jsonify({'documents': documents, 'similarities': similarities, 'indices': indices}) 

if __name__ == '__main__':
    app.run(debug=True)
