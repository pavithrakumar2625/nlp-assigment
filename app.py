from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

app = Flask(__name__)

# Function to extract common words
def get_common_words(doc1, doc2):
    # Convert documents to lowercase and extract words using regular expressions
    words_doc1 = set(re.findall(r'\b\w+\b', doc1.lower()))
    words_doc2 = set(re.findall(r'\b\w+\b', doc2.lower()))

    # Find the common words and return them sorted alphabetically
    common_words = sorted(words_doc1.intersection(words_doc2))
    return common_words

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/compare', methods=['POST'])
def compare():
    doc1 = request.form.get('doc1')
    doc2 = request.form.get('doc2')

    if not doc1 or not doc2:
        return jsonify({'error': 'Both documents must be provided!'}), 400

    # Compute TF-IDF vectors
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([doc1, doc2])

    # Compute cosine similarity
    similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

    # Extract common words between the two documents
    common_words = get_common_words(doc1, doc2)

    # Return the similarity score and common words
    return jsonify({
        'similarity': round(similarity_score[0][0] * 100, 2),
        'common_words': common_words
    })

if __name__ == '__main__':
    app.run(debug=True)

