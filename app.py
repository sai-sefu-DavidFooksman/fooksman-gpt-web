from flask import Flask, request, render_template
import numpy as np
import requests
import os
import joblib
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA

app = Flask(__name__)

# 設定
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
DISTILROBERTA_API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/paraphrase-MiniLM-L6-v2"
GPT2_API_URL = "https://api-inference.huggingface.co/models/openai-community/gpt2"

def call_huggingface_api(url, headers, payload):
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()

def vectorize_text(source_sentence, sentences):
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    payload = {"inputs": {"source_sentence": source_sentence, "sentences": sentences}}
    response = call_huggingface_api(DISTILROBERTA_API_URL, headers, payload)
    
    if isinstance(response, list):
        return np.array(response)
    else:
        raise ValueError("Unexpected API response format")

def load_word_vectors(filename, n_components=6):
    filepath = os.path.join(BASE_DIR, filename)
    word_vectors = joblib.load(filepath)
    
    words = list(word_vectors.keys())
    vectors = np.array(list(word_vectors.values()))
    
    if vectors.shape[1] <= 1:
        return {word: vector for word, vector in zip(words, vectors)}
    
    pca = PCA(n_components=min(n_components, vectors.shape[1]))
    reduced_vectors = pca.fit_transform(vectors)
    
    return {word: vec for word, vec in zip(words, reduced_vectors)}

def approximate_gradient(params, word_vectors, user_input_vector):
    delta = 1e-5
    gradients = np.zeros(user_input_vector.shape[0])
    
    for i in range(user_input_vector.shape[0]):
        params_plus = params.copy()
        params_minus = params.copy()
        
        params_plus[i] += delta
        params_minus[i] -= delta
        
        vector_plus = vectorize_text(generate_text_simple(params_plus, word_vectors, user_input_vector), [])
        vector_minus = vectorize_text(generate_text_simple(params_minus, word_vectors, user_input_vector), [])
        
        gradients[i] = np.mean(vector_plus - vector_minus) / (2 * delta)
    
    return gradients

def generate_text_simple(params, word_vectors, user_input_vector):
    closest_words = find_closest_words(user_input_vector, word_vectors, params)
    return " ".join(closest_words)

def find_closest_words(user_input_vector, word_vectors, gradients):
    closest_words = []
    for word, vector in word_vectors.items():
        distance = cosine(user_input_vector + gradients, vector)
        closest_words.append((distance, word))
    closest_words.sort()
    return [word for _, word in closest_words[:5]]

def generate_text_with_params(params, word_vectors, user_input_vector):
    gradients = approximate_gradient(params, word_vectors, user_input_vector)
    closest_words = find_closest_words(user_input_vector, word_vectors, gradients)
    return " ".join(closest_words)

def generate_text_from_gradient(params, user_input_vector):
    word_vectors = load_word_vectors('word_vectors.pkl')
    if not word_vectors:
        return "Error: Could not load word vectors."
    return generate_text_with_params(params, word_vectors, user_input_vector) 

def generate_text_with_gpt(prompt):
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    payload = {"inputs": prompt}
    response = call_huggingface_api(GPT2_API_URL, headers, payload)
    
    if isinstance(response, list) and len(response) > 0 and 'generated_text' in response[0]:
        return response[0]['generated_text'].strip()
    else:
        raise ValueError("Response does not contain 'generated_text'")

def load_optimized_parameters(filename):
    try:
        return np.load(os.path.join(BASE_DIR, filename))
    except Exception as e:
        print(f"Error loading optimized parameters: {e}")
        return np.zeros(10)

def generate_response(user_input):
    generated_sentences = [generate_text_with_gpt(user_input) for _ in range(5)]
    user_input_vector = vectorize_text(user_input, generated_sentences)
    
    filename = 'optimized_params.npy'
    optimized_params = load_optimized_parameters(filename)
    
    if user_input_vector.shape[0] != optimized_params.shape[0]:
        if user_input_vector.shape[0] < optimized_params.shape[0]:
            user_input_vector = np.pad(user_input_vector, (0, optimized_params.shape[0] - user_input_vector.shape[0]), mode='constant')
        else:
            optimized_params = np.tile(optimized_params, int(np.ceil(user_input_vector.shape[0] / optimized_params.shape[0])))[:user_input_vector.shape[0]]
    
    prompt = generate_text_from_gradient(optimized_params, user_input_vector)
    return generate_text_with_gpt(prompt)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        user_input = request.form["user_input"]
        try:
            response = generate_response(user_input)
        except requests.exceptions.HTTPError as e:
            response = f"Error occurred: {e}"
        except ValueError as e:
            response = f"Error: {e}"
        return render_template("index.html", response=response, user_input=user_input)
    return render_template("index.html", response="", user_input="")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)
