import numpy as np
import requests
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA

# 環境変数からAPIキーを取得する
HUGGINGFACE_API_KEY = "your_huggingface_api_key"  # 実際のAPIキーに置き換えてください

# APIエンドポイントの設定
DISTILROBERTA_API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/paraphrase-MiniLM-L6-v2"
GPT2_API_URL = "https://api-inference.huggingface.co/models/openai-community/gpt2"

def call_huggingface_api(url, headers, payload):
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()

def vectorize_text(source_sentence, sentences):
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    payload = {
        "inputs": {
            "source_sentence": source_sentence,
            "sentences": sentences
        }
    }
    
    response = call_huggingface_api(DISTILROBERTA_API_URL, headers, payload)
    
    if isinstance(response, list) and len(response) > 0:
        vector = np.array(response[0])  # レスポンスからベクトルを取得
    else:
        raise ValueError("レスポンスが予期しない形式です")
    
    return vector

def load_word_vectors(filename):
    try:
        word_vectors = np.load(filename, allow_pickle=True).item()
        return word_vectors
    except Exception as e:
        print(f"ベクトルファイルの読み込み中にエラーが発生しました: {e}")
        return {}

def approximate_gradient(params, word_vectors, user_input_vector, delta=1e-5):
    vector_length = user_input_vector.shape[0]
    gradients = np.zeros(vector_length)
    
    for i in range(vector_length):
        params_plus = params.copy()
        params_minus = params.copy()
        
        params_plus[i] += delta
        params_minus[i] -= delta
        
        vector_plus = vectorize_text(generate_text_simple(params_plus, word_vectors, user_input_vector), [])
        vector_minus = vectorize_text(generate_text_simple(params_minus, word_vectors, user_input_vector), [])
        
        gradient = np.mean(vector_plus - vector_minus) / (2 * delta)
        gradients[i] = gradient
    
    return gradients

def generate_6_points_vector(user_input_vector, gradients):
    delta = 0.1
    points = []
    
    for i in range(6):
        perturbation = np.zeros(user_input_vector.shape[0])
        perturbation[i % user_input_vector.shape[0]] = delta * (i - 2.5)
        
        point = user_input_vector + perturbation
        points.append(point)
    
    return np.array(points)

def generate_text_simple(params, word_vectors, user_input_vector):
    closest_words = find_closest_words(user_input_vector, word_vectors)
    return " ".join(closest_words)

def find_closest_words(user_input_vector, word_vectors):
    closest_words = []
    
    for word, vector in word_vectors.items():
        distance = cosine(user_input_vector, vector)
        closest_words.append((distance, word))
    
    closest_words.sort()
    return [word for _, word in closest_words[:5]]

def generate_text_with_gpt(prompt):
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    payload = {"inputs": prompt}
    
    response = call_huggingface_api(GPT2_API_URL, headers, payload)
    if isinstance(response, list) and len(response) > 0 and 'generated_text' in response[0]:
        return response[0]['generated_text'].strip()
    else:
        raise ValueError("レスポンスに 'generated_text' が含まれていません")

def generate_response(user_input):
    word_vectors = load_word_vectors('word_vectors.pkl')
    if not word_vectors:
        return "エラー: ワードベクトルがロードできません。"
    
    # 仮のパラメータ
    params = np.random.rand(768)
    
    user_input_vector = vectorize_text(user_input, [""])  # ダミーのベクトル化
    gradients = approximate_gradient(params, word_vectors, user_input_vector)
    six_points_vector = generate_6_points_vector(user_input_vector, gradients)
    
    prompt = generate_text_simple(params, word_vectors, user_input_vector)
    generated_text = generate_text_with_gpt(prompt)
    
    return generated_text

# Flaskアプリケーションの設定
from flask import Flask, request, render_template

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        user_input = request.form["user_input"]
        try:
            response = generate_response(user_input)
        except requests.exceptions.HTTPError as e:
            response = f"エラーが発生しました: {e}"
        except ValueError as e:
            response = f"エラー: {e}"
        return render_template("index.html", response=response, user_input=user_input)
    return render_template("index.html", response="", user_input="")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)
