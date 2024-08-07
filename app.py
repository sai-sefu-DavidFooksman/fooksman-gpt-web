from flask import Flask, request, render_template
import numpy as np
import requests
import os
import joblib
from scipy.spatial.distance import cosine

app = Flask(__name__)

# ディレクトリのベースパスを設定
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")  # 環境変数からAPIキーを取得

# Hugging Face APIのエンドポイント
DISTILROBERTA_API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/paraphrase-MiniLM-L6-v2"
GPT2_API_URL = "https://api-inference.huggingface.co/models/openai/gpt-2"

def call_huggingface_api(api_url, headers, payload, retries=3):
    for attempt in range(retries):
        try:
            response = requests.post(api_url, headers=headers, json=payload)
            response.raise_for_status()  # エラーが発生した場合に例外をスロー
            return response.json()
        except requests.exceptions.HTTPError as e:
            if response.status_code == 503:
                print(f"503 サーバーエラー: {e}, リトライ {attempt + 1} / {retries}")
            elif response.status_code == 400:
                print(f"400 クライアントエラー: {response.json()}")  # エラーメッセージを表示
            else:
                raise
        except Exception as e:
            print(f"API呼び出し中にエラーが発生しました: {e}")
            raise
    raise requests.exceptions.HTTPError(f"503 サーバーエラー: {retries}回の試行後もサービスが利用できません")

def vectorize_text(text):
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    payload = {"inputs": {"source_sentence": text}}  # 正しいフィールド名に修正
    
    # Sentence-TransformersモデルのAPI呼び出し
    response = call_huggingface_api(DISTILROBERTA_API_URL, headers, payload)
    if isinstance(response, list) and len(response) > 0 and isinstance(response[0], list):
        vector = response[0]
        return np.array(vector)
    else:
        raise ValueError("レスポンスに 'embeddings' が含まれていません")

def load_word_vectors(filename):
    try:
        filepath = os.path.join(BASE_DIR, filename)
        word_vectors = joblib.load(filepath)
        return word_vectors
    except Exception as e:
        print(f"ベクトルファイルの読み込み中にエラーが発生しました: {e}")
        return {}

def approximate_gradient(params, word_vectors, user_input_vector):
    delta = 1e-5
    gradients = np.zeros(2)
    
    for i in range(2):
        params_plus = params.copy()
        params_minus = params.copy()
        
        params_plus[4 + i] += delta
        params_minus[4 + i] -= delta
        
        vector_plus = vectorize_text(generate_text_simple(params_plus, word_vectors, user_input_vector))
        vector_minus = vectorize_text(generate_text_simple(params_minus, word_vectors, user_input_vector))
        
        gradient = np.mean(vector_plus - vector_minus) / (2 * delta)
        gradients[i] = gradient
    
    return gradients

def generate_text_simple(params, word_vectors, user_input_vector):
    closest_words = find_closest_words(user_input_vector, word_vectors, params[4:5])
    return " ".join(closest_words)

def find_closest_words(user_input_vector, word_vectors, gradients):
    closest_words = []
    gradients_broadcasted = np.tile(gradients, (user_input_vector.shape[0] // gradients.shape[0], 1)).flatten()
    for word, vector in word_vectors.items():
        distance = cosine(user_input_vector + gradients_broadcasted, vector)
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
        return "エラー: ワードベクトルがロードできません。"
    
    return generate_text_with_params(params, word_vectors, user_input_vector) 

def generate_text_with_gpt(prompt):
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    payload = {"inputs": {"prompt": prompt}}  # GPT-2モデルには正しいフィールド名を使用
    
    # GPT-2モデルのAPI呼び出し
    response = call_huggingface_api(GPT2_API_URL, headers, payload)
    if isinstance(response, list) and len(response) > 0 and 'generated_text' in response[0]:
        return response[0]['generated_text'].strip()
    else:
        raise ValueError("レスポンスに 'generated_text' が含まれていません")

def load_optimized_parameters(filename):
    try:
        filepath = os.path.join(BASE_DIR, filename)
        return np.load(filepath)
    except Exception as e:
        print(f"最適化パラメータの読み込み中にエラーが発生しました: {e}")
        return np.zeros(10)

def generate_response(user_input):
    user_input_tokens = user_input.split()
    user_input_vectors = [vectorize_text(token) for token in user_input_tokens]
    
    filename = 'optimized_params.npy'
    optimized_params = load_optimized_parameters(filename)
    
    prompts = []
    for vec in user_input_vectors:
        prompt = generate_text_from_gradient(optimized_params, vec)
        prompts.append(prompt)
    
    combined_prompt = " ".join(prompts)
    
    generated_text = generate_text_with_gpt(combined_prompt)
    
    return generated_text

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
    port = int(os.getenv("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)
