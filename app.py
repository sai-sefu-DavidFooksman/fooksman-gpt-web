from flask import Flask, request, render_template
import numpy as np
import requests
import os
import joblib
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA

app = Flask(__name__)

# ディレクトリのベースパスを設定
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")  # 環境変数からAPIキーを取得

# Hugging Face APIのエンドポイント
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
    
    # Sentence-TransformersモデルのAPI呼び出し
    response = call_huggingface_api(DISTILROBERTA_API_URL, headers, payload)
    print("APIレスポンス:", response)  # レスポンス内容を出力して確認
    
    # レスポンスがベクトル（リストまたは配列）であると仮定
    if isinstance(response, list):
        vector = np.array(response)  # レスポンスをNumPy配列に変換
        
        # 768次元のベクトルを6次元に圧縮
        pca = PCA(n_components=6)
        reduced_vector = pca.fit_transform(vector.reshape(1, -1)).flatten()
        
    else:
        raise ValueError("レスポンスが予期しない形式です")

    print("ベクトルの形状:", reduced_vector.shape)  # ベクトルの形状を確認
    return reduced_vector

def load_word_vectors(filename, n_components=6):
    try:
        filepath = os.path.join(BASE_DIR, filename)
        word_vectors = joblib.load(filepath)
        
        # word_vectorsが辞書形式の場合、ベクトルをリストに変換
        words = list(word_vectors.keys())
        vectors = np.array(list(word_vectors.values()))
        
        # PCAの適用前にデータの形状を確認
        print("ベクトルの形状:", vectors.shape)
        
        n_samples, n_features = vectors.shape
        
        # n_componentsをサンプル数または特徴量数の最小値に設定
        if n_components > min(n_samples, n_features):
            n_components = min(n_samples, n_features)
            print(f"n_componentsがサンプル数または特徴量数を超えたため、n_componentsを{n_components}に修正しました。")
        
        # PCAを使用して次元削減
        pca = PCA(n_components=n_components)
        reduced_vectors = pca.fit_transform(vectors)
        
        # 次元削減後のベクトルを再び辞書形式に変換
        reduced_word_vectors = {word: vec for word, vec in zip(words, reduced_vectors)}
        
        return reduced_word_vectors
    except Exception as e:
        print(f"ベクトルファイルの読み込み中にエラーが発生しました: {e}")
        return {}


# 6次元に削減したword_vectorsを読み込み
reduced_word_vectors = load_word_vectors('word_vectors.pkl', n_components=6)

def approximate_gradient(params, word_vectors, user_input_vector):
    delta = 1e-5
    vector_length = user_input_vector.shape[0]
    gradients = np.zeros(vector_length)  # ベクトルの次元に合わせてゼロ配列を作成
    
    for i in range(vector_length):
        params_plus = params.copy()
        params_minus = params.copy()
        
        params_plus[4 + i] += delta
        params_minus[4 + i] -= delta
        
        vector_plus = vectorize_text(generate_text_simple(params_plus, word_vectors, user_input_vector), [])
        vector_minus = vectorize_text(generate_text_simple(params_minus, word_vectors, user_input_vector), [])
        
        gradient = np.mean(vector_plus - vector_minus) / (2 * delta)
        gradients[i] = gradient
    
    return gradients

def generate_text_simple(params, word_vectors, user_input_vector):
    closest_words = find_closest_words(user_input_vector, word_vectors, params[4:5])
    return " ".join(closest_words)

def find_closest_words(user_input_vector, word_vectors, gradients):
    closest_words = []
    # gradientsをword_vectorsの形状に合わせる
    gradients_broadcasted = np.tile(gradients, int(np.ceil(user_input_vector.shape[0] / gradients.shape[0])))[:user_input_vector.shape[0]]
    
    for word, vector in word_vectors.items():
        distance = cosine(user_input_vector + gradients_broadcasted[:user_input_vector.shape[0]], vector)
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
    payload = {"inputs": prompt}
    
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
    generated_sentences = []
    for _ in range(5):
        generated_sentence = generate_text_with_gpt(user_input)
        generated_sentences.append(generated_sentence)
    
    user_input_vector = vectorize_text(user_input, generated_sentences)
    
    print("API応答: ", user_input_vector)
    print("形状: ", user_input_vector.shape) 
    
    filename = 'optimized_params.npy'
    optimized_params = np.load(filename)
    
    print("最適化パラメータの形状: ", optimized_params.shape)
    
    # 次元の一致を確認
    if user_input_vector.shape[0] != optimized_params.shape[0]:
        if user_input_vector.shape[0] < optimized_params.shape[0]:
            padding = optimized_params.shape[0] - user_input_vector.shape[0]
            user_input_vector = np.pad(user_input_vector, (0, padding), mode='constant')
        else:
            optimized_params = np.tile(optimized_params, int(np.ceil(user_input_vector.shape[0] / optimized_params.shape[0])))[:user_input_vector.shape[0]]
    
    print("ユーザー入力ベクトルの形状: ", user_input_vector.shape) 
    
    prompts = []
    prompt = generate_text_from_gradient(optimized_params, user_input_vector)
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
            response = f"エラーが発生しました: {e}"
        return render_template("index.html", response=response)
    return render_template("index.html", response="")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)
