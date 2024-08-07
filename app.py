from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import requests
import os

app = Flask(__name__)

# Hugging Face APIのエンドポイントとトークン
HUGGING_FACE_API_URL_BERT = 'https://api-inference.huggingface.co/models/tohoku-nlp/bert-base-japanese'
HUGGING_FACE_API_URL_GPT2 = 'https://api-inference.huggingface.co/models/openai-community/gpt2'
HUGGING_FACE_API_TOKEN = 'HUGGINGFACE_API_KEY'

def vectorize_text(text):
    words = text.split(' ')
    i = 0
    formatted_words = [f"[{"MASK"}]" if j == i else word for j, word in enumerate(words)]
    formatted_text = ' '.join(formatted_words)
    headers = {"Authorization": f"Bearer {HUGGING_FACE_API_TOKEN}"}
    response = requests.post(HUGGING_FACE_API_URL_BERT, headers=headers, json={"inputs": formatted_text})
    response.raise_for_status()
    outputs = response.json()
    last_hidden_state = np.array(outputs['last_hidden_state'])
    return last_hidden_state[:, 0, :].flatten()

def load_word_vectors(filename):
    try:
        word_vectors = joblib.load(filename)
        return word_vectors
    except Exception as e:
        print(f"ベクトルファイルの読み込み中にエラーが発生しました: {e}")
        return {}

def approximate_gradient(params, word_vectors, user_input_vector):
    delta = 1e-5
    gradients = np.zeros(2)  # 勾配は2つだけ
    
    for i in range(2):  # 5番目と6番目のパラメータのみ
        params_plus = params.copy()
        params_minus = params.copy()
        
        params_plus[4 + i] += delta  # インデックスを4 + 0, 4 + 1に修正
        params_minus[4 + i] -= delta
        
        vector_plus = vectorize_text(generate_text_simple(params_plus, word_vectors, user_input_vector)) 
        vector_minus = vectorize_text(generate_text_simple(params_minus, word_vectors, user_input_vector))
        
        gradient = np.mean(vector_plus - vector_minus) / (2 * delta)
        gradients[i] = gradient
    
    return gradients

def generate_text_simple(params, word_vectors, user_input_vector):
    closest_words = find_closest_words(user_input_vector, word_vectors, params[4:5]) # params の一部を疑似勾配として使用する
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
    headers = {"Authorization": f"Bearer {HUGGING_FACE_API_TOKEN}"}
    response = requests.post(HUGGING_FACE_API_URL_GPT2, headers=headers, json={"inputs": prompt})
    response.raise_for_status()
    outputs = response.json()
    return outputs['generated_text'].strip()

def load_optimized_parameters(filename):
    try:
        return np.load(filename)
    except Exception as e:
        print(f"最適化パラメータの読み込み中にエラーが発生しました: {e}")
        return np.zeros(10)  # エラー時にはデフォルトのパラメータを返す

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
        response = generate_response(user_input)
        return response  # Return the generated text directly
    return render_template("index.html", response="", user_input="")


if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)
