from flask import Flask, request, render_template
import numpy as np
import torch
from transformers import BertTokenizer, BertModel, GPT2Tokenizer, GPT2LMHeadModel
from scipy.spatial.distance import cosine
import joblib

app = Flask(__name__)

def vectorize_text(text):
    tokenizer = BertTokenizer.from_pretrained('cl-tohoku/bert-base-japanese')
    model = BertModel.from_pretrained('cl-tohoku/bert-base-japanese')
    
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    outputs = model(**inputs)
    vector = outputs.last_hidden_state[:, 0, :].detach().numpy()
    
    return vector.flatten()

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
        
        # approximate_gradient に依存しない関数を呼び出す
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
    try:
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2LMHeadModel.from_pretrained('gpt2')
    except Exception as e:
        print(f"モデルまたはトークナイザーの読み込みに失敗しました: {e}")
        return "モデルまたはトークナイザーの読み込みに失敗しました。"

    inputs = tokenizer(prompt, return_tensors='pt')
    
    outputs = model.generate(
        inputs['input_ids'],
        max_length=150,
        num_return_sequences=1,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        no_repeat_ngram_size=2,
        early_stopping=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text.strip()

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
        return render_template("index.html", response=response, user_input=user_input)
    return render_template("index.html", response="", user_input="")

if __name__ == "__main__":
    app.run(debug=True)
