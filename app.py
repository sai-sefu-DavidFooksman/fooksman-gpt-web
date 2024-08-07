from flask import Flask, request, render_template
import numpy as np
import torch
from transformers import BertJapaneseTokenizer, BertModel, GPT2Tokenizer, GPT2LMHeadModel
from scipy.spatial.distance import cosine
import joblib
import os

app = Flask(__name__)

# ディレクトリのベースパスを設定
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# モデルとトークナイザーのインスタンスをグローバルに一度だけ作成
tokenizer_bert = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese')
model_bert = BertModel.from_pretrained('cl-tohoku/bert-base-japanese')
tokenizer_gpt = GPT2Tokenizer.from_pretrained('gpt2')
model_gpt = GPT2LMHeadModel.from_pretrained('gpt2')

def vectorize_text(text):
    inputs = tokenizer_bert(text, return_tensors='pt', truncation=True, padding=True)
    outputs = model_bert(**inputs)
    vector = outputs.last_hidden_state[:, 0, :].detach().numpy()
    return vector.flatten()

def generate_text_with_gpt(prompt):
    inputs = tokenizer_gpt(prompt, return_tensors='pt')
    outputs = model_gpt.generate(
        inputs['input_ids'],
        max_length=150,
        num_return_sequences=1,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        no_repeat_ngram_size=2,
        early_stopping=True,
        pad_token_id=tokenizer_gpt.eos_token_id
    )
    text = tokenizer_gpt.decode(outputs[0], skip_special_tokens=True)
    return text.strip()

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        user_input = request.form["user_input"]
        response = generate_response(user_input)
        return render_template("index.html", response=response, user_input=user_input)
    return render_template("index.html", response="", user_input="")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)
