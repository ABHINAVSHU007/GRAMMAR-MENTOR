from flask import Flask, render_template, request, jsonify
from transformers import pipeline
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest
import re
import pickle
import tensorflow as tf
from tensorflow import keras

app = Flask(__name__)

# Initialize the summarization pipeline
summarizer = pipeline("summarization")

# Load the Spacy model
nlp = spacy.load('en_core_web_sm')

# Load the wordlist
with open('wordlist.txt', 'r', encoding="utf8") as f:
    wordlist_data = f.read()
wordlist = set(re.findall(r'\w+', wordlist_data.lower()))

def colab_1(word, allow_switches=True):
    colab_1 = set()
    colab_1.update(delete_letter(word))
    if allow_switches:
        colab_1.update(switch_(word))
    colab_1.update(replace_(word))
    colab_1.update(insert_(word))
    return colab_1

def delete_letter(word):
    delete_list = [word[:i] + word[i+1:] for i in range (len(word))]
    return delete_list

def switch_(word):
    switch_list = [word[:i] + word[i+1] + word[i] + word[i+2:] for i in range(len(word) - 1)]
    return switch_list

def replace_(word):
    replace_list = [word[:i] + l + word[i+1:] for i in range(len(word)) for l in 'abcdefghijklmnopqrstuvwxyz']
    return replace_list

def insert_(word):
    insert_list = [word[:i] + l + word[i:] for i in range(len(word)+1) for l in 'abcdefghijklmnopqrstuvwxyz']
    return insert_list

def get_corrections(input_word, word_list, n=5):
    input_word = input_word.lower()

    candidates = (
        word_list.intersection(colab_1(input_word))
        or colab_2(input_word).intersection(word_list)
        or [input_word]
    )

    return list(candidates)[:n]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/text')
def word():
    return render_template('word.html')

@app.route('/check_spelling', methods=['POST'])
def check_spelling():
    data = request.get_json()
    input_word = data['word']
    corrections = get_corrections(input_word, wordlist, n=3)
    return jsonify({"corrections": corrections})

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.get_json()
    input_text = data['text']
    
    doc = nlp(input_text)
    stopwords = list(STOP_WORDS)
    
    tokens = [token.text for token in doc]

    word_freq = {}
    for word in doc:
       if word.text.lower() not in stopwords and word.text.lower() not in punctuation:
           if word.text not in word_freq.keys():
              word_freq[word.text] = 1
           else:
              word_freq[word.text] += 1

    max_freq = max(word_freq.values())

    for word in word_freq.keys():
      word_freq[word] = word_freq[word] / max_freq

    sent_tokens = [sent for sent in doc.sents]

    sent_scores = {}
    for sent in sent_tokens:
      for word in sent:
          if word.text in word_freq.keys():
             if sent not in sent_scores.keys():
                sent_scores[sent] = word_freq[word.text]
             else:
                sent_scores[sent] += word_freq[word.text]

    select_len = int(len(sent_tokens) * 0.3)
    summary = nlargest(select_len, sent_scores, key=sent_scores.get)
    final_summary = [word.text for word in summary]
    summary = ' '.join(final_summary)

    return jsonify({"summary": summary, "original_length": len(tokens), "summary_length": len(summary.split())})

@app.route('/option_three')
def option_three():
    return render_template('option_three.html')



tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))
model = keras.models.load_model('predict.h5')

def predict_next_word(text, num_words_to_generate=1):
    for _ in range(num_words_to_generate):
        sequence = tokenizer.texts_to_sequences([text])[0]
        predicted_word_probs = model.predict([sequence], verbose=0)
        predicted_word_index = int(tf.argmax(predicted_word_probs, axis=-1))
        predicted_word = tokenizer.index_word[predicted_word_index]
        text += " " + predicted_word
    return text

@app.route('/first', methods=['GET', 'POST'])
def first():
    prediction = None
    input_text = ""  # Initialize input_text
    if request.method == 'POST':
        input_text = request.form['input_text']
        num_words_to_generate = 1
        prediction = predict_next_word(input_text, num_words_to_generate)
    return render_template('first_option.html', input_text=input_text, prediction=prediction)




@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/project')
def project():
    return render_template('project.html')



if __name__ == '__main__':
    app.run(debug=True)
