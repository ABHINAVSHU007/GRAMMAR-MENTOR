import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation 
from heapq import nlargest
from transformers import pipeline

# Initialize the summarization pipeline
summarizer = pipeline("summarization")

def summarizer(rawdocs):
    stopwords = list(STOP_WORDS)
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(rawdocs)  # Use the rawdocs argument instead of input_text
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

    return summary, doc, len(rawdocs.split(' ')), len(summary.split(' '))

input_text = """
Your input text here.
"""

summary, _, _, _ = summarizer(input_text)

# Print the summary
print(summary)
