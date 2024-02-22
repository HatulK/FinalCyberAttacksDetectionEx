from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

app = Flask(__name__)

# Load the model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')


# Create a function to get predictions
def get_prediction(title, text):
    inputs = tokenizer(title + " " + text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=-1).item()
    logits = torch.softmax(outputs.logits, dim=-1).tolist()[0]
    return prediction, logits


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    title = request.form['title']
    text = request.form['text']
    prediction, logits = get_prediction(title, text)
    result = {'prediction': 'Real' if prediction == 1 else 'Fake', 'probability': logits[prediction]}
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)
