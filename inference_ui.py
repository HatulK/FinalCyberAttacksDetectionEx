from PyQt5.QtGui import QFont
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QTextEdit, QLabel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import sys

# Load the model and tokenizer
model_path = "model"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)


# Create a function to get predictions
def get_prediction(title, text):
    inputs = tokenizer(title + " " + text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=-1).item()
    logits = torch.softmax(outputs.logits, dim=-1).tolist()[0]
    return prediction, logits


class FakeNewsDetectionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.resize(800, 600)
        self.setWindowTitle('Fake News Detection App')

        self.setStyleSheet("background-color: #333; color: #EEE;")

        vbox = QVBoxLayout()
        self.setLayout(vbox)

        self.title = QTextEdit(self)
        self.title.setFixedHeight(50)
        self.title.setFont(QFont('Arial', 14))

        self.text = QTextEdit(self)
        self.text.setFont(QFont('Arial', 12))

        self.btn = QPushButton('Predict', self)
        self.result = QLabel(self)

        vbox.addWidget(QLabel('Title:'))
        vbox.addWidget(self.title)
        vbox.addWidget(QLabel('Text:'))
        vbox.addWidget(self.text)
        vbox.addWidget(self.btn)
        vbox.addWidget(self.result)

        self.btn.clicked.connect(self.predict)

    def predict(self):
        self.result.hide()
        QTimer.singleShot(0, self.run_prediction)

    def run_prediction(self):
        title = self.title.toPlainText()
        text = self.text.toPlainText()

        prediction, logits = get_prediction(title, text)
        self.result.setText(f"Prediction: {'Real' if prediction == 1 else 'Fake'}, Probability: {logits[prediction]}")

        self.result.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = FakeNewsDetectionApp()
    ex.show()
    sys.exit(app.exec_())
