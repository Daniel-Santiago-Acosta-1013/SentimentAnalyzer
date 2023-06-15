import tkinter as tk
from tkinter import messagebox
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import pandas as pd

class SentimentModel:

    def __init__(self):
        self.vectorizer = CountVectorizer()
        self.model = MultinomialNB()
        self.is_trained = False

    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
        X_train_dtm = self.vectorizer.fit_transform(X_train)
        X_test_dtm = self.vectorizer.transform(X_test)
        self.model.fit(X_train_dtm, y_train)
        y_pred_class = self.model.predict(X_test_dtm)
        self.is_trained = True
        return metrics.accuracy_score(y_test, y_pred_class)

    def predict(self, X):
        if not self.is_trained:
            raise Exception("Model is not trained yet!")
        X_dtm = self.vectorizer.transform(X)
        return self.model.predict(X_dtm)

# Function to train the model
def train_model():
    df = pd.read_csv('sentiment.csv')
    model = SentimentModel()
    accuracy = model.train(df['text'], df['sentiment'])
    messagebox.showinfo("Model Training", f"Model trained with accuracy: {accuracy}")
    return model

# Function to predict sentiment
def predict_sentiment(model):
    text = text_entry.get()
    prediction = model.predict([text])
    messagebox.showinfo("Prediction", f"The sentiment is: {prediction[0]}")

# Create the main window
root = tk.Tk()

# Create a StringVar to store the text
text_entry = tk.StringVar()

# Create the training window
train_window = tk.Toplevel(root)
train_window.title("Train Model")
train_button = tk.Button(train_window, text="Train Model", command=train_model)
train_button.pack()

# Create the prediction window
predict_window = tk.Toplevel(root)
predict_window.title("Predict Sentiment")
text_label = tk.Label(predict_window, text="Enter Text:")
text_label.pack()
text_entry_widget = tk.Entry(predict_window, textvariable=text_entry)
text_entry_widget.pack()
predict_button = tk.Button(predict_window, text="Predict Sentiment", command=lambda: predict_sentiment(model))
predict_button.pack()

root.mainloop()