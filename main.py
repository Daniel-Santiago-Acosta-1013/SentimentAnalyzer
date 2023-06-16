import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import nltk

nltk.download('wordnet')

class SentimentModel:
    def __init__(self):
        self.root = tk.Tk()
        self.vectorizer = CountVectorizer()
        self.model = MultinomialNB()
        self.is_trained = False
        self.setup_ui()

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, classes, cmap=plt.cm.Blues):
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=classes, yticklabels=classes,
            title='Confusion Matrix',
            ylabel='True label',
            xlabel='Predicted label')

        plt.show()

    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
        X_train_dtm = self.vectorizer.fit_transform(X_train)
        X_test_dtm = self.vectorizer.transform(X_test)
        self.model.fit(X_train_dtm, y_train)
        y_pred_class = self.model.predict(X_test_dtm)
        self.is_trained = True
        return X_test_dtm, y_test, y_pred_class, metrics.accuracy_score(y_test, y_pred_class)

    def predict(self, X):
        if not self.is_trained:
            raise Exception("Model is not trained yet!")
        X_dtm = self.vectorizer.transform(X)
        return self.model.predict(X_dtm)

    def show_stats(self, X_test, y_test):
        disp = self.plot_confusion_matrix(self.model, X_test, y_test, cmap=plt.cm.Blues)
        disp.ax_.set_title("Confusion Matrix")
        plt.show()

    def load_file(self):
        filename = filedialog.askopenfilename(filetypes=[('CSV Files', '*.csv'), ('Excel Files', '*.xlsx'), ('JSON Files', '*.json')])
        if filename:
            try:
                if filename.endswith('.csv'):
                    df = pd.read_csv(filename, encoding='utf-8')
                elif filename.endswith('.xlsx'):
                    df = pd.read_excel(filename, encoding='utf-8')
                elif filename.endswith('.json'):
                    df = pd.read_json(filename, encoding='utf-8')
                else:
                    messagebox.showerror("File Type Error", "Unsupported file type!")
                    return

                lemmatizer = WordNetLemmatizer()
                df['text'] = df['text'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))
                X_test, y_test, y_pred_class, accuracy = self.train(df['text'], df['sentiment'])
                messagebox.showinfo("Model Training", f"Model trained with accuracy: {accuracy}")
                self.show_stats(X_test, y_test)
            except Exception as e:
                messagebox.showerror("Error loading file", str(e))

    def predict_sentiment(self):
        text = self.text_entry.get('1.0', 'end')
        prediction = self.predict([text])
        messagebox.showinfo("Prediction", f"The sentiment is: {prediction[0]}")

    def setup_ui(self):
        notebook = ttk.Notebook(self.root)

        train_frame = ttk.Frame(notebook)
        notebook.add(train_frame, text='Train Model')

        load_button = ttk.Button(train_frame, text="Load CSV", command=self.load_file)
        load_button.pack()

        test_frame = ttk.Frame(notebook)
        notebook.add(test_frame, text='Test Model')

        text_label = ttk.Label(test_frame, text="Enter Text:")
        text_label.pack()
        self.text_entry = tk.Text(test_frame, width=50, height=10)
        self.text_entry.pack()
        predict_button = ttk.Button(test_frame, text="Predict Sentiment", command=self.predict_sentiment)
        predict_button.pack()

        notebook.pack(expand=1, fill='both')

    def run(self):
        self.root.mainloop()


model = SentimentModel()
model.run()