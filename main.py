import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import pandas as pd

class SentimentModel:
    def __init__(self):
        self.root = tk.Tk()
        self.vectorizer = CountVectorizer()
        self.model = MultinomialNB()
        self.is_trained = False
        self.setup_ui()

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
                accuracy = self.train(df['text'], df['sentiment'])
                messagebox.showinfo("Model Training", f"Model trained with accuracy: {accuracy}")
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