import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import tkinter as tk
from tkinter import messagebox

#training the model
def train_model():
    # load the dataset
    file_path = 'spam-or-ham.csv'
    if not file_path:
        print("Please provide a valid file path.")
        return
    
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return

    # initialize the TfidfVectorizer to preprocess the text data
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(data['text'])
    y = data['label']

    # split the dataset 80% for training and 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # train the Naive Bayes Classifier
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Evaluate the Model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)


    # save our model and vectorizer
    joblib.dump(model, 'spam_classifier_model.pkl')
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

# load the model and vectorizer we save earlier
def load_model():
    global loaded_model, loaded_vectorizer
    loaded_model = joblib.load('spam_classifier_model.pkl')
    loaded_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# function to classify text from user input
def classify_text(text):
    # preprocess the text user entered
    text_transformed = loaded_vectorizer.transform([text])
    # classify the text using our model
    prediction = loaded_model.predict(text_transformed)
    
    return prediction[0]

# creating the GUI
def create_gui():
    #method for the classify button
    def on_classify():
        # get the text from the text entry widget and strip any whitespace
        input_text = text_entry.get("1.0", tk.END).strip()
        # classify the text from the user
        if input_text:
            result = classify_text(input_text)
            messagebox.showinfo("Result", f"The message is a: {result}")
        else:
            # show an error message if the user didn't enter any text
            messagebox.showwarning("Please enter some text to indentify if it as spam or ham.")

    # initialize the Tkinter window
    window = tk.Tk()
    window.title("Spam/Ham Classifier")

    # create and place the text widget
    text_label = tk.Label(window, text="Please enter your message:")
    text_label.pack()

    text_entry = tk.Text(window, height=10, width=50)
    text_entry.pack()

    # create and place the classify button
    classify_button = tk.Button(window, text="Submit", command=on_classify)
    classify_button.pack()

    # start the GUI event loop
    window.mainloop()

# main execution
if __name__ == "__main__":
    # train the model (if needed)
    # train_model()  # uncomment this line if the model is not already trained and saved (if you want to download the file)

    # load the model and vectorizer
    load_model()

    # create the GUI
    create_gui()
