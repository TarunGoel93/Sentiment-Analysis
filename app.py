import numpy as np
import pickle
from flask import Flask , request ,  render_template

app = Flask(__name__)


# @app.route('/')
# def index():
#   return render_template("index.html", data = "Hey")

# Load the trained model and vectorizer
classifier = pickle.load(open('classifier.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

@app.route("/")
def home():
  return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    # Get the text from the form
    sample_text = request.form['text']
    
    # Preprocess the text
    sample_text_tfidf = tfidf_vectorizer.transform([sample_text])
    
    # Predict sentiment
    predicted_sentiment = classifier.predict(sample_text_tfidf)
    
    # Render the result in the template
    return render_template('index.html', text=sample_text, sentiment=predicted_sentiment[0])

if __name__ == "__main__":
  app.run(debug=True)