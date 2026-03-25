import pandas as pd
import subprocess
import matplotlib.pyplot as plt
import seaborn as sns
import os
from flask import Flask, render_template, request # <--- ADDED THIS
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

# === Initialize Flask App ===
app = Flask(__name__) # <--- THIS IS THE "APP" RENDER IS LOOKING FOR

# === Load Dataset ===
# Using the small file or column fix we discussed
df = pd.read_csv("emails.csv")
df = df.dropna(subset=['Email Text', 'Email Type']) 

X = df['Email Text']
# Standardizing labels to 0 and 1
y = df['Email Type'].apply(lambda x: 1 if 'Phishing' in str(x) or 'Fraud' in str(x) else 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=3000, stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)

# Train model (using Naive Bayes for speed on Render)
nb_model = MultinomialNB().fit(X_train_vec, y_train)

# === Routes for the Website ===
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/awareness')
def awareness():
    return render_template('awareness.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        email_content = request.form['email_content']
        vec = vectorizer.transform([email_content])
        prediction = nb_model.predict(vec)[0]
        
        verdict = "🚨 Phishing Detected" if prediction == 1 else "✅ Safe Email"
        return render_template('index.html', prediction_text=verdict)

# === Dashboard Chart Generator ===
def generate_dashboard():
    # Ensure static folder exists
    if not os.path.exists('static'):
        os.makedirs('static')
    
    counts = df['Email Type'].value_counts()
    plt.figure(figsize=(6,4))
    sns.barplot(x=counts.index, y=counts.values, palette="coolwarm")
    plt.title("Dataset Distribution")
    plt.savefig("static/charts.png")
    plt.close()

# Start the process
generate_dashboard()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
    
