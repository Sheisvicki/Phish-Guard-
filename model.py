import pandas as pd
import subprocess
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

# === Load Dataset ===
df = pd.read_csv("emails.csv")
df = df.dropna(subset=['Email Text', 'Email Type'])  # remove empty rows

X = df['Email Text']
y = df['Email Type'].map({'Safe Email': 0, 'Phishing Email': 1})  # convert to numbers

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(max_features=3000, stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train models
nb_model = MultinomialNB().fit(X_train_vec, y_train)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train_vec.toarray(), y_train)

# === Kali Tool Integrations ===
def run_exiftool(filepath):
    result = subprocess.run(['exiftool', filepath], capture_output=True, text=True)
    return result.stdout

def check_domain_age(domain):
    result = subprocess.run(['whois', domain], capture_output=True, text=True)
    if 'Creation Date' in result.stdout:
        return result.stdout
    return 'Domain info not found'

def check_dns(domain):
    result = subprocess.run(['dig', domain, 'MX', '+short'], capture_output=True, text=True)
    if result.stdout.strip():
        return 'Has valid mail server'
    return 'No mail records = suspicious!'

# === Prediction Function ===
def check_email(email_text, domain=None, attachment=None):
    vec = vectorizer.transform([email_text])
    nb_result = nb_model.predict(vec)[0]
    rf_result = rf_model.predict(vec.toarray())[0]

    verdict = "✅ Safe Email"
    reasons = []

    if nb_result == 1 or rf_result == 1:
        verdict = "🚨 Phishing Detected"
        reasons.append("AI model flagged suspicious patterns")

    if domain:
        reasons.append(check_domain_age(domain))
        reasons.append(check_dns(domain))
    if attachment:
        reasons.append(run_exiftool(attachment))

    if not reasons:
        reasons.append("No suspicious patterns found")

    return verdict, reasons

# === Dashboard Chart Generator ===
def generate_dashboard():
    counts = df['Email Type'].value_counts()
    plt.figure(figsize=(6,4))
    sns.barplot(x=counts.index, y=counts.values, palette="coolwarm")
    plt.title("Dataset Distribution: Safe vs Phishing Emails")
    plt.xlabel("Email Type")
    plt.ylabel("Count")
    plt.savefig("static/charts.png")
    plt.close()