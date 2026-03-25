# 🛡️ AI-Powered Phishing Detection & Awareness Platform

A cybersecurity web application built on **Kali Linux** that uses **Machine Learning** to detect phishing emails in real time. Built with Flask, Scikit-learn, and integrated with native Kali Linux security tools.

---

## 🚀 Features

- **AI Detection** — Two ML models (Naive Bayes + Random Forest) analyze email text for phishing patterns
- **Kali Tool Integration** — Uses `whois`, `dig`, and `exiftool` for deeper domain and attachment analysis
- **Live Web Interface** — Clean, responsive Flask web app styled with Bootstrap 5
- **Dashboard** — Visual chart showing dataset distribution of safe vs phishing emails
- **Real-time Results** — Instant verdict with detailed reasoning

---

## 🖥️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3 |
| Web Framework | Flask |
| Machine Learning | Scikit-learn (Naive Bayes, Random Forest) |
| Text Vectorization | TF-IDF |
| Data Processing | Pandas |
| Visualization | Matplotlib, Seaborn |
| Frontend | Bootstrap 5, Bootstrap Icons |
| Security Tools | exiftool, whois, dig (Kali Linux) |

---

## 📂 Project Structure

```
phishing_webapp/
├── static/
│   └── charts.png          # Generated dashboard chart
├── templates/
│   ├── index.html          # Home page
│   ├── result.html         # Detection result page
|   |── awareness.html      # Description of phishing
│   └── dashboard.html      # Dashboard page
├── model.py                # ML models + Kali tool integrations
├── app.py                  # Flask web application
├── .gitignore
└── README.md
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Kali Linux
- Python 3
- pip

### 1. Clone the Repository
```bash
git clone https://github.com/lysa-on-git/ai-powered-phishing-detection.git
cd ai-powered-phishing-detection
```

### 2. Install Python Dependencies
```bash
pip install pandas --break-system-packages
pip install scikit-learn --break-system-packages
pip install matplotlib --break-system-packages
pip install seaborn --break-system-packages
pip install flask --break-system-packages
```

### 3. Install Kali Security Tools
```bash
sudo apt install exiftool -y
sudo apt install whois -y
sudo apt install dnsutils -y
```

### 4. Add the Dataset
Download the **Phishing Email Detection** dataset from [Kaggle](https://www.kaggle.com), rename it to `emails.csv` and place it in the project root.

> The dataset must have two columns: `Email Text` and `Email Type` with values `Safe Email` / `Phishing Email`.

### 5. Run the App
```bash
python3 app.py
```

Then open your browser and go to:
```
http://127.0.0.1:5000
```

---

## 🔍 How It Works

1. The user pastes an email into the web interface
2. The text is vectorized using **TF-IDF** (converts words into numbers)
3. Two ML models analyze the vectorized text:
   - **Naive Bayes** — fast, great for text classification
   - **Random Forest** — 100 decision trees voting together
4. If either model flags the email → **Phishing Detected**
5. Optionally, `whois` checks the sender domain age and `dig` checks DNS records
6. The app returns a verdict + list of reasons

---

## ⚠️ Disclaimer

This project was built for **educational purposes** as part of a cybersecurity bootcamp. It is not intended for use in production environments.

---

## 👤 Author

**globalCodeHers-Techsters**  

---