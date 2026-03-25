from flask import Flask, render_template, request
from model import check_email, generate_dashboard

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        email_text = request.form["email_text"]
        domain = request.form.get("domain")
        attachment = request.form.get("attachment")

        verdict, reasons = check_email(email_text, domain, attachment)
        return render_template("result.html", email=email_text, verdict=verdict, reasons=reasons)

    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    generate_dashboard()
    return render_template("dashboard.html")

@app.route("/awareness")
def awareness():
    return render_template("awareness.html")
    
if __name__ == "__main__":
    app.run(debug=True)