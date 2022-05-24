from flask import Flask, redirect, url_for, render_template, Response
import cv2
import numpy as np
from VirtualPainter import VirtualPainter

app = Flask(__name__)
app.register_blueprint(VirtualPainter, url_prefix="")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/feature")
def feature():
    return render_template("feature.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

if __name__ == "__main__":
    app.run(debug=True)
