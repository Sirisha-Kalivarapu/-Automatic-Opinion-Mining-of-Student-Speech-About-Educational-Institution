from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename 
from views.audio import start_recording, stop_recording
from views.nltk import predict_text
from views.sentiment_svc import train_model

app = Flask(__name__)

# @app.route('/store-audio/', methods= ['POST', 'GET'])
# def store_audio():
#     if request.method == 'POST':
#         audio = request.files['audiofile']
#         audio.save('./views/' + secure_filename(audio.filename))
#         audio_file_to_text(secure_filename(audio.filename))
#         return "upload successfull"
#     else:
#         return redirect(url_for('index'))

@app.route('/predict', methods= ["POST", "GET"])
def prediction():
    if request.method == "POST":
        text = request.form['spoken_text']
        result = predict_text(text)
        return render_template('predict.html', result=result)
    pass

@app.route('/recording', methods= ["POST", "GET"])
def recording():
    if request.method == 'POST':
        name = request.form['record_state']
        if name == "Start":
            start_recording()
            return render_template('index.html', button_name="Stop Recording")
        elif name == "Stop":
            text = stop_recording()
            return render_template('result.html', audio_text=text)
        pass
    return redirect(url_for('index'))

@app.route('/feedback')
def index():
    return render_template('index.html', button_name="Start Recording")

if __name__ == "__main__":
    # train_model()
    app.run() 
   