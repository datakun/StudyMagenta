from flask import Flask, render_template, request, send_from_directory
from werkzeug import secure_filename

import onsets_frames

MIDI_DIRECTORY = "./"

app = Flask(__name__)
 
@app.route('/')
def render_file():
    return render_template('upload.html')

@app.route('/fileUpload', methods = ['GET', 'POST'])
def upload_file_and_run_onsets_frames():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        uploaded_file.save(secure_filename(uploaded_file.filename))

        filename = onsets_frames.inference(uploaded_file.filename)

        return send_from_directory(MIDI_DIRECTORY, filename, as_attachment=True)
 
if __name__ == '__main__':
    app.run()