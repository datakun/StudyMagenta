from flask import Flask, render_template, request, send_from_directory
from werkzeug import secure_filename

import onsets_frames

MIDI_DIRECTORY = "/mnt/d/wsl/magenta/app"

app = Flask(__name__)
 
@app.route('/')
def render_file():
    return render_template('upload.html')

@app.route('/fileUpload', methods = ['GET', 'POST'])
def upload_file_and_run_onsets_frames():
    if request.method == 'POST':
        f = request.files['file']
        f.save(secure_filename(f.filename))

        filename = onsets_frames.inference(f.filename)

        return send_from_directory(MIDI_DIRECTORY, filename, as_attachment=True)
 
if __name__ == '__main__':
    app.run()