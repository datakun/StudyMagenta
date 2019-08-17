from flask import Flask, render_template, request, send_from_directory
from flask_restful import Resource, Api
from flask_restful import reqparse
from werkzeug import secure_filename
from pydub import AudioSegment
import os
import base64

import environment as env
import onsets_frames

app = Flask(__name__)
api = Api(app)

env.WAVE_DIRECTORY = os.path.join(app.root_path, 'wav')
env.MIDI_DIRECTORY = os.path.join(app.root_path, 'mid')

class CallLittleMozart(Resource):
    def post(self):
        try:
            parser = reqparse.RequestParser()
            parser.add_argument('filename', type=str)
            parser.add_argument('binaryData', type=str)
            parser.add_argument('error', type=int)
            parser.add_argument('errorMessage', type=str)
            args = parser.parse_args()

            _filename = args['filename']
            _binary_data = args['binaryData']
            _error = args['error']
            _error_message = args['errorMessage']

            # base64 문자열을 바이너리로 변환
            uploaded_filename = os.path.join(env.WAVE_DIRECTORY, secure_filename(_filename))
            uploaded_file = open(uploaded_filename, 'wb') 
            uploaded_file.write(base64.b64decode(_binary_data))
            uploaded_file.close()

            print(uploaded_filename)
        
            # mp4를 wav로 변환
            if os.path.splitext(uploaded_filename)[1] == '.mp4':
                basename = os.path.splitext(uploaded_filename)[0]
                sound = AudioSegment.from_file(uploaded_filename, "mp4")
                new_name = basename + '.wav'
                sound.export(new_name, format='wav')
                uploaded_filename = new_name

            # 예측
            output_filename = onsets_frames.inference(uploaded_filename)

            # 바이너리를 base64 문자열로 변환
            data = open(os.path.join(env.MIDI_DIRECTORY, secure_filename(output_filename)), "rb").read()
            encoded_string = base64.b64encode(data)
            encoded_string = encoded_string.decode('utf-8')

            return {'filename': output_filename, 'binaryData': encoded_string, 'error': 0, 'errorMessage': ''}
        except Exception as e:
            print(str(e))

            return {'filename': '', 'binaryData': '', 'error': 1, 'errorMessage': str(e)}

api.add_resource(CallLittleMozart, '/call-little-mozart')
 
@app.route('/')
def render_file():
    return render_template('upload.html')

@app.route('/fileUpload', methods = ['GET', 'POST'])
def upload_file_and_run_onsets_frames():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        uploaded_filename = os.path.join(env.WAVE_DIRECTORY, secure_filename(uploaded_file.filename))
        uploaded_file.save(uploaded_filename)
        
        # mp3를 wav로 변환
        if os.path.splitext(uploaded_filename)[1] == '.mp3':
            basename = os.path.splitext(uploaded_filename)[0]
            sound = AudioSegment.from_mp3(uploaded_filename)
            new_name = basename + '.wav'
            sound.export(new_name, format='wav')
            uploaded_filename = new_name

        output_filename = onsets_frames.inference(uploaded_filename)

        return send_from_directory(env.MIDI_DIRECTORY, output_filename, as_attachment=True)
 
if __name__ == '__main__':
    app.run(host='0.0.0.0')