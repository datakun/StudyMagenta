# StudyMagenta

- Magenta 라이브러리를 이용하여 피아노 연주를 (wav 파일) MIDI 파일로 추출하는 예제
  - https://github.com/tensorflow/magenta 여기를 참고함

## 준비물
- Windows에서 WSL을 설치한 뒤, Ubuntu 환경에서 Python 3.6 으로 테스트
- 학습된 모델 (Onsets and Frames 모델 이용)
  - https://magenta.tensorflow.org/onsets-frames 자세한 설명
  - [gsutil](https://cloud.google.com/storage/docs/gsutil_install?hl=ko) 을 이용하여 모델을 다운로드
    ```
    gsutil -q -m cp -R gs://magentadata/models/onsets_frames_transcription/maestro_checkpoint.zip .
    ```
  - train 디렉터리에 모델을 배치
- MIDI 파일을 재생할 떄 사용하는 pyfluidsynth 모듈은 Python 3에서 지원하지 않음
- Python 모듈 설치
  - tensorflow, magenta, flask
  
 ## 시작
 - SourceCode 디렉터리에 app.py을 빌드 및 실행
 - localhost:5000 로 접속 시 파일을 업로드 할 수 있음
 - 파일을 업로드 한 후 submit 하면 MIDI 파일 다운로드 창이 뜸
