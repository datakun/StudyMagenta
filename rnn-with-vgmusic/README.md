# Music-RNN for Indie Game

인디 게임 개발자를 위한 8Bit 게임기의 배경음악을 학습해서 유사한 분위기의 배경음악을 만들어주는 프로젝트와 과련된 저장소

## 예제에 사용될 미디

```
$ python3 -m venv .venv
$ source ./venv/bin/activate
$ pip install beautifulsoup4 requests
$ python get_midi.py
```

## 실행방법

```
$ python3 -m venv .venv
$ source ./venv/bin/activate
$ pip install magenta
$ git clone this REPO
$ python ./scripts/convert_dir_to_note_sequences.py --input_dir=./midi/ --output_file=notesequences.tfrecord --recursive
$ python ./scripts/melody_rnn_create_dataset.py --config=attention_rnn --input=./notesequences.tfrecord --output_dir=./out/ --eval_ratio=0.10
$ python ./scripts/melody_rnn_train.py --config=attention_rnn --run_dir=./run/ --sequence_example_file=./out/training_melodies.tfrecord --hparams="batch_size=64,rnn_layer_sizes=[13,64,64,13]"
$ python ./scripts/melody_rnn_generate --config=attention_rnn --run_dir=./run/ --output_dir=./generated-midi/ --num_outputs=10 --num_steps=480 --hparams="batch_size=64,rnn_layer_sizes=[13,64,64,13]" --primer_melody="[]"
```