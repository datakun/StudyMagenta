import tensorflow as tf
import librosa
import numpy as np
import os

from magenta.common import tf_utils
from magenta.music import audio_io
import magenta.music as mm
from magenta.models.onsets_frames_transcription import audio_label_data_utils
from magenta.models.onsets_frames_transcription import configs
from magenta.models.onsets_frames_transcription import constants
from magenta.models.onsets_frames_transcription import data
from magenta.models.onsets_frames_transcription import train_util
from magenta.music import midi_io
from magenta.protobuf import music_pb2
from magenta.music import sequences_lib

import environment as env

# Checkpoint 경로 설정
CHECKPOINT_DIR = '../train'

# 하이퍼 파라미터 설정
config = configs.CONFIG_MAP['onsets_frames']
hparams = config.hparams
hparams.use_cudnn = False
hparams.batch_size = 1

# Placeholder 설정
examples = tf.placeholder(tf.string, [None])

# 배치 생성
dataset = data.provide_batch(examples=examples, preprocess_examples=True,
    params=hparams, is_training=False, shuffle_examples=False,
    skip_n_initial_records=0)

# Estimator 생성
estimator = train_util.create_estimator(
    config.model_fn, CHECKPOINT_DIR, hparams)

# Iterator 생성
iterator = dataset.make_initializable_iterator()
next_record = iterator.get_next()

# 세션 생성
sess = tf.Session()

# 세션 초기화
sess.run([tf.initializers.global_variables(), tf.initializers.local_variables()])

# 모델 데이터로 Datasets 만들기
def input_fn(params):
    del params
    return tf.data.Dataset.from_tensors(sess.run(next_record))


def inference(filename):
    # 오디오 파일(.wav) 읽기
    wav_file = open(filename, mode='rb')
    wav_data = wav_file.read()
    wav_file.close()
    
    print('User uploaded file "{name}" with length {length} bytes'.format(name=filename, length=len(wav_data)))

    # 청크로 분할 후 protobufs 포맷으로 데이터 생성
    to_process = []
    example_list = list(
    audio_label_data_utils.process_record(wav_data=wav_data, ns=music_pb2.NoteSequence(),
        example_id=filename, min_length=0, max_length=-1, allow_empty_notesequence=True))
    
    # Serialize
    to_process.append(example_list[0].SerializeToString())

    # 세션 실행
    sess.run(iterator.initializer, {examples: to_process})

    # 예측
    prediction_list = list(estimator.predict(input_fn, yield_single_examples=False))
    assert len(prediction_list) == 1

    # 예측 결과 데이터 가져오기
    frame_predictions = prediction_list[0]['frame_predictions'][0]
    onset_predictions = prediction_list[0]['onset_predictions'][0]
    velocity_values = prediction_list[0]['velocity_values'][0]

    # 예측 결과 데이터를 이용해서 미디 시퀀스 생성
    sequence_prediction = sequences_lib.pianoroll_to_note_sequence(
        frame_predictions,
        frames_per_second=data.hparams_frames_per_second(hparams),
        min_duration_ms=0,
        min_midi_pitch=constants.MIN_MIDI_PITCH,
        onset_predictions=onset_predictions,
        velocity_values=velocity_values)

    basename = os.path.split(os.path.splitext(filename)[0])[1] + '.mid'
    output_filename = os.path.join(env.MIDI_DIRECTORY, basename)

    # 미디 시퀀스를 파일로 내보내기
    midi_filename = (output_filename)
    midi_io.sequence_proto_to_midi_file(sequence_prediction, midi_filename)

    return basename