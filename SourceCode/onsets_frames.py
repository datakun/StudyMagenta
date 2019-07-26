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

CHECKPOINT_DIR = '../train'

config = configs.CONFIG_MAP['onsets_frames']
hparams = config.hparams
hparams.use_cudnn = False
hparams.batch_size = 1

examples = tf.placeholder(tf.string, [None])

dataset = data.provide_batch(
    examples=examples,
    preprocess_examples=True,
    params=hparams,
    is_training=False,
    shuffle_examples=False,
    skip_n_initial_records=0)

estimator = train_util.create_estimator(
    config.model_fn, CHECKPOINT_DIR, hparams)

iterator = dataset.make_initializable_iterator()
next_record = iterator.get_next()

sess = tf.Session()

sess.run([
    tf.initializers.global_variables(),
    tf.initializers.local_variables()
])

def input_fn(params):
    del params
    return tf.data.Dataset.from_tensors(sess.run(next_record))


def inference(filename):
    wav_file = open(filename, mode='rb')
    wav_data = wav_file.read()
    wav_file.close()

    to_process = []
    print('User uploaded file "{name}" with length {length} bytes'.format(
    name=filename, length=len(wav_data)))
    example_list = list(
    audio_label_data_utils.process_record(
        wav_data=wav_data,
        ns=music_pb2.NoteSequence(),
        example_id=filename,
        min_length=0,
        max_length=-1,
        allow_empty_notesequence=True))
    to_process.append(example_list[0].SerializeToString())

    print('Processing complete for', filename)

    sess.run(iterator.initializer, {examples: to_process})

    prediction_list = list(
        estimator.predict(
            input_fn,
            yield_single_examples=False))
    assert len(prediction_list) == 1

    frame_predictions = prediction_list[0]['frame_predictions'][0]
    onset_predictions = prediction_list[0]['onset_predictions'][0]
    velocity_values = prediction_list[0]['velocity_values'][0]

    sequence_prediction = sequences_lib.pianoroll_to_note_sequence(
        frame_predictions,
        frames_per_second=data.hparams_frames_per_second(hparams),
        min_duration_ms=0,
        min_midi_pitch=constants.MIN_MIDI_PITCH,
        onset_predictions=onset_predictions,
        velocity_values=velocity_values)

    basename = os.path.split(os.path.splitext(filename)[0])[1]
    output_filename = basename + '.mid'

    midi_filename = (output_filename)
    midi_io.sequence_proto_to_midi_file(sequence_prediction, midi_filename)

    return output_filename