import copy
import tensorflow as tf
import magenta
from magenta.models.shared import events_rnn_model
import magenta.music as mm

DEFAULT_MIN_NOTE = 48
DEFAULT_MAX_NOTE = 84
DEFAULT_TRANSPOSE_TO_KEY = 0

class MelodyRnnModel(events_rnn_model.EventSequenceRnnModel):
  def generate_melody(self, num_steps, primer_melody, temperature=1.0,beam_size=1, branch_factor=1, steps_per_iteration=1):
    melody = copy.deepcopy(primer_melody)
    transpose_amount = melody.squash(self._config.min_note,self._config.max_note,self._config.transpose_to_key)
    melody = self._generate_events(num_steps, melody, temperature, beam_size,branch_factor, steps_per_iteration)
    melody.transpose(-transpose_amount)
    return melody

  def melody_log_likelihood(self, melody):
    melody_copy = copy.deepcopy(melody)
    melody_copy.squash(self._config.min_note,self._config.max_note,self._config.transpose_to_key)
    return self._evaluate_log_likelihood([melody_copy])[0]

class MelodyRnnConfig(events_rnn_model.EventSequenceRnnConfig):
  def __init__(self, details, encoder_decoder, hparams,min_note=DEFAULT_MIN_NOTE, max_note=DEFAULT_MAX_NOTE,transpose_to_key=DEFAULT_TRANSPOSE_TO_KEY):
        super(MelodyRnnConfig, self).__init__(details, encoder_decoder, hparams)
    if min_note < mm.MIN_MIDI_PITCH:
        raise ValueError('min_note must be >= 0. min_note is %d.' % min_note)
    if max_note > mm.MAX_MIDI_PITCH + 1:
        raise ValueError('max_note must be <= 128. max_note is %d.' % max_note)
    if max_note - min_note < mm.NOTES_PER_OCTAVE:
        raise ValueError('max_note - min_note must be >= 12. min_note is %d. max_note is %d. max_note - min_note is %d.' %(min_note, max_note, max_note - min_note))
    if (transpose_to_key is not None and (transpose_to_key < 0 or transpose_to_key > mm.NOTES_PER_OCTAVE - 1)):
        raise ValueError('transpose_to_key must be >= 0 and <= 11. transpose_to_key is %d.' % transpose_to_key)
    self.min_note = min_note
    self.max_note = max_note
    self.transpose_to_key = transpose_to_key

default_configs = {
    'attention_rnn': MelodyRnnConfig(
        magenta.protobuf.generator_pb2.GeneratorDetails(
            id='attention_rnn',
            description='Melody RNN with lookback encoding and attention.'),
        magenta.music.KeyMelodyEncoderDecoder(
            min_note=DEFAULT_MIN_NOTE,
            max_note=DEFAULT_MAX_NOTE),
        tf.contrib.training.HParams(
            batch_size=128,
            rnn_layer_sizes=[128, 128],
            dropout_keep_prob=0.5,
            attn_length=40,
            clip_norm=3,
            learning_rate=0.001))
}