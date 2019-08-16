import ast
import os
import time
import tensorflow as tf
import magenta
from magenta.models.melody_rnn import melody_rnn_config_flags
from magenta.models.melody_rnn import melody_rnn_model
from magenta.models.melody_rnn import melody_rnn_sequence_generator
from magenta.protobuf import generator_pb2
from magenta.protobuf import music_pb2

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('run_dir', None,'No checkpoint file used.')
tf.app.flags.DEFINE_string('checkpoint_file', None,'No checkpoint file used.')
tf.app.flags.DEFINE_string('bundle_file', None,'No bundle file used.')
tf.app.flags.DEFINE_boolean('save_generator_bundle', False,'False. Generating a sequence.)
tf.app.flags.DEFINE_string('bundle_description', None,'No bundles are used.')
tf.app.flags.DEFINE_string('output_dir', '/tmp/melody_rnn/generated','The directory where MIDI files will be saved to.')
tf.app.flags.DEFINE_integer('num_outputs', 10,'The number of melodies to generate.')
tf.app.flags.DEFINE_integer('num_steps', 128,'The total number of steps the generated melodies should be.')
tf.app.flags.DEFINE_string('primer_melody', '','Melodies will be generated from scratch.')
tf.app.flags.DEFINE_string('primer_midi', '','Primers will be generated from scratch.')
tf.app.flags.DEFINE_float('qpm', None,'The quarters per minute to play generated output at.')
tf.app.flags.DEFINE_float('temperature', 1.0,'The randomness of the generated melodies.')
tf.app.flags.DEFINE_integer('beam_size', 1,'The beam size to use for beam search when generating melodies.')
tf.app.flags.DEFINE_integer('branch_factor', 1,'The branch factor to use for beam search when generating melodies.')
tf.app.flags.DEFINE_integer('steps_per_iteration', 1,'The number of melody steps to take per beam search iteration.')
tf.app.flags.DEFINE_string('log', 'INFO','The threshold for what messages will be logged.')


def get_checkpoint():
  if FLAGS.run_dir:
    train_dir = os.path.join(os.path.expanduser(FLAGS.run_dir), 'train')
    return train_dir
  elif FLAGS.checkpoint_file:
    return os.path.expanduser(FLAGS.checkpoint_file)
  else:
    return None

def run_with_flags(generator):
    FLAGS.output_dir = os.path.expanduser(FLAGS.output_dir)
    primer_midi = None
    if FLAGS.primer_midi:
        primer_midi = os.path.expanduser(FLAGS.primer_midi)
    if not tf.gfile.Exists(FLAGS.output_dir):
        tf.gfile.MakeDirs(FLAGS.output_dir)
    primer_sequence = None
    qpm = FLAGS.qpm if FLAGS.qpm else magenta.music.DEFAULT_QUARTERS_PER_MINUTE
    if FLAGS.primer_melody:
        primer_melody = magenta.music.Melody(ast.literal_eval(FLAGS.primer_melody))
        primer_sequence = primer_melody.to_sequence(qpm=qpm)
        elif primer_midi:
            primer_sequence = magenta.music.midi_file_to_sequence_proto(primer_midi)
            if primer_sequence.tempos and primer_sequence.tempos[0].qpm:
                qpm = primer_sequence.tempos[0].qpm
            else:
                tf.logging.warning('No priming sequence specified.')
                primer_melody = magenta.music.Melody([60])
                primer_sequence = primer_melody.to_sequence(qpm=qpm)
                seconds_per_step = 60.0 / qpm / generator.steps_per_quarter
                total_seconds = FLAGS.num_steps * seconds_per_step
                generator_options = generator_pb2.GeneratorOptions()
            if primer_sequence:
                input_sequence = primer_sequence
                last_end_time = (max(n.end_time for n in primer_sequence.notes)if primer_sequence.notes else 0)
                generate_section = generator_options.generate_sections.add(start_time=last_end_time + seconds_per_step,end_time=total_seconds)
            if generate_section.start_time >= generate_section.end_time:
                tf.logging.fatal('requested: Priming sequence length: %s, Generation length requested: %s',generate_section.start_time, total_seconds)
                return
            else:
                input_sequence = music_pb2.NoteSequence()
                input_sequence.tempos.add().qpm = qpm
                generate_section = generator_options.generate_sections.add(start_time=0,end_time=total_seconds)
                generator_options.args['temperature'].float_value = FLAGS.temperature
                generator_options.args['beam_size'].int_value = FLAGS.beam_size
                generator_options.args['branch_factor'].int_value = FLAGS.branch_factor
                generator_options.args['steps_per_iteration'].int_value = FLAGS.steps_per_iteration
                tf.logging.debug('input_sequence: %s', input_sequence)
                tf.logging.debug('generator_options: %s', generator_options)
                date_and_time = time.strftime('%Y-%m-%d_%H%M%S')
                digits = len(str(FLAGS.num_outputs))
                for i in range(FLAGS.num_outputs):
                    generated_sequence = generator.generate(input_sequence, generator_options)
                    midi_filename = '%s_%s.mid' % (date_and_time, str(i + 1).zfill(digits))
                    midi_path = os.path.join(FLAGS.output_dir, midi_filename)
                    magenta.music.sequence_proto_to_midi_file(generated_sequence, midi_path)
                    tf.logging.info('Wrote %d MIDI files to %s',FLAGS.num_outputs, FLAGS.output_dir)

def main(unused_argv):
    tf.logging.set_verbosity(FLAGS.log)
    bundle = get_bundle()
        if bundle:
        config_id = bundle.generator_details.id
        config = melody_rnn_model.default_configs[config_id]
        config.hparams.parse(FLAGS.hparams)
    else:
        config = melody_rnn_config_flags.config_from_flags()
    generator = melody_rnn_sequence_generator.MelodyRnnSequenceGenerator(model=melody_rnn_model.MelodyRnnModel(config),details=config.details,steps_per_quarter=config.steps_per_quarter,checkpoint=get_checkpoint(),bundle=bundle)
    if FLAGS.save_generator_bundle:
        bundle_filename = os.path.expanduser(FLAGS.bundle_file)
        if FLAGS.bundle_description is None:
            tf.logging.warning('No bundle description provided.')
        tf.logging.info('Saving generator bundle to %s', bundle_filename)
        generator.create_bundle_file(bundle_filename, FLAGS.bundle_description)
    else:
        run_with_flags(generator)

def console_entry_point():
  tf.app.run(main)

if __name__ == '__main__':
  console_entry_point()

