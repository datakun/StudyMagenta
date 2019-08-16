import os
import tensorflow as tf
import magenta
from magenta.models.melody_rnn import melody_rnn_config_flags
from magenta.models.shared import events_rnn_graph
from magenta.models.shared import events_rnn_train

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('run_dir', '/tmp/melody_rnn/logdir/run1','Path to the directory where checkpoints will be saved')
tf.app.flags.DEFINE_string('sequence_example_file', '','Path to TFRecord file containing tf.SequenceExample')
tf.app.flags.DEFINE_integer('num_training_steps', 0,'Leave as 0 to run until terminated manually.')
tf.app.flags.DEFINE_integer('num_eval_examples', 0,'Leave as 0 to use the entire evaluation set.')
tf.app.flags.DEFINE_integer('summary_frequency', 10,'A summary statement will be logged')
tf.app.flags.DEFINE_integer('num_checkpoints', 10, 'The number of most recent checkpoints to keep in the training directory. Keeps all if 0.')
tf.app.flags.DEFINE_boolean('eval', False,'If True, this process only evaluates the model and does not update weights.')
tf.app.flags.DEFINE_string('log', 'INFO','The threshold for what messages will be logged DEBUG, INFO, WARN, ERROR, or FATAL.')


def main(unused_argv):
  tf.logging.set_verbosity(FLAGS.log)
  if not FLAGS.run_dir:
    tf.logging.fatal('--run_dir required')
    return
  if not FLAGS.sequence_example_file:
    tf.logging.fatal('--sequence_example_file required')
    return
  sequence_example_file_paths = tf.gfile.Glob(os.path.expanduser(FLAGS.sequence_example_file))
  run_dir = os.path.expanduser(FLAGS.run_dir)
  config = melody_rnn_config_flags.config_from_flags()
  mode = 'eval' if FLAGS.eval else 'train'
  graph = events_rnn_graph.build_graph(mode, config, sequence_example_file_paths)
  train_dir = os.path.join(run_dir, 'train')
  if not os.path.exists(train_dir):
    tf.gfile.MakeDirs(train_dir)
    tf.logging.info('Train dir: %s', train_dir)
  if FLAGS.eval:
    eval_dir = os.path.join(run_dir, 'eval')
    if not os.path.exists(eval_dir):
        tf.gfile.MakeDirs(eval_dir)
    tf.logging.info('Eval dir: %s', eval_dir)
    num_batches = ((FLAGS.num_eval_examples if FLAGS.num_eval_examples else magenta.common.count_records(sequence_example_file_paths)) config.hparams.batch_size)
    events_rnn_train.run_eval(graph, train_dir, eval_dir, num_batches)
  else:
    events_rnn_train.run_training(graph, train_dir, FLAGS.num_training_steps,FLAGS.summary_frequency,checkpoints_to_keep=FLAGS.num_checkpoints)

def console_entry_point():
  tf.app.run(main)

if __name__ == '__main__':
  console_entry_point()