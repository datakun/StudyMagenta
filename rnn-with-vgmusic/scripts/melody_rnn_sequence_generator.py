from functools import partial
from magenta.models.melody_rnn import melody_rnn_model
import magenta.music as mm

class MelodyRnnSequenceGenerator(mm.BaseSequenceGenerator):
  def __init__(self, model, details, steps_per_quarter=4, checkpoint=None,bundle=None):
    super(MelodyRnnSequenceGenerator, self).__init__(model, details, checkpoint, bundle)
    self.steps_per_quarter = steps_per_quarter
  def _generate(self, input_sequence, generator_options):
    if len(generator_options.input_sections) > 1:
        raise mm.SequenceGeneratorException('This model supports at most one input_sections message, but got %s' %len(generator_options.input_sections))
    if len(generator_options.generate_sections) != 1:
        raise mm.SequenceGeneratorException('This model supports only 1 generate_sections message, but got %s' %len(generator_options.generate_sections))
    qpm = (input_sequence.tempos[0].qpm if input_sequence and input_sequence.tempos else mm.DEFAULT_QUARTERS_PER_MINUTE)
    steps_per_second = mm.steps_per_quarter_to_steps_per_second(self.steps_per_quarter, qpm)
    generate_section = generator_options.generate_sections[0]
    if generator_options.input_sections:
        input_section = generator_options.input_sections[0]
        primer_sequence = mm.trim_note_sequence(input_sequence, input_section.start_time, input_section.end_time)
        input_start_step = mm.quantize_to_step(input_section.start_time, steps_per_second, quantize_cutoff=0)
    else:
        primer_sequence = input_sequence
        input_start_step = 0
    last_end_time = (max(n.end_time for n in primer_sequence.notes)
                        if primer_sequence.notes else 0)
    if last_end_time > generate_section.start_time:
        raise mm.SequenceGeneratorException('start time: %s, Final note end time: %s' % (generate_section.start_time, last_end_time))
    quantized_sequence = mm.quantize_note_sequence(primer_sequence, self.steps_per_quarter)
    extracted_melodies, _ = mm.extract_melodies(quantized_sequence, search_start_step=input_start_step, min_bars=0,min_unique_pitches=1, gap_bars=float('inf'),ignore_polyphonic_notes=True)
    assert len(extracted_melodies) <= 1
    start_step = mm.quantize_to_step(
        generate_section.start_time, steps_per_second, quantize_cutoff=0)
    end_step = mm.quantize_to_step(generate_section.end_time, steps_per_second, quantize_cutoff=1.0)
    if extracted_melodies and extracted_melodies[0]:
        melody = extracted_melodies[0]
    else:
        steps_per_bar = int(mm.steps_per_bar_in_quantized_sequence(quantized_sequence))
        melody = mm.Melody([],start_step=max(0, start_step - 1),steps_per_bar=steps_per_bar,steps_per_quarter=self.steps_per_quarter)
    melody.set_length(start_step - melody.start_step)
    arg_types = {
        'temperature': lambda arg: arg.float_value,
        'beam_size': lambda arg: arg.int_value,
        'branch_factor': lambda arg: arg.int_value,
        'steps_per_iteration': lambda arg: arg.int_value
    }
    args = dict((name, value_fn(generator_options.args[name])) for name, value_fn in arg_types.items() if name in generator_options.args)
    generated_melody = self._model.generate_melody(end_step - melody.start_step, melody, **args)
    generated_sequence = generated_melody.to_sequence(qpm=qpm)
    assert (generated_sequence.total_time - generate_section.end_time) <= 1e-5
    return generated_sequence
def get_generator_map():
def create_sequence_generator(config, **kwargs):
    return MelodyRnnSequenceGenerator(melody_rnn_model.MelodyRnnModel(config), config.details,steps_per_quarter=config.steps_per_quarter, **kwargs)
    return {key: partial(create_sequence_generator, config) for (key, config) in melody_rnn_model.default_configs.items()}

