from transformers import GPT2Config, TFGPT2LMHeadModel, GPT2Tokenizer

output_dir = "./model_hi_custom/"

tokenizer = GPT2Tokenizer.from_pretrained(output_dir)
model = TFGPT2LMHeadModel.from_pretrained(output_dir)

text = "नमस्ते "
# encoding the input text
input_ids = tokenizer.encode(text, return_tensors='tf')
# getting out output
beam_output = model.generate(
  input_ids,
  max_length = 50,
  num_beams = 5,
  temperature = 0.7,
  no_repeat_ngram_size=2,
  num_return_sequences=5
)

print("BEAM OUTPUT ", tokenizer.decode(beam_output[0]))