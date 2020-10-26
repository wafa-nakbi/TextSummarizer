import tensorflow
from transformers import T5Tokenizer, TFT5ForConditionalGeneration
import sys
import os

def load_model(dir_path):
    loaded_model = TFT5ForConditionalGeneration.from_pretrained(dir_path)
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    return loaded_model, tokenizer

def generate_summary(text, model, tokenizer):
    source = tokenizer.prepare_seq2seq_batch(
        src_texts=text,
        max_length=512,
        padding='max_length',
        return_tensors='tf')
    input_ids = source['input_ids']
    attention_mask = source['attention_mask']
    summary = model.generate(input_ids=input_ids,
                             attention_mask=attention_mask,
                             max_length=150,
                             num_beams=3,
                             no_repeat_ngram_size=2,
                             early_stopping=True)
    decoded_summary = tokenizer.decode(summary.numpy()[0])
    return decoded_summary

if __name__ == '__main__':

    dir_path = sys.argv[1]
    text_file = sys.argv[2]
    base_path = os.getcwd()
    model, tokenizer = load_model(os.path.join(base_path,dir_path))
    with open(text_file, 'r') as f:
        text = f.read()
    text = 'summarize: ' + text
    summary = generate_summary(text, model, tokenizer)
    print ('Summary:\n', summary )


