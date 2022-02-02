"""
Code slightly modified from:
https://towardsdatascience.com/paraphrase-any-question-with-t5-text-to-text-transfer-transformer-pretrained-model-and-cbb9e35f1555
"""

import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer
import time 

paraphraser = "ramsrigouthamg/t5_paraphraser"

def set_seed(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = T5ForConditionalGeneration.from_pretrained(paraphraser).to(device)
tokenizer = T5Tokenizer.from_pretrained(paraphraser)


def t5_paraphrase(sentence, device=device, num_return_sequences=5):
    # 'preprocessing'
    text = f"paraphrase: {sentence}"  # </s>
    # encode
    encoding = tokenizer.encode_plus(text, padding=True, return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)
    # generate 
    beam_outputs = model.generate(
        input_ids=input_ids, attention_mask=attention_masks,
        do_sample=True,
        max_length=256,
        top_k=120,
        top_p=0.98,
        early_stopping=True,
        num_return_sequences=num_return_sequences)
    # decode
    final_outputs = set()
    for beam_output in beam_outputs:
        sent = tokenizer.decode(beam_output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
        if sent.lower() != sentence.lower() and sent not in final_outputs:
            final_outputs.add(sent)
    
    return final_outputs


def t5_batchwise_paraphrase(batch_sentences, device=device, num_return_sequences=5):
    # batch_sentences = ['this is one sentence', 'and this serves as a second sentence', 'the third sentence is here']
    processed, augmented_batch = list(), list()

    for sentence in batch_sentences:
        if type(sentence) == list():
            sentence = ' '.join(sentence)
        txt = f"paraphrase: {sentence}"
        # txt = tokenizer.encode(sentence, truncation=True, max_length=512)
        processed.append(txt)
    # encode
    encoded_batch = tokenizer(processed, padding=True, add_special_tokens=True, return_tensors='pt')
    input_ids, attention_masks = encoded_batch["input_ids"].to(device), encoded_batch["attention_mask"].to(device)
    # generate 
    beam_outputs = model.generate(
        input_ids=input_ids, attention_mask=attention_masks,
        do_sample=True,
        max_length=256,
        top_k=120,
        top_p=0.98,
        early_stopping=True,
        num_return_sequences=5)
        # decode and save
    for i in range(len(beam_outputs)):
        decoded = tokenizer.decode(beam_outputs[i])
        augmented_batch.append(decoded)
    
    return augmented_batch


if __name__ == "__main__":
    
    start_time = time.time()
    
    batch_sentences = ['this is one sentence', 'and this serves as a second sentence', 'the third sentence is here']
    augmented_batch = t5_batchwise_paraphrase(batch_sentences=batch_sentences)

    print("--- %s seconds ---" % (time.time() - start_time))
    print(augmented_batch)
