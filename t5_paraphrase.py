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

# model = model.to(device)

# sentence = "Which course should I take to get started in data science?"
# sentence = "What are the ingredients required to bake a perfect cake?"
# sentence = "What is the best possible approach to learn aeronautical engineering?"
# sentence = "Do apples taste better than oranges in general?"


def t5_paraphrase(sentence, device, num_return_sequences=5):
    # 'preprocessing'
    text = f"paraphrase: {sentence} </s>"
    # encode
    encoding = tokenizer.encode_plus(text, pad_to_max_length=True, return_tensors="pt")
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
    final_outputs = list()
    for beam_output in beam_outputs:
        sent = tokenizer.decode(beam_output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
        if sent.lower() != sentence.lower() and sent not in final_outputs:
            final_outputs.append(sent)
    
    return final_outputs


if __name__ == "__main__":
    print(time.time())
    sentence = "A man with a red shirt is napping under a tree and children are playing"
    final_outputs = t5_paraphrase(sentence, device)

    for i, final_output in enumerate(final_outputs):
        print(f"{i}: {final_output}")
