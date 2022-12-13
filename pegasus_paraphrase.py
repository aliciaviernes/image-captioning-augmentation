import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

device = "cuda"
modelname = 'tuner007/pegasus_paraphrase'
tokenizer = PegasusTokenizer.from_pretrained(modelname)
model = PegasusForConditionalGeneration.from_pretrained(modelname).to(device)

# note: pegasus can also take a bunch of sentences!
def pegasus(sentence, nr=10, beams=10, tokenizer=tokenizer, model=model):
    
    encoding = tokenizer([sentence],truncation=True,padding='longest',max_length=60, return_tensors="pt").to(device)
    
    output_tensors = model.generate(**encoding,max_length=60,num_beams=beams, num_return_sequences=nr, temperature=1.5)

    return tokenizer.batch_decode(output_tensors, skip_special_tokens=True)


if __name__ == "__main__":
    context = "The ultimate test of your knowledge is your capacity to convey it to another."
    response = pegasus(context, 5)
    for item in response:
        print(item)

"""
output:
['The test of your knowledge is your ability to convey it.',
 'The ability to convey your knowledge is the ultimate test of your knowledge.',
 'The ability to convey your knowledge is the most important test of your knowledge.',
 'Your capacity to convey your knowledge is the ultimate test of it.',
 'The test of your knowledge is your ability to communicate it.',
 'Your capacity to convey your knowledge is the ultimate test of your knowledge.',
 'Your capacity to convey your knowledge to another is the ultimate test of your knowledge.',
 'Your capacity to convey your knowledge is the most important test of your knowledge.',
 'The test of your knowledge is how well you can convey it.',
 'Your capacity to convey your knowledge is the ultimate test.']
"""
