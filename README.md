This work is based on the turorial from {link}[https://github.com/huggingface/notebooks/blob/main/peft/Fine_tune_BLIP2_on_an_image_captioning_dataset_PEFT.ipynb]

Tasks:
1. Lora-based fine tuning blip2 model on nlphuji/flickr30k, an image-caption dataset
2. Evaluate on the test split

Scripts:
lora_4bit: quantize the model in 4bit using bitsandbytes
lora_half: use fp16 

test: evaluate the fine-tuned model on the test set
