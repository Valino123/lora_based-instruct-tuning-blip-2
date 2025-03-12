# set os at top 
import os
import torch
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = '/data1/wln/hf_cache'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
from transformers import AutoProcessor, Blip2ForConditionalGeneration, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# load model and processor
processor = AutoProcessor.from_pretrained('Salesforce/blip2-opt-2.7b')
base_model = Blip2ForConditionalGeneration.from_pretrained(
    'Salesforce/blip2-opt-2.7b', 
    local_files_only=True,
    
)

# set training args 
training_args = TrainingArguments(
    output_dir= 'lora_half',
    num_train_epochs=2,
    per_device_train_batch_size=2,
)

# lora config
lora_config = LoraConfig(
    r=8, #8
    lora_alpha=16, #16 
    lora_dropout=0.05, #0.05
    # target_modules=["q_proj", "k_proj"],
    bias="none"
)

# get model for training
adapter_model = get_peft_model(base_model, lora_config)
adapter_model.print_trainable_parameters()

from datasets import load_from_disk
from torch.utils.data import DataLoader, Dataset
device = "cuda:0" if torch.cuda.is_available() else "cpu"
class ImageCaptionDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        encoding = self.processor(
            images = item['image'],
            return_tensors="pt"
        )
        for k,v in encoding.items():  encoding[k] = v.squeeze()
        encoding['caption'] = item['caption'][0] 

        return encoding

def collate_fn(batch):
    processed_batch = {}
    for key in batch[0].keys():
        if key != 'caption':
            processed_batch[key] = torch.stack([item[key] for item in batch])
        else:
            tokenized_caption = processor.tokenizer(
                [item[key] for item in batch],
                padding=True,
                return_tensors="pt"
            )
            processed_batch['input_ids'] = tokenized_caption['input_ids']
    
    return processed_batch



# load train set
train_ds = load_from_disk('../dataset/train_dataset')

# convert a huggingface dataset type to pytorch dataset type
train_ds_pt = ImageCaptionDataset(train_ds, processor=processor)
train_dataloader = DataLoader(train_ds_pt, shuffle=True, batch_size=training_args.per_device_train_batch_size, collate_fn=collate_fn)

from transformers import get_scheduler
from tqdm import tqdm

# Initialize optimizer and learning rate scheduler
optimizer = torch.optim.AdamW(adapter_model.parameters(), lr=1e-5, eps=1e-5)
num_training_steps = len(train_dataloader) * 2  # Assume 2 epochs
num_warmup_steps = int(num_training_steps * 0.1)  # 10% warmup

lr_scheduler = get_scheduler(
    "cosine",
    optimizer=optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps,
)
adapter_model.to(device)
adapter_model.half()
adapter_model.train()

epoch_loss = 0.0
round_loss = 0.0
for epoch in range(training_args.num_train_epochs):
    print("Epoch:", epoch)
    for step, batch in enumerate(tqdm(train_dataloader)):
        input_ids = batch['input_ids'].to(device)
        pixel_values = batch['pixel_values'].to(device, torch.float16)
        outputs = adapter_model(
            input_ids = input_ids,
            pixel_values = pixel_values, 
            labels = input_ids
        )
        
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        optimizer.zero_grad()
        torch.cuda.empty_cache()
        
        epoch_loss += loss.item()
        round_loss += loss.item()
        if (step+1) % 100 == 0: 
            # Extract logits from the model output
            logits = outputs.logits  # Shape: (batch_size, seq_len, vocab_size)
            # Get predicted token IDs by taking the argmax over the vocab dimension
            predicted_ids = torch.argmax(logits, dim=-1)  # Shape: (batch_size, seq_len)
            # Limit the number of tokens to 30 before decoding
            seq_length = 10
            predicted_ids = predicted_ids[:, :seq_length]  # Truncate to max 30 tokens
            # Convert token IDs to text using tokenizer
            decoded_output = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
            # Print generated text
            print("Generated Output:", decoded_output)

            # Print avg loss for 40 training samples
            print(round_loss/100)
            round_loss = 0.0

    
    print(f"Epoch {epoch} - Avg Loss: {epoch_loss / len(train_dataloader)}")
    epoch_loss = 0.0

adapter_model.save_pretrained(training_args.output_dir)

# nohup python lora_half.py >> ../Logs/log_$(date +"%Y-%m-%d_%H-%M-%S").log 2>&1 &

