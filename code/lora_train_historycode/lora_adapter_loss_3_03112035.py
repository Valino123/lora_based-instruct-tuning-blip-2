# %%
# set os at top 
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = '/data1/wln/hf_cache'

import torch
from transformers import AutoProcessor, Blip2ForConditionalGeneration, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# load model and processor
processor = AutoProcessor.from_pretrained('Salesforce/blip2-opt-2.7b')
base_model = Blip2ForConditionalGeneration.from_pretrained(
    'Salesforce/blip2-opt-2.7b', 
    local_files_only=True,
    quantization_config=bnb_config
)

base_model = prepare_model_for_kbit_training(base_model)

# set training args 
training_args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_checkpointing=True,
)

# lora config
lora_config = LoraConfig(
    r=8, #8
    lora_alpha=16, #16 
    lora_dropout=0.05, #0.05
    bias="none"
)

# get model for training
adapter_model = get_peft_model(base_model, lora_config)

# %%
import copy
from datasets import load_from_disk
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
IGNORE_INDEX = -100
max_length = 64
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# def pad_to_max_length(input_ids, max_length, pad_token_ids):
#     input_ids = input_ids[:max_length]
#     padded_ids = torch.cat((input_ids, torch.tensor([pad_token_ids] * (max_length - len(input_ids)))), dim=0)
#     return padded_ids

def pad_to_max_length(input_ids, max_length, pad_token_id):
    if isinstance(input_ids, list):
        input_ids = torch.tensor(input_ids, dtype=torch.long)

    input_ids = input_ids[:max_length]

    pad_size = max_length - input_ids.shape[0]

    if pad_size > 0:
        pad_tensor = torch.full((pad_size,), pad_token_id, dtype=input_ids.dtype, device=input_ids.device)
        input_ids = torch.cat((input_ids, pad_tensor), dim=0)

    return input_ids

class ImageCaptionDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        instruct = 'A short image caption: '
        tokenized_image = self.processor(
            images = item['image'],
            return_tensors = 'pt'
        )
        # print(pixel_values_ids.shape)
        tokenized_instruct = self.processor.tokenizer(
            instruct,
            max_length=max_length,
            truncation=True,
            add_special_tokens=False
        )
        # print(instruct_ids.shape)
        tokenized_label = self.processor.tokenizer(
            item['caption'][0],
            max_length=max_length,
            truncation=True,
            add_special_tokens=False
        )

        input_ids = torch.tensor((tokenized_instruct['input_ids'] + tokenized_label['input_ids']), dtype=torch.long)
        labels = torch.tensor([IGNORE_INDEX for _ in range(len(tokenized_instruct['input_ids']))] + copy.deepcopy(tokenized_label['input_ids']), dtype=torch.long)
      
    
        return {
            'inputs': input_ids,
            'pixel_values': tokenized_image['pixel_values'],
            'labels': labels
        }
        
def collator(batch):
    input_ids = []
    pixel_values = []
    labels = []
    for item in batch:
        input_ids.append(item['inputs'])
        pixel_values.append(item['pixel_values'])
        labels.append(item['labels'])
    
    # Apply padding
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=processor.tokenizer.pad_token_id)
    pixel_values = torch.stack(pixel_values)
    labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

    return {
        'inputs': input_ids,
        'pixel_values': pixel_values.squeeze(1),
        'labels': labels
    }

# load train set
train_ds = load_from_disk('../dataset/train_dataset')

# convert a huggingface dataset type to pytorch dataset type
train_ds_pt = ImageCaptionDataset(train_ds, processor=processor)
train_dataloader = DataLoader(train_ds_pt, shuffle=True, batch_size=training_args.per_device_train_batch_size, collate_fn=collator)

# %%
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
adapter_model.train()

if training_args.gradient_checkpointing:
    adapter_model.gradient_checkpointing_enable()

loss_list=[]
loss_print = 0
for epoch in range(2):
    print("Epoch:", epoch)
    sum_loss_list = []
    for step, batch in enumerate(tqdm(train_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        input_ids = batch.pop('inputs').squeeze(1) #instruct + labels
        pixel_values = batch['pixel_values'].squeeze(1) # encoded pixel_values
        labels = batch['labels'].squeeze(1) # IGNORE_IDS + labels
       
        
        outputs = adapter_model(
            input_ids = input_ids,
            pixel_values = pixel_values, 
            labels = labels, 
        )
        loss = outputs.loss
        sum_loss_list.append(float(loss.item()))
        loss_print += loss.item()

        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        optimizer.zero_grad()
        torch.cuda.empty_cache()
        
        if (step+1) % 10 == 0: 
            generated_output = adapter_model.generate(pixel_values=pixel_values, max_new_tokens=20)
            print("Generated caption:", processor.batch_decode(generated_output, skip_special_tokens=True))
            print(loss_print/10)
            loss_print = 0.0

    avg_sum_loss = sum(sum_loss_list) / len(sum_loss_list)
    print(f"Epoch {epoch} - Avg Loss: {avg_sum_loss}")
    loss_list.append(avg_sum_loss)

# %%
from peft import PeftModel

# Assuming your model is wrapped with LoRA
adapter_model.save_pretrained("lora_adapter_loss_5_03101757")


# %%
train_ds.save_to_disk("../dataset/train_dataset")
val_ds.save_to_disk("../dataset/val_dataset")
test_ds.save_to_disk("../dataset/test_dataset")



