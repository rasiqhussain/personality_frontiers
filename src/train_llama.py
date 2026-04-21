#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch 
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.current_device())  # Should return the GPU device number


# In[2]:


import transformers
transformers.__version__


# In[3]:


import accelerate
print(accelerate.__version__)
import peft
print(peft.__version__)
import bitsandbytes
print(bitsandbytes.__version__)
import transformers
print(transformers.__version__)
import trl
print(trl.__version__)
import datasets
print(datasets.__version__)


# In[4]:


# import torch.distributed


# In[5]:


# import pandas as pd
# import os
# import json
# import torch
# from torch.utils.data import Dataset, DataLoader
# from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# import contractions
# import torch.nn as nn
# # from torch.nn.parallel import DataParallel

# from peft import get_peft_model, LoraConfig, TaskType

import pandas as pd
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
# from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    HfArgumentParser, 
    TrainingArguments, 
    pipeline, 
    logging
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import contractions
import torch.nn as nn

# Set up your Hugging Face token
os.environ["HF_TOKEN"] = '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'


# In[6]:


from trl import SFTTrainer


# In[7]:


import transformers


# In[8]:


transformers.__version__


# In[9]:


# Define file paths for JSON data
file_path_1 = "/work/users/jerryma/LLM_Psych/Data/1.json"
file_path_2 = "/work/users/jerryma/LLM_Psych/Data/2.json"
file_path_3 = "/work/users/jerryma/LLM_Psych/Data/3.json"
file_path_4 = "/work/users/jerryma/LLM_Psych/Data/4.json"
file_path_5 = "/work/users/jerryma/LLM_Psych/Data/5.json"

# Initialize an empty DataFrame to store processed data
train_data_df = pd.DataFrame()

# Loop through each file to process training data
for file in [file_path_1, file_path_2, file_path_3, file_path_4]:
    if os.path.exists(file):
        with open(file, 'r') as f:
            data = json.load(f)
            list_of_dicts = [item for item in data]
            train_data = pd.DataFrame(list_of_dicts)
        
        train_data = train_data[train_data['PNEOA_scaled'] != '.']
        scaler = StandardScaler()
        train_data['PNEOA_scaled_new'] = scaler.fit_transform(train_data['PNEOA_scaled'].to_numpy().reshape(-1, 1))
        train_data = train_data[train_data['text'] != '']
        train_data['text'] = train_data['text'].apply(lambda x: contractions.fix(x))

        train_texts = train_data['text'].astype(str).tolist()
        train_labels = train_data['PNEOA_scaled_new'].astype(float).tolist()
        train_partids = train_data['PARTID'].astype(float).tolist()
        train_data_temp = pd.DataFrame({'PARTID': train_partids, 'text': train_texts, 'labels': train_labels})

        train_data_df = pd.concat([train_data_df, train_data_temp], ignore_index=True)

test_data_df = pd.DataFrame()
file = file_path_5
if os.path.exists(file):
    with open(file, 'r') as f:
        data = json.load(f)
        list_of_dicts = [item for item in data]
        test_data = pd.DataFrame(list_of_dicts)
    
    test_data = test_data[test_data['PNEOA_scaled'] != '.']
    scaler = StandardScaler()
    test_data['PNEOA_scaled_new'] = scaler.fit_transform(test_data['PNEOA_scaled'].to_numpy().reshape(-1, 1))
    test_data = test_data[test_data['text'] != '']
    test_data['text'] = test_data['text'].apply(lambda x: contractions.fix(x))

    test_texts = test_data['text'].astype(str).tolist()
    test_labels = test_data['PNEOA_scaled_new'].astype(float).tolist()
    test_partids = test_data['PARTID'].astype(float).tolist()
    
    test_data_temp = pd.DataFrame({'PARTID': test_partids, 'text': test_texts, 'labels': test_labels})

    test_data_df = pd.concat([test_data_df, test_data_temp], ignore_index=True)

# Create a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten().to('cuda'),
            'attention_mask': encoding['attention_mask'].flatten().to('cuda'),
            'labels': torch.tensor(label, dtype=torch.float).to('cuda')
        }


# In[10]:


from huggingface_hub import login
login(token='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')


# In[11]:


# Step 2: Define Parameters and Configurations
# Model and Dataset Parameters

model_name = "meta-llama/Meta-Llama-3.1-8B"
new_model = "llama-3.1-8B-personality-detection"

# QLoRA Parameters

lora_r = 64
lora_alpha = 16
lora_dropout = 0.1

# BitsAndBytes Parameters

use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False

# Training Parameters
    # To get more information about the other parameters, check the TrainingArguments, PeftModel, and SFTTrainer documentation.

output_dir = "./results"
num_train_epochs = 40
fp16 = True
bf16 = False
per_device_train_batch_size = 4
per_device_eval_batch_size = 4
gradient_accumulation_steps = 1
gradient_checkpointing = True
max_grad_norm = 0.3
learning_rate = 2e-4
weight_decay = 0.001
optim = "paged_adamw_32bit"
lr_scheduler_type = "constant"
max_steps = -1
warmup_ratio = 0.03
group_by_length = True
save_steps = 25
logging_steps = 25

# Step 3: Load Data and Initialize Components

    # Load Dataset:

# dataset = load_dataset(dataset_name, split="train")

# 2. Configure BitsAndBytes for 4-bit Quantization:

compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

# 3. Check GPU Compatibility:

if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)


# In[ ]:





# In[12]:


# 4.Load the Llama 2 Model and Tokenizer:
from transformers import AutoModelForCausalLM
model_name = "meta-llama/Meta-Llama-3.1-8B"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map = "auto",
    quantization_config=bnb_config,
    # device_map={"": 0}
)
print(next(model.parameters()).device)


# In[13]:


model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# if tokenizer.pad_token is None:
#     print('adding padding token')
#     tokenizer.add_special_tokens({'pad_token': '[PAD]'})
#     model.resize_token_embeddings(len(tokenizer))

# Create dataset and dataloader
max_length = 512
train_dataset = CustomDataset(train_data_df['text'].tolist(), train_data_df['labels'].tolist(), tokenizer, max_length)
test_dataset = CustomDataset(test_data_df['text'].tolist(), test_data_df['labels'].tolist(), tokenizer, max_length)


# Define compute_metrics function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions
    mse = mean_squared_error(labels, preds)
    mae = mean_absolute_error(labels, preds)
    r2 = r2_score(labels, preds)
    return {
        'mse': mse,
        'mae': mae,
        'r2': r2
    }



# 5. Load LoRA Configuration:

peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)

# 6. Set Training Parameters:

training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=1,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to="tensorboard"
)

# 7. Initialize SFTTrainer:
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    peft_config=peft_config,
    dataset_text_field="instruction",
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False,
    compute_metrics=compute_metrics
)


# In[14]:


# %%time
# 8. Start Training:
print(torch.cuda.is_available())  # Should print True
trainer.train()
print('trainig finished')
torch.cuda.empty_cache()
print('cuda cache emptied')
# In[15]:


# Save model and tokenizer
model.save_pretrained("./Atext/llama3.1/")
tokenizer.save_pretrained("./Atext/llama3.1/")
print('model saved')

# In[ ]:


test_loader = DataLoader(
    test_dataset, 
    batch_size=per_device_eval_batch_size, 
    shuffle=False, 
    # num_workers=1,  # Adjust this based on your system
    # pin_memory=True
)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import gc

model.eval()  # Set the model to evaluation mode
predictions = []
true_labels = []

count = 0
with torch.no_grad():
    print(test_loader, len(test_loader))
    for batch in test_loader:
        count += 1
        print(count, batch['labels'])
        input_ids = batch['input_ids'].to('cuda')
        attention_mask = batch['attention_mask'].to('cuda')
        labels = batch['labels'].to('cuda')

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.squeeze().cpu().numpy()  # Convert to numpy array
        predictions.extend(logits)
        true_labels.extend(labels.cpu().numpy())
        
        # Clear memory
        del input_ids, attention_mask, labels, outputs
        torch.cuda.empty_cache()
        gc.collect()
    
print('my kernel survived!')
mse = mean_squared_error(true_labels, predictions)
mae = mean_absolute_error(true_labels, predictions)
r2 = r2_score(true_labels, predictions)

print(f'MSE: {mse}, MAE: {mae}, R2: {r2}')


# In[ ]:


Results.append(r2, mae, mse)
results_df = pd.DataFrame(Results, columns=['R2', 'MAE', 'MSE'])
# Calculate and print average results
avg_results = results_df.mean()
print(f"Average R2: {avg_results['R2']}, MSE: {avg_results['MSE']}, MAE: {avg_results['MAE']}")

# Define the directory where you want to save the results
print(test_file, type(test_file))
save_directory = './Models/Atext/'+str(re.findall(r'\d', str(test_file))[0])+'/'

# Save to Excel
results_df.to_excel(save_directory + 'A_results.xlsx', index=False)

# Save to CSV
results_df.to_csv(save_directory + 'A_results.csv', index=False)
# Save model, tokenizer, and configuration
model_save_path = './Atext/'+str(re.findall(r'\d', str(test_file))[0])+'/'
model.save_model(model_save_path)
files = os.listdir(model_save_path)
print(files)
model.model.save_pretrained(model_save_path)
files = os.listdir(model_save_path)
print(files)
model.tokenizer.save_pretrained(model_save_path)
files = os.listdir(model_save_path)
print(files)
model.config.save_pretrained(model_save_path)
files = os.listdir(model_save_path)
print(files)
print("Model, tokenizer, and configuration saved successfully to", model_save_path)


# # In[ ]:


# # Evaluate the model
# result, model_outputs, wrong_predictions = model.eval_model(test_data_df, r2=r2_score, mse=mean_squared_error, mae=mean_absolute_error)
# Results.append([result['r2'], result['mse'], result['mae']])


# # Convert Results to a Pandas DataFrame for easier manipulation and saving
# results_df = pd.DataFrame(Results, columns=['R2', 'MSE', 'MAE'])

# # Calculate and print average results
# avg_results = results_df.mean()
# print(f"Average R2: {avg_results['R2']}, MSE: {avg_results['MSE']}, MAE: {avg_results['MAE']}")

# # Define the directory where you want to save the results
# test_file = 'temp_test_dne'
# print(test_file, type(test_file))
# save_directory = './Models/Atext/'+str(re.findall(r'\d', str(test_file))[0])+'/'

# # Save to Excel
# results_df.to_excel(save_directory + 'A_results.xlsx', index=False)

# # Save to CSV
# results_df.to_csv(save_directory + 'A_results.csv', index=False)


# # Save model, tokenizer, and configuration
# model_save_path = './Atext/'+str(re.findall(r'\d', str(test_file))[0])+'/'
# model.save_model(model_save_path)
# files = os.listdir(model_save_path)
# print(files)
# model.model.save_pretrained(model_save_path)
# files = os.listdir(model_save_path)
# print(files)
# model.tokenizer.save_pretrained(model_save_path)
# files = os.listdir(model_save_path)
# print(files)
# model.config.save_pretrained(model_save_path)
# files = os.listdir(model_save_path)
# print(files)
# print("Model, tokenizer, and configuration saved successfully to", model_save_path)


# # In[ ]: