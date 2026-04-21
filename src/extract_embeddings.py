



"""# 2. Import Dependencies"""

# After installations, we import the required Python modules for data handling,
# model training, and evaluation.

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import logging
import contractions
from IPython.display import display, Javascript


import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
if torch.cuda.is_available():
    # to use GPU
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('GPU is:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")



"""# 3. Data Preparation"""

# Here, we read in the training data, preprocess it, and prepare it for training.
# This involves cleaning, merging datasets if necessary, and scaling numerical labels.

# Read in training data
train_data = pd.read_json('./data/train_data_JO.json')

# Preprocess and standardize labels
# Optional: Merge with sentence data if needed and drop redundant columns (removed)
# Example of preprocessing (adjust according to your dataset specifics)
train_data = train_data[train_data['PNEOA_scaled'] != '.']
scaler = StandardScaler()
train_data['PNEOA_scaled_new'] = scaler.fit_transform(train_data['PNEOA_scaled'].to_numpy().reshape(-1, 1))
train_data = train_data[train_data['text'] != '']
train_data['text'] = train_data['text'].apply(lambda x: contractions.fix(x))

# Prepare DataFrame for training
train_texts = train_data['text'].astype(str).tolist()
train_labels = train_data['PNEOA_scaled_new'].astype(float).tolist()
train_data_df = pd.DataFrame({'text': train_texts, 'labels': train_labels})

print(train_data_df.shape)
#train_data_df= train_data_df.iloc[0:10,:]

# Sliding window size and overlap
window_size = 500
overlap = 100

# Function to generate sliding window samples
def create_sliding_windows(text, window_size, overlap):
    windows = []
    text=text.split()
    for i in range(0, len(text), overlap):
        window = text[i:i+window_size]
        window = " ".join(window)
        windows.append(window)
    return windows

train_data_df['sliding_windows'] = train_data_df['text'].apply(lambda x: create_sliding_windows(x, window_size, overlap))
print(train_data_df.head())


"""# 4. Model Training With 5 folds (Roberta-large)"""

# Set up k-fold cross-validation to train and evaluate the model's performance
# across different subsets of the training data.

# Logging setup
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
Results = []

for train_index, val_index in kf.split(train_data_df):
    training_data = train_data_df.iloc[train_index]
    validation_data = train_data_df.iloc[val_index]

    # Define model arguments
    model_args = ClassificationArgs(
        sliding_window=True,
        use_early_stopping=True,
        early_stopping_metric="r2",
        early_stopping_metric_minimize=False,
        early_stopping_patience=5,
        num_train_epochs=50,
        learning_rate=2e-5,
        evaluate_during_training=True,
        regression=True,
        train_batch_size=16,
        eval_batch_size=8,
        evaluate_during_training_steps=1000,
        max_seq_length=512,
        no_cache=True,
        no_save=True,
        overwrite_output_dir=True,
        reprocess_input_data=True,
        gradient_accumulation_steps=2,
        save_best_model=True,
    )

    # Initialize and train the model
    model = ClassificationModel(
        "roberta",
        "roberta-large",
        num_labels=1,
        args=model_args,
        use_cuda=False
        )
    #model.train_model(training_data, eval_df=validation_data, r2=r2_score, mse=mean_squared_error, mae=mean_absolute_error)

    # Evaluate the model
    #result, model_outputs, wrong_predictions = model.eval_model(validation_data, r2=r2_score, mse=mean_squared_error, mae=mean_absolute_error)
    #Results.append([result['r2'], result['mse'], result['mae']])

# Convert Results to a Pandas DataFrame for easier manipulation and saving
#results_df = pd.DataFrame(Results, columns=['R2', 'MSE', 'MAE'])

# Calculate and print average results
#avg_results = results_df.mean()
#print(f"Average R2: {avg_results['R2']}, MSE: {avg_results['MSE']}, MAE: {avg_results['MAE']}")

# Define the directory in Google Drive where you want to save the results
save_directory = './saved_models/ModelA/'

# Save to Excel
#results_df.to_excel(save_directory + 'modelA_results.xlsx', index=False)

# Save to CSV as a text file alternative, easier for large datasets
#results_df.to_csv(save_directory + 'modelA_results.csv', index=False)

def end_colab_session():
    display(Javascript('google.colab.kernel.halt()'))

# Call this function at the end of your code
#end_colab_session()

"""# Model Training With FULL training data (Roberta-large) -- Save Layer 23 -- GPT version (** For final embeddings **)"""

## This code works -- it is reduced, but saves the embeddings from layer 23 onto Drive correctly.
# The model save code here works as well.


from simpletransformers.classification import ClassificationModel, ClassificationArgs
import numpy as np
import logging
import pandas as pd

# Initialize logging
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# Define function to extract embeddings from layer 23
def get_layer_embeddings(model, texts, layer_num=23):
    tokenizer = model.tokenizer
    device = model.device  # Get the device model is currently on

    cls_embeddings = []
    mean_embeddings = []
    print("in emnbeddings")
    for text in texts:
        m_emb=[]
        c_emb=[]
        for window in text:
            #print(len(window.split()))
            #print(window)
            inputs = tokenizer(window, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = inputs.to(device)  # Move inputs to the same device as the model
            #print("input",inputs.input_ids.shape)
            #print(inputs)
            outputs = model.model(**inputs, output_hidden_states=True)
            #print("outputs",len(outputs.hidden_states))
            #print("output2",outputs.hidden_states[0].shape)
            #print("outputs3",outputs.hidden_states[-1].shape)
            layer_embeddings = outputs.hidden_states[layer_num]
            #print("layer_embeddings", layer_embeddings.shape)
            # Average across the sequence length to get a single embedding vector per input text
            last_embeddings = layer_embeddings[:,0,:].detach().cpu().numpy()
            #avg_embeddings = layer_embeddings.mean(dim=1).detach().cpu().numpy()
            #print("last_embeddings",last_embeddings.shape)
            #print("avg_embeddings",avg_embeddings.shape)
            c_emb.append(last_embeddings.squeeze())
            #m_emb.append(avg_embeddings.squeeze())
        #print(len(window_emb))
        cls_embeddings.append(c_emb)  # Ensure it's a 1D array per text
        #mean_embeddings.append(m_emb)  # Ensure it's a 1D array per text
        print(len(cls_embeddings))
        embeddings_save_path = './saved_models/ModelA/ModelA_cls_embeddings'
        with open(embeddings_save_path, "wb") as f:
            pickle.dump(cls_embeddings, f)

    # Convert the list of 1D arrays into a 2D NumPy array
    return cls_embeddings#, mean_embeddings


# Extract embeddings for the entire dataset using the trained model
all_texts = train_data_df['sliding_windows'].tolist()  # Assuming 'text' is the column with text data
#print(len(all_texts))
#print(len(all_texts[0]))
#print(len(all_texts[1]))
#print(len(all_texts[2]))


model_save_path = "./saved_models/ModelA"
#model.model.save_pretrained(model_save_path)
#model.tokenizer.save_pretrained(model_save_path)
#model.config.save_pretrained(model_save_path)

model.model = model.model.from_pretrained(model_save_path)
model.tokenizer = model.tokenizer.from_pretrained(model_save_path)
model.config = model.config.from_pretrained(model_save_path)


print("Model, tokenizer, and configuration saved successfully to", model_save_path)

import pickle
c_embeddings = get_layer_embeddings(model, all_texts)
print(len(c_embeddings))
# Save embeddings to Google Drive
embeddings_save_path = './saved_models/ModelA/ModelA_mean_embeddings'

#with open(embeddings_save_path, "wb") as f:
#     pickle.dump(m_embeddings, f)

embeddings_save_path = './saved_models/ModelA/ModelA_cls_embeddings'
with open(embeddings_save_path, "wb") as f:
     pickle.dump(c_embeddings, f)


