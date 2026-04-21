# Import necessary libraries
import pandas as pd
import os
import sys
import torch
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import logging
import contractions
import json
import random
from sklearn.preprocessing import StandardScaler
import re

if not os.path.exists('./bdc/Atext') or not os.path.exists('./bdc/Models/Atext'):
    sys.exit()

# Define file paths for JSON data
list_of_files = ["./data/5.json"]
for test_file in list_of_files:
    train_files = ["./data/1.json", "./data/2.json", "./data/3.json", "./data/4.json"]
        
    # Print or process the new list
    print(f"test_file: {test_file}; train files: {train_files}")

    # Initialize an empty DataFrame to store processed data
    train_data_df = pd.DataFrame()
    
    # Loop through each file to process training data
    for file in train_files:
        # Check if the file exists
        if os.path.exists(file):
            # Read in training data from the JSON file
            with open(file, 'r') as f:
                data = json.load(f)
                data_list = list(data)
                # print(data_list[:2])
                list_of_dicts = [item for item in data_list]
                
                # Create a DataFrame from the list of dictionaries
                train_data = pd.DataFrame(list_of_dicts)
            
            sidp_df = pd.read_csv('./data/PD_scores_top3_9_24.csv')
            sidp_df.drop(['PNEOSZ_SCALED', 'PNEOBD_SCALED', 'PNEOOC_SCALED'], axis = 1)
            sidp_df = sidp_df[sidp_df['MAPPSZ'] != ' ']
            sidp_df = sidp_df[sidp_df['MAPPBD'] != ' ']
            sidp_df = sidp_df[sidp_df['MAPPOC'] != ' ']
            print(f"Contents of {file}:")
            print(train_data.shape)

            # merge dataframes
            train_data = pd.merge(train_data, sidp_df, on='PARTID', how='inner')
            
            # Preprocess and standardize labels
            train_data = train_data[train_data['MAPPBD'] != '.'] 
            # scaler = StandardScaler()
            # train_data['PNEOA_scaled_new'] = scaler.fit_transform(train_data['PNEOA_scaled'].to_numpy().reshape(-1, 1))
            train_data = train_data[train_data['text'] != '']
            train_data['text'] = train_data['text'].apply(lambda x: contractions.fix(x))
    
            # Prepare DataFrame for training
            train_texts = train_data['text'].astype(str).tolist()
            train_labels = train_data['MAPPBD'].astype(float).tolist()
            #train_labels = [x * 9 for x in train_labels]

            train_partids = train_data['PARTID'].astype(int).tolist()
            train_data_temp = pd.DataFrame({'PARTID': train_partids, 'text': train_texts, 'labels': train_labels})
    
            # Concatenate data to train_data_df
            train_data_df = pd.concat([train_data_df, train_data_temp], ignore_index=True)
            print(train_data_df.head())
            print(train_data_df.describe(),'\n')
    
        else:
            print("File does not exist:", file)
    
    print(train_data_df.shape)
    
    # Initialize an empty DataFrame to store processed data
    test_data_df = pd.DataFrame()
    
    # Check if the file exists
    if os.path.exists(test_file):
        # Read in training data from the JSON file
        with open(test_file, 'r') as f:
            data = json.load(f)
            data_list = list(data)
            # print(data_list[:2])
            list_of_dicts = [item for item in data_list]
            
            # Create a DataFrame from the list of dictionaries
            test_data = pd.DataFrame(list_of_dicts)
        
        sidp_df = pd.read_csv('./data/PD_scores_top3_9_24.csv')
        sidp_df.drop(['PNEOSZ_SCALED', 'PNEOBD_SCALED', 'PNEOOC_SCALED'], axis = 1)
        sidp_df = sidp_df[sidp_df['MAPPSZ'] != ' ']
        sidp_df = sidp_df[sidp_df['MAPPBD'] != ' ']
        sidp_df = sidp_df[sidp_df['MAPPOC'] != ' ']
        print(f"Contents of {test_file}:")
        print(test_data.shape)

        # merge dataframes
        test_data = pd.merge(test_data, sidp_df, on='PARTID', how='inner')
    
        # Preprocess and standardize labels
        test_data = test_data[test_data['MAPPBD'] != '.']
        # scaler = StandardScaler()
        # test_data['PNEOA_scaled_new'] = scaler.fit_transform(test_data['PNEOA_scaled'].to_numpy().reshape(-1, 1))
        test_data = test_data[test_data['text'] != '']
        test_data['text'] = test_data['text'].apply(lambda x: contractions.fix(x))

        # Prepare DataFrame for training
        test_texts = test_data['text'].astype(str).tolist()
        test_labels = test_data['MAPPBD'].astype(float).tolist()
        #test_labels = [x * 9 for x in test_labels]

        test_partids = test_data['PARTID'].astype(int).tolist()
        
        test_data_temp = pd.DataFrame({'PARTID': test_partids, 'text': test_texts, 'labels': test_labels})

        # Concatenate data to test_data_df
        test_data_df = pd.concat([test_data_df, test_data_temp], ignore_index=True)
        print(test_data_df.head())
        print(test_data_df.describe(),'\n')
    else:
        print("File does not exist:", test_file)
    
    print(test_data_df.shape)
        
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)
    
    # Set up k-fold cross-validation
    Results = []
    
    # Initialize training and validation data
    ids = random.sample(list(train_data_df["PARTID"]), int(0.95 * (train_data_df.shape[0])))
    training_data = train_data_df[train_data_df["PARTID"].isin(ids)]
    validation_data = train_data_df[~train_data_df["PARTID"].isin(ids)]
    
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
    model_args.max_seq_length = 512
    model_args.stride = 0.8
    patience = 200
    model_args.early_stopping_patience = patience
    model_args.evaluate_during_training = True
    model_args.evaluate_during_training_steps = 500
    model_args.use_cached_eval_features = False  # If caching causes issues
    
    # Initialize and train the model
    model = ClassificationModel(
        "roberta",
        "roberta-large",
        num_labels=1,
        args=model_args,
        use_cuda=True
    )
    
    model.train_model(training_data, eval_df=validation_data, r2=r2_score, mse=mean_squared_error, mae=mean_absolute_error)
    
    # Evaluate the model
    result, model_outputs, wrong_predictions = model.eval_model(test_data_df, r2=r2_score, mse=mean_squared_error, mae=mean_absolute_error)
    Results.append([result['r2'], result['mse'], result['mae']])
    
    
    # Convert Results to a Pandas DataFrame for easier manipulation and saving
    results_df = pd.DataFrame(Results, columns=['R2', 'MSE', 'MAE'])
    
    # Calculate and print average results
    avg_results = results_df.mean()
    print(f"Average R2: {avg_results['R2']}, MSE: {avg_results['MSE']}, MAE: {avg_results['MAE']}")
    
    # Define the directory where you want to save the results
    print(test_file, type(test_file))
    save_directory = './bdc/Models/Atext/'+str(re.findall(r'\d', str(test_file))[0])+'/'
    
    # Save to Excel
    results_df.to_excel(save_directory + 'A_results.xlsx', index=False)
    
    # Save to CSV
    results_df.to_csv(save_directory + 'A_results.csv', index=False)
    
    
    # Save model, tokenizer, and configuration
    model_save_path = './bdc/Atext/'+str(re.findall(r'\d', str(test_file))[0])+'_'+str(patience)+'/'
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
