import os
import json
from transformers import RobertaConfig
from transformers import RobertaForMaskedLM
import torch
from time import time
from tqdm import tqdm
import numpy as np
import pandas as pd
from finbertarmy.finbert_modeling import FinTokenizer, FinDataSet

tokenizer = FinTokenizer("./dicts_and_tokenizers/finroberta_tokenizer.json")
dataset = FinDataSet("/Users/ralfkellner/Data/Textdata/FinRobertaTextsProcessed.sqlite", shuffle_reports=True, batch_size = 32)
dataset.set_limit_obs()

all_sequences = 0
for table_name in dataset.table_infos.keys():
    all_sequences += dataset.table_infos[table_name]["use_obs"]
iterations = int(np.ceil(all_sequences / dataset.batch_size))

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    print ("MPS device not found.")
    device = torch.device("cpu")

max_seq_length = 252
embedding_dim = 768
n_heads = 12
n_layers = 6

config = RobertaConfig(
    vocab_size = tokenizer.vocab_size,  
    max_position_embeddings=max_seq_length + 2,
    hidden_size=embedding_dim,
    num_attention_heads=n_heads,
    num_hidden_layers=n_layers,
    type_vocab_size=1
)

model = RobertaForMaskedLM(config)
model.to(device)

# activate training mode
model.train()

# initialize optimizer
optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

loop = tqdm(dataset, leave=True)
print(f"Starting training with {iterations} iterations.")
start = time()
losses = []
for i, rows in enumerate(loop):
    if i == 29:
        start_epoch_time_estimation = time()

    # masking
    lines = [element[0]for element in rows]
    inputs = tokenizer(lines, padding="max_length", max_length=252, truncation=True, return_tensors = "pt")
    inputs["labels"] = inputs.input_ids.clone()
    rand = torch.rand(inputs['input_ids'].shape)
    mask_arr = (rand < 0.15) * (inputs['input_ids'] != 0) * (inputs['input_ids'] != 1) * (inputs['input_ids'] != 2) * (inputs['input_ids'] != 3)
    inputs['input_ids'][mask_arr] = 4

    optim.zero_grad()
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    labels = inputs['labels'].to(device)
    outputs = model(input_ids = input_ids, attention_mask = attention_mask, labels = labels)
    loss = outputs.loss
    loss.backward()
    optim.step()

    if i % 10 == 0:
        losses.append(loss.item())

    if i == 29:
        end_epoch_time_estimation = time()
        time_per_iteration = (end_epoch_time_estimation - start_epoch_time_estimation) 
        print(f"\n\nEstimated run time for this epoch in days: {(time_per_iteration * iterations) / (60*24):.2f} \n")

    # print relevant info to progress bar
    loop.set_postfix(loss=loss.item())
end = time()

print(f"Training took {end - start} seconds. Saving results...")

# save results_to_model_path
new_folder = f"finroberta_seq_len_{max_seq_length}_hidden_dim_{embedding_dim}_nheads_{n_heads}_nlayers_{n_layers}"
os.mkdir(os.path.join("trained_models", new_folder))
os.chdir(os.path.join("trained_models", new_folder))

with open("dataset_info.json", "w") as jf:
    json.dump(dataset.table_infos, jf)
loss_df = pd.DataFrame(dict(loss_during_training = losses))
loss_df.to_csv("./loss_during_training.csv", index = False)
model.save_pretrained(f"./finroberta")

print("...done!")