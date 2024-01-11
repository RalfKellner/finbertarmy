
import os
import json
from transformers import RobertaConfig
from transformers import RobertaForMaskedLM
import torch
from time import time
from tqdm import tqdm
import numpy as np
import pandas as pd
from finroberta.finroberta_model import FinRobertaTokenizer, FinRobertaDataSet


def financial_masks(input_id_list: list[int], attention_mask_list: list[int], low_mask_pd: float = 0.025, high_mask_pd: float = 0.275) -> list[float]:
    seq_len = np.sum(attention_mask_list)
    mask_identifier = financial_mask_scores.loc[input_id_list].rank() < (seq_len / 2)
    return mask_identifier.apply(lambda x: x * low_mask_pd + (1 - x) * high_mask_pd).values.flatten().tolist()

financial_mask_scores = pd.read_csv("./dicts_and_tokenizers/financial_mask_scores.csv")
financial_mask_scores.set_index("token_id", inplace = True)
tokenizer = FinRobertaTokenizer("./dicts_and_tokenizers/finroberta_tokenizer.json")
dataset = FinRobertaDataSet("/Users/ralfkellner/Data/Textdata/FinRobertaTextsProcessed.sqlite", shuffle_reports=False, batch_size = 32)
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
hidden_size = 768
n_heads = 12
n_layers = 6

config = RobertaConfig(
    vocab_size = tokenizer.vocab_size,  
    max_position_embeddings=max_seq_length + 2,
    hidden_size=hidden_size,
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

    # financial masking
    lines = [element[0]for element in rows]
    inputs_raw = tokenizer(lines, padding="max_length", max_length=252, truncation=True)
    inputs_pt = tokenizer(lines, padding="max_length", max_length=252, truncation=True, return_tensors = "pt")

    mask = torch.Tensor(list(map(financial_masks, inputs_raw.input_ids, inputs_raw.attention_mask)))
    rand = torch.rand(mask.shape)
    mask_arr = rand < mask * (inputs_pt['input_ids'] != 0) * (inputs_pt['input_ids'] != 1) * (inputs_pt['input_ids'] != 2) * (inputs_pt['input_ids'] != 3)

    inputs_pt["labels"] = inputs_pt.input_ids.clone()
    inputs_pt['input_ids'][mask_arr] = 4

    optim.zero_grad()
    input_ids = inputs_pt['input_ids'].to(device)
    attention_mask = inputs_pt['attention_mask'].to(device)
    labels = inputs_pt['labels'].to(device)
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
new_folder = f"finroberta_seq_len_{max_seq_length}_hidden_dim_{hidden_size}_nheads_{n_heads}_nlayers_{n_layers}_financial_mask"
os.mkdir(os.path.join("trained_models", new_folder))
os.chdir(os.path.join("trained_models", new_folder))

with open("dataset_info.json", "w") as jf:
    json.dump(dataset.table_infos, jf)
loss_df = pd.DataFrame(dict(loss_during_training = losses))
loss_df.to_csv("./loss_during_training.csv", index = False)
model.save_pretrained(f"./finroberta")

print("...done!")