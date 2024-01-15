from transformers import RobertaForSequenceClassification
from finbertarmy.finbert_modeling import FinTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
from torch.optim import AdamW
from tqdm.auto import tqdm
import evaluate

# determine if gpu is available
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    print ("MPS device not found.")
    device = torch.device("cpu")

tokenizer = FinTokenizer("./dicts_and_tokenizers/finroberta_tokenizer.json")

def tokenize_function(dataset_row):
    return tokenizer(dataset_row["sentence"], padding = "max_length", max_length = 252, truncation = True)

# prepare data
dataset = load_dataset("financial_phrasebank", 'sentences_66agree')

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets["train"]
tokenized_datasets = tokenized_datasets.remove_columns(["sentence"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")
tokenized_datasets = tokenized_datasets.shuffle(seed = 42)

training_data = tokenized_datasets.select(list(range(3000)))
evaluation_data = tokenized_datasets.select(list(range(3000, tokenized_datasets.num_rows)))

train_dataloader = DataLoader(training_data, shuffle=True, batch_size=8)
eval_dataloader = DataLoader(evaluation_data, batch_size=8)

# load model
model_name = "finroberta_seq_len_252_hidden_dim_768_nheads_12_nlayers_6/"
model_type = "finroberta/"
model_path = "./trained_models/" + model_name + model_type 
model = RobertaForSequenceClassification.from_pretrained(model_path, num_labels = 3)
model.to(device)

# activate training mode
model.train()

# initialize optimizer
optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

losses = []
epochs = 3
for epoch in range(epochs):
    # determine a metric for training
    train_metric = evaluate.load("accuracy")
    loop = tqdm(train_dataloader, leave=True)
    print(f"Starting training for epoch: {epoch}")
    for batch in loop:
        batch = {k: v.to(device) for k, v in batch.items()}
        optim.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optim.step()

        logits = outputs.logits 
        predictions = torch.argmax(logits, dim = 1)
        train_metric.add_batch(predictions = predictions, references = batch["labels"])

        # print relevant info to progress bar
        loop.set_postfix(loss=loss.item())
        losses.append(loss.item())

    train_acc = train_metric.compute()
    print(f"Training accuracy after epoch: {train_acc['accuracy']}")

    # determine a metric for evaluation
    eval_metric = evaluate.load("accuracy")
    eval_loop = tqdm(eval_dataloader)
    print(f"Starting evaluation for epoch: {epoch}")
    for batch in eval_loop:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.inference_mode():
            outputs = model(**batch)
        logits = outputs.logits 
        predictions = torch.argmax(logits, dim = 1)
        eval_metric.add_batch(predictions = predictions, references = batch["labels"])

    eval_acc = eval_metric.compute()
    print(f"Evaluation accuracy after epoch: {eval_acc['accuracy']}")
    

save_path = "./finetuned_models/"
task = "financial_phrasebank/"
model.save_pretrained(save_path + model_name + task)
