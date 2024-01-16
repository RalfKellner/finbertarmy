from transformers import RobertaForMaskedLM
from finbertarmy.finbert_modeling import FinTokenizer, FinDataSet
import torch
import evaluate
from tqdm.auto import tqdm

tokenizer = FinTokenizer("./dicts_and_tokenizers/finroberta_tokenizer.json")
dataset = FinDataSet("/Users/ralfkellner/Data/Textdata/FinRobertaTextsProcessed.sqlite", use_10k=True, use_ec=False, use_esg=False, shuffle_reports=False, batch_size = 16)
dataset.set_limit_obs()
model = RobertaForMaskedLM.from_pretrained("../trained_models/finroberta_seq_len_252_hidden_dim_768_nheads_12_nlayers_6/finroberta")

acc_metric = evaluate.load("accuracy")
for batch in tqdm(dataset):
    lines = [element[0] for element in batch]
    inputs = tokenizer(lines, padding="max_length", max_length=252, truncation=True, return_tensors = "pt")
    inputs["labels"] = inputs.input_ids.clone()
    rand = torch.rand(inputs['input_ids'].shape)
    mask_arr = (rand < 0.15) * (inputs['input_ids'] != 0) * (inputs['input_ids'] != 1) * (inputs['input_ids'] != 2) * (inputs['input_ids'] != 3)
    inputs['input_ids'][mask_arr] = 4

    with torch.inference_mode():
        outputs = model(**inputs)
    masked_pred = outputs.logits.argmax(dim = 2)[mask_arr]

    acc_metric.add_batch(predictions = masked_pred, references = inputs['labels'][mask_arr])

train_acc = acc_metric.compute()
print(f"Prediction accuracy for masked words is: {train_acc['accuracy']:.4f}" )
