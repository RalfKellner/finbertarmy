import sqlite3
from tokenizers import normalizers, processors, decoders, Regex, Tokenizer
from tokenizers.normalizers import NFD, StripAccents, Lowercase, Replace
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
import json
from datasets import load_dataset


def iter_databases(use_10k: bool = True, use_ec: bool = True, use_esg: bool = True, use_wiki: bool = True) -> None:
    db_10k = "/Users/ralfkellner/Data/Textdata/K_reports_full_w_delisted.sqlite"
    table_name_10k = "k_reports_full"
    conn_10k = sqlite3.connect(db_10k, check_same_thread=False)
    curs_10k = conn_10k.cursor()

    db_ecs = "/Users/ralfkellner/Data/Textdata/ec_fmp_full.sqlite"
    table_name_ecs = "earningcalls"
    conn_ecs = sqlite3.connect(db_ecs, check_same_thread=False)
    curs_ecs = conn_ecs.cursor()

    with open("/Users/ralfkellner/Data/Textdata/ESG_reports.json", "r") as file:
        data_dict = json.load(file)
    keys = list(data_dict.keys())

    stay_at_10ks = True
    stay_at_ecs = True

    curs_10k.execute(f"SELECT * FROM {table_name_10k};")
    curs_ecs.execute(f"SELECT * FROM {table_name_ecs};")
    
    if use_10k:
        while stay_at_10ks:
            row = curs_10k.fetchone()
            if not row:
                stay_at_10ks = False
            else:
                report = row[18]
                if report is None:
                    continue
                else:
                    yield report
    curs_10k.close()
    conn_10k.close()

    if use_ec:
        while stay_at_ecs:
            row = curs_ecs.fetchone()
            if not row:
                stay_at_ecs = False
            else:
                ec = row[3]
                yield ec
    curs_ecs.close()
    conn_ecs.close()

    if use_wiki:
        wiki_ds = load_dataset("/Users/ralfkellner/Data/Textdata/wikipedia_dumps")
        for i in range(0, len(wiki_ds["train"]), 1000): 
            yield wiki_ds["train"][i : i + 1000]["text"]

    if use_esg:
        for key in keys:
            for report_year in data_dict[key].keys():
                yield data_dict[key][report_year][0]


print("Decide which corpus should be included when training the tokenizer:")
print(f"10K reports (yes/no)?")
ask_10k = input()
assert ask_10k in ["yes", "no"], "Answer must be yes or no."
if ask_10k == "yes":
    use_10k = True
else:
    use_10k = False
print(f"Earning call transcripts (yes/no)?")
ask_ec = input()
assert ask_ec in ["yes", "no"], "Answer must be yes or no."
if ask_ec == "yes":
    use_ec = True
else:
    use_ec = False
print(f"ESG reports (yes/no)?")
ask_esg = input()
assert ask_esg in ["yes", "no"], "Answer must be yes or no."
if ask_esg == "yes":
    use_esg = True
else:
    use_esg = False
print(f"English wikipedia (yes/no)?")
ask_wiki = input()
assert ask_wiki in ["yes", "no"], "Answer must be yes or no."
if ask_wiki == "yes":
    use_wiki = True
else:
    use_wiki = False

print("Type the filename under which the tokenizer is supposed to be saved:")
tokenizer_name = input()

all_texts = iter_databases(use_10k=use_10k, use_ec = use_ec, use_esg = use_esg, use_wiki = use_wiki)

# if we want to replace numbers by a number token
regex_nbr = '\d+\s|\s\d+\s|\s\d+$' 
# if we want to remove special characters
#regex = '[0-9!@#$%^&*()_+{}\[\]:;<>,.?~\\-\'"]'
regex = '[0-9]'

normalizer = normalizers.Sequence([NFD(), StripAccents(), Lowercase(), Replace(Regex(regex), ""), Replace(Regex(' +'), ' ')]) 
pre_tokenizer = ByteLevel(add_prefix_space=False) 

# bpe tokenizer
tokenizer = Tokenizer(BPE(unk_token='<unk>')) 
tokenizer.normalizer = normalizer
tokenizer.pre_tokenizer = pre_tokenizer

# train tokenizer
print("Starting with training....")
trainer = BpeTrainer(vocab_size = 50000, min_frequency = 5, special_tokens=['<s>', '<pad>', '<unk>', '</s>', '<mask>'])  
tokenizer.train_from_iterator(all_texts, trainer)

bos_token_id = tokenizer.token_to_id("<s>")
eos_token_id = tokenizer.token_to_id("</s>")

post_processor_handle_offsets = processors.ByteLevel(trim_offsets=False)
post_processor_template = processors.TemplateProcessing(
    single="<s> $A </s>",
    special_tokens=[("<s>", bos_token_id), ("</s>", eos_token_id)],
)

post_processor = processors.Sequence([post_processor_handle_offsets, post_processor_template])
tokenizer.post_processor = post_processor
tokenizer.decoder = decoders.ByteLevel()

# save tokenizer
tokenizer.save(f"./dicts_and_tokenizers/{tokenizer_name}.json")
print("Training ended, tokenizer has been saved.")