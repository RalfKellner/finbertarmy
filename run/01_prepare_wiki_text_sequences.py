import json
import sqlite3
from finbertarmy.finbert_modeling import FinTokenizer
from finbertarmy.utils import WikiProcessor


def json_save(count_dict: dict, filename: str) -> None:
    json_out = json.dumps(count_dict, indent=4)
    with open(filename, "w") as js:
        js.write(json_out)

db_out = "/Users/ralfkellner/Data/Textdata/WikiTextsProcessed_wo_punctuation.sqlite"
table_name_out = "wiki_sequences"

tokenizer = FinTokenizer("./dicts_and_tokenizers/finroberta_tokenizer_wo_punctuation.json")

count_dict_wiki= dict()
for i in range(tokenizer.vocab_size):
    count_dict_wiki[i] = 0

wiki_processor = WikiProcessor(
    db_out=db_out,
    table_name_out=table_name_out,
    count_dict=count_dict_wiki,
    min_whitespace_len=5,
    limit = None
)

conn = sqlite3.connect(wiki_processor.db_out)
curs = conn.cursor()
curs.execute(f"SELECT * FROM {wiki_processor.table_name_out} LIMIT 10;")
res = curs.fetchall()
curs.close()
conn.close()

process_wiki_texts = "yes"
if len(res) > 0:
    print(f"FYI: The database includes sequences for the table with name: {table_name_out}.")
print(f"Do you want to process sequences for the table with name: {table_name_out} (yes/no)?")
ask_user = input()
assert ask_user in ["yes", "no"], "You must answer with yes or no."
if ask_user == "no":
    print("No sequences for this corpus type are added.")
    process_wiki_texts = "no"

if process_wiki_texts == "yes":
    print("Processing of wiki texts starts...")
    wiki_processor.process_wiki_texts(tokenizer=tokenizer)
    print("Processing of reports if done. Database with text sequence should be ready. Starting to save token count dictionaries.")

    for i in range(tokenizer.vocab_size):
        count_dict_wiki[i] = int(count_dict_wiki[i])

    json_save(count_dict_wiki, "./dicts_and_tokenizers/token_counts_wiki_corpus_wo_punctuation.json")

    print("Count dictionary is saved...all done!")
else:
    print("Exited script without processing.")
