from finroberta.utils import KReportProcessor, EarningCallProcessor, ESGReportProcessor
from finroberta.finroberta_model import FinRobertaTokenizer
import sqlite3
from tqdm.notebook import tqdm
import json

def json_save(count_dict: dict, filename: str) -> None:
    json_out = json.dumps(count_dict, indent=4)
    with open(filename, "w") as js:
        js.write(json_out)


tokenizer = FinRobertaTokenizer("./dicts_and_tokenizers/finroberta_tokenizer.json")
count_dict_10k, count_dict_ec, count_dict_esg, count_dict_finance_corpus = dict(), dict(), dict(), dict()  
for i in range(tokenizer.vocab_size):
    count_dict_10k[i] = 0
    count_dict_ec[i] = 0
    count_dict_esg[i] = 0
    
db_10k_in = "/Users/ralfkellner/Data/Textdata/K_reports_full_w_delisted.sqlite"
db_ec_in = "/Users/ralfkellner/Data/Textdata/ec_fmp_full.sqlite"
json_esg = "/Users/ralfkellner/Data/Textdata/ESG_reports.json"
table_name_10k_in = "k_reports_full"
table_name_ec_in = "earningcalls"

db_out = "/Users/ralfkellner/Data/Textdata/FinRobertaTextsProcessed.sqlite"
table_name_10k_out = "k_report_sequences"
table_name_ec_out = "ec_sequences"
table_name_esg_out = "esg_sequences"

kreport_processor = KReportProcessor(
    db_10k_in,
    db_out,
    table_name_10k_in,
    table_name_10k_out,
    18,
    "txt",
    min_whitespace_len = 5,
    limit_number_of_ciks = None,
    count_dict = count_dict_10k
)

ec_processor = EarningCallProcessor(
    db_ec_in,
    db_out,
    table_name_ec_in,
    table_name_ec_out,
    3,
    min_whitespace_len = 5,
    limit_number_of_rows = None,
    count_dict = count_dict_ec
)

esg_processor = ESGReportProcessor(
    json_esg,
    db_out,
    table_name_esg_out,
    min_whitespace_len = 5,
    limit_number_of_keys_identifiers = None,
    count_dict = count_dict_esg
)

process_tables = dict()
# does not matter for which instance we call this method, it returns table names for the outgoing database with text sequences
table_names = esg_processor._get_table_names()
for name in table_names:
    process_tables[name] = "yes"
 

# check if database already includes processed text sequences
includes_sequences = []
for name in table_names:
    conn = sqlite3.connect(db_out)
    curs = conn.cursor()
    curs.execute(f"SELECT * FROM {name} LIMIT 5;")
    res = curs.fetchall()
    curs.close()
    conn.close()
    if len(res) > 0:
        print(f"FYI: The database includes sequences for the table with name: {name}.")
    print(f"Do you want to process sequences for the table with name: {name} (yes/no)?")
    ask_user = input()
    assert ask_user in ["yes", "no"], "You must answer with yes or no."
    if ask_user == "no":
        print("No sequences for this report type are added.")
        process_tables[name] = "no"


if process_tables[table_name_10k_out] == "yes":
    print("Starting to process K reports....")
    kreport_processor.process_reports_per_cik(tokenizer)
    print("Processing of K reports finished.")

if process_tables[table_name_ec_out] == "yes":
    print("Starting to process earning call transcripts....")
    ec_processor.process_earning_calls(tokenizer)
    print("Processing of earning call transcripts finished.")

if process_tables[table_name_esg_out] == "yes":
    print("Starting to process ESG reports....")
    esg_processor.process_reports_per_key_identifier(tokenizer)
    print("Processing of ESG reports finished.")


print("Processing of reports if done. Database with text sequence should be ready. Starting to save token count dictionaries.")

for i in range(tokenizer.vocab_size):
    count_dict_10k[i] = int(count_dict_10k[i])
    count_dict_ec[i] = int(count_dict_ec[i])
    count_dict_esg[i] = int(count_dict_esg[i])
    count_dict_finance_corpus[i] = count_dict_10k[i] + count_dict_ec[i] + count_dict_esg[i]

json_save(count_dict_10k, "./dicts_and_tokenizers/token_counts_10k.json")
json_save(count_dict_ec, "./dicts_and_tokenizers/token_counts_ec.json")
json_save(count_dict_esg, "./dicts_and_tokenizers/token_counts_esg.json")
json_save(count_dict_finance_corpus, "./dicts_and_tokenizers/token_counts_finance_corpus.json")

print("Count dictionaries saved...all done!")
