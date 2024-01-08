from bs4 import BeautifulSoup
import nltk.data
import sqlite3
import pandas as pd
import re
import json
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
from .finroberta_model import FinRobertaTokenizer


class CorpusProcessor:
    def __init__(
            self,
            db_in: str,
            db_out: str,
            table_name_in: str,
            table_name_out: str,
            count_dict: dict = None
        ) -> None:

        self.db_in = db_in
        self.db_out = db_out
        self.table_name_in = table_name_in
        self.table_name_out = table_name_out
        self.count_dict = count_dict
        
    """ 
    A class which serves as the parent class for processing raw reports into a database such that each row includes a text sequence with comparable length
    in comparison to the other text sequences. 
    
    Arguments:
    ----------
    db_in: data path to the database which includes raw text data
    db_out: data path to database to which processed text sequences should be exported
    table_name_in: name of the table with raw texts
    table_name_out: name of the table where text sequences are supposed to be stored
    tokenizer: the pre-trained tokenizer
    """

    def _chunk_sequences(self, texts: list[str], tokenizer: FinRobertaTokenizer, seq_length: int = 252) -> list[str]:

        """ 
        This function chunks text sequences together as long as the overall token length is not longer than seq_length.
        
        Arguments:
        ----------
        texts: a list of paragraphs or sentences from the reports
        seq_length: the desired sequence length of all texts sequences which are supposed for pre-training afterwards
        """

        # subtract 2 due to bos and eos token for each sequence
        max_length = seq_length - 2
        text_chunks = []
        current_length = 0
        current_chunk = ''
        for text in texts:
            # determine the number of tokens using the tokenizer
            seq_token_ids = tokenizer(text).input_ids[1:-1] # do not count bos and eos tokens
            # if id occurrences are supposed to be counted during processing, a dictionary for counting will be included
            if self.count_dict:
                ids, counts = np.unique(seq_token_ids, return_counts=True)
                for id, count in zip(ids, counts):
                    self.count_dict[id] += count
            seq_length = len(seq_token_ids)
            # it is possible that the chunk length is higher than max_length, if this is the case for one text, currently, I leave it as is and truncate these sequences during training
            current_length += seq_length
            if current_length < max_length:
                current_chunk += text
            # if the added text leads to exceedance of seq_length create a new chunk
            else:
                text_chunks.append(current_chunk)
                current_chunk = ''
                current_length = seq_length
                current_chunk += text
        return text_chunks
    
    def _get_table_names(self) -> list:

        """ 
        Get table names for the database with processed text sequences.
        """

        conn = sqlite3.connect(self.db_out)
        curs = conn.cursor()
        get_table_names_query = "SELECT name FROM sqlite_master WHERE type='table';"
        curs.execute(get_table_names_query)
        res = curs.fetchall()
        curs.close()
        conn.close()
        if res:
            names = [element[0] for element in res]
        else:
            names = []
        return names

    def _create_table(self):

        """ 
        Usually a new and empty database for outgoing text sequences is generated. If this is the case a new table must be created.
        """
        
        conn = sqlite3.connect(self.db_out)
        curs = conn.cursor()
        create_table_query = f"CREATE TABLE {self.table_name_out} (sequences text);"
        curs.execute(create_table_query)
        conn.commit()
        curs.close()
        conn.close()

    def _insert_texts(self, texts: list[str]) -> None:

        """ 
        After texts are processed and chunked, they are exported to the outgoing database with this function.
        """
        
        texts_out = [tuple([text]) for text in texts]
        conn = sqlite3.connect(self.db_out)
        curs = conn.cursor()
        insert_text_query = f"INSERT INTO {self.table_name_out} (sequences) VALUES (?);"
        curs.executemany(insert_text_query, texts_out)
        conn.commit()
        curs.close()
        conn.close()


class KReportProcessor(CorpusProcessor):
    def __init__(
        self,
        db_in: str,
        db_out: str,
        table_name_in: str,
        table_name_out: str, 
        report_row_id: int,
        report_type: str,    
        min_whitespace_len: int = 5,
        limit_number_of_ciks = None,
        count_dict: dict = None  
        ):

        """ 
        A class which is used processing raw 10K reports into a database such that each row includes a text sequence with comparable length
        in comparison to the other text sequences. 
        
        Arguments:
        ----------
        db_in: data path to the database which includes company reports
        db_out: data path to database to which processed text sequences are supposed to be exported
        table_name_in: name of the table with reports in the company report database
        table_name_out: name of the table where text sequences are supposed to be stored
        report_row_id: the number of the column which includes the text of the report
        report_type: txt or html depending if we use the database with 10Ks in html or text form
        min_whitespace_len: the number of whitespace separated words a sentence needs to have for being included in the outgoing database
        """
        
        assert report_type in ["html", "txt"], "report_type should be html or txt"
        super().__init__(db_in, db_out, table_name_in, table_name_out, count_dict)
        self.report_row_id = report_row_id
        self.report_type = report_type
        self.sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
        self.min_whitespace_len = min_whitespace_len  

        # usually the outgoing database is created and a table must be created after initialization
        table_names = self._get_table_names()
        if not(self.table_name_out in table_names):
            self._create_table()

        self.limit_number_of_ciks = limit_number_of_ciks
        self.ciks = self._get_ciks()
        if self.limit_number_of_ciks:
            self.ciks = np.random.choice(self.ciks, size = self.limit_number_of_ciks, replace = False)

    def _get_ciks(self) -> list[str]: 

        """ 
        Output a list of strings in the database.
        """
        
        conn = sqlite3.connect(self.db_in)
        curs = conn.cursor()
        curs.execute(f"SELECT DISTINCT cik FROM {self.table_name_in};")
        res = curs.fetchall()
        curs.close()
        conn.close()
        ciks = [element[0] for element in res]
        return ciks
    
    def _split_html_to_paragraphs(self, report: str) -> list[str]:

        """
        Arguments:
        ----------
        report: company report in html format
        """

        # extract paragraphs by paragraph tag
        soup = BeautifulSoup(report, "lxml")
        paragraphs = soup.find_all('p')

        # keep only paragraphs which are longer than min_whitespace_len
        paragraph_texts = []
        for par in paragraphs:
            par_text = par.get_text()
            par_length = len(par_text.split(" "))
            if par_length > self.min_whitespace_len:
                paragraph_texts.append(par_text)
        return paragraph_texts

    def _split_str_to_sentences(self, report: str) -> list[str]:

        """
        Arguments:
        ----------
        report: company report in text format
        """

        # extract sentences with the sentence tokenizer
        sentences = self.sent_detector.tokenize(report)
        
        # keep only sentences which are longer than min_whitespace_len
        sentence_texts = []
        for sentence_text in sentences:
            sentence_length = len(sentence_text.split(" "))
            if sentence_length > self.min_whitespace_len:
                sentence_texts.append(sentence_text)
        return sentence_texts
    
    def _collect_reports_per_cik_and_deduplicate(self, cik: str) -> list[str]:

        """ 
        This function gets all reports from a company, splits the reports into paragraphs or sentences and drop duplicate texts.

        Arguments:
        ----------
        cik: string identifier for each company
        """

        # get all reports for the cik
        conn = sqlite3.connect(self.db_in)
        curs = conn.cursor()
        sql_query = f"SELECT * FROM {self.table_name_in} WHERE cik='{cik}';"
        curs.execute(sql_query)
        res = curs.fetchall()
        curs.close()
        conn.close()

        # split every report into paragraphs of sentences
        reports_per_cik_splitted = []
        for row in res:
            try:
                if self.report_type == "html":
                    report = row[self.report_row_id].decode()
                    splitted_report = self._split_html_to_paragraphs(report)
                else:
                    report = row[self.report_row_id]
                    splitted_report = self._split_str_to_sentences(report)    
                reports_per_cik_splitted.extend(splitted_report)
            except:
                print(f"Problem occurred while processing cik: {cik}")

        # drop duplicate text parts
        reports_per_cik_splitted = pd.Series(reports_per_cik_splitted).drop_duplicates().tolist()
        return reports_per_cik_splitted

    def process_reports_per_cik(self, tokenizer: FinRobertaTokenizer, seq_length: int = 252) -> None:

            """ 
            This function combines individual operations. It collects non-duplicate text parts from all reports for a given company, chunks
            these parts into text sequences with comparable length and writes these chunks to the outgoing database.

            Arguments:
            ----------
            cik: string identifier for each company
            seq_length: the desired amount of tokens for every text sequence which is prepared for pre-training
            """
            for cik in tqdm(self.ciks):
                texts = self._collect_reports_per_cik_and_deduplicate(cik)
                text_chunks = self._chunk_sequences(texts, tokenizer, seq_length)
                self._insert_texts(text_chunks)


class EarningCallProcessor(CorpusProcessor):
    def __init__(
        self,
        db_in: str,
        db_out: str,
        table_name_in: str,
        table_name_out: str, 
        ec_row_id: int,
        min_whitespace_len: int = 5,
        limit_number_of_rows = None,
        count_dict: dict = None    
        ):

        """ 
        A class which is used processing raw 10K reports into a database such that each row includes a text sequence with comparable length
        in comparison to the other text sequences. 
        
        Arguments:
        ----------
        db_in: data path to the database which includes company reports
        db_out: data path to database to which processed text sequences are supposed to be exported
        table_name_in: name of the table with reports in the company report database
        table_name_out: name of the table where text sequences are supposed to be stored
        ec_row_id: the number of the column which includes the text of the earning call
        min_whitespace_len: the number of whitespace separated words a sentence needs to have for being included in the outgoing database
        """
        
        super().__init__(db_in, db_out, table_name_in, table_name_out, count_dict)
        self.ec_row_id = ec_row_id
        self.min_whitespace_len = min_whitespace_len  
        self.limit_number_of_rows = limit_number_of_rows
        self.sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

        # usually the outgoing database is created and a table must be created after initialization
        table_names = self._get_table_names()
        if not(self.table_name_out in table_names):
            self._create_table()

    def _split_ec_to_sentences(self, ec_transcript: str) -> list[str]:
        ec_fractions = ec_transcript.split("\n")
        speech_content = ""
        for ec_split in ec_fractions:
            try:
                speech_content += ec_split.split(":", maxsplit = 1)[1][1:]
            except Exception as e:
                print(f"A single speech content in an earning call resulted in the following error:\n")
                print(e)
        sentences = self.sent_detector.tokenize(speech_content)
        # keep only sentences which are longer than min_whitespace_len
        sentence_texts = []
        for sentence_text in sentences:
            sentence_length = len(sentence_text.split(" "))
            if sentence_length > self.min_whitespace_len:
                sentence_texts.append(sentence_text)
        return sentence_texts
    
    def process_earning_calls(self, tokenizer: FinRobertaTokenizer, seq_length: int = 252) -> None:
        conn = sqlite3.connect(self.db_in)
        curs = conn.cursor()
        if self.limit_number_of_rows:
            res = curs.execute(f"SELECT * FROM {self.table_name_in} LIMIT {self.limit_number_of_rows};")
        else:
            res = curs.execute(f"SELECT * FROM {self.table_name_in};")

        for row in tqdm(res):
            ec = row[self.ec_row_id]
            ec_sentences = self._split_ec_to_sentences(ec)
            ec_chunks = self._chunk_sequences(ec_sentences, tokenizer, seq_length)
            self._insert_texts(ec_chunks)
        

class ESGReportProcessor(CorpusProcessor):
    def __init__(
        self,
        json_file: str,
        db_out: str,
        table_name_out: str, 
        min_whitespace_len: int = 5,
        limit_number_of_keys_identifiers = None,
        count_dict: dict = None  
        ):

        """ 
        A class which is used processing raw ESG reports into a database such that each row includes a text sequence with comparable length
        in comparison to the other text sequences. 
        
        Arguments:
        ----------
        json_file: path to json file including ESG reports
        db_out: data path to database to which processed text sequences are supposed to be exported
        table_name_out: name of the table where text sequences are supposed to be stored
        min_whitespace_len: the number of whitespace separated words a sentence needs to have for being included in the outgoing database
        """
        
        super().__init__(db_in = None, db_out = db_out, table_name_in = None, table_name_out = table_name_out, count_dict = count_dict)
        self.json_file = json_file
        self._load_json_dict()
        self.min_whitespace_len = min_whitespace_len  
        self.limit_number_of_keys_identifiers = limit_number_of_keys_identifiers

        # usually the outgoing database is created and a table must be created after initialization
        table_names = self._get_table_names()
        if not(self.table_name_out in table_names):
            self._create_table()

        self.key_identifiers = self._get_key_identifiers()
        if self.limit_number_of_keys_identifiers:
            self.key_identifiers = np.random.choice(self.key_identifiers, size = self.limit_number_of_keys_identifiers, replace = True)

    def _load_json_dict(self):

        """Load json file with ESG reports"""

        with open(self.json_file, "r") as jb:
            self.esg_dict = json.load(jb)

    def _get_key_identifiers(self) -> list[str]: 

        """ 
        Output a list of strings with all primary keys in the json file.
        """
        key_identifiers = list(self.esg_dict.keys())
        key_identifiers.sort()
        return key_identifiers
    
    def _split_esg_report_to_sentences(self, report: str, min_whitespace_len: int = 5) -> list[str]:
        
        """ 
        Split each report into sentences, exclude numbers and punctuation; this is done here due to the uncommon current way of storing report strings.
        """

        regex = re.compile('[0-9!@#$%^&*()_+{}\[\]:;<>,.?~\\-\'"]')
        sentences = []
        for sentence in report.split("Sentence"):
            try:
                _, sentence = sentence.split(":", maxsplit = 1)
                sentence = re.sub(regex, "", sentence)
                sentence = f"{sentence.lstrip().rstrip()}."
                len_sentence = len(sentence.split(" ")) 
                if len_sentence > min_whitespace_len:
                    sentences.append(sentence)
            except: 
                continue
        return sentences

    def _collect_reports_per_key_identifier_and_deduplicate(self, key_identifier: str) -> list[str]:

        """ 
        Collect sentences from all reports for one key identifier and depuplicate duplicate sentences.
        """

        splitted_reports = []
        for key in self.esg_dict[key_identifier]:
            splitted_report = self._split_esg_report_to_sentences(self.esg_dict[key_identifier][key][0])
            splitted_reports.extend(splitted_report)
        splitted_reports = pd.Series(splitted_reports).drop_duplicates().tolist()
        return splitted_reports

    def process_reports_per_key_identifier(self, tokenizer: FinRobertaTokenizer, seq_length: int = 252) -> None:

            """ 
            This function combines individual operations. It collects non-duplicate text parts from all reports for a given key identifier, chunks
            these parts into text sequences with comparable length and writes these chunks to the outgoing database.

            Arguments:
            ----------
            key_identifier: string identifier for each company
            seq_length: the desired amount of tokens for every text sequence which is prepared for pre-training
            """

            for key_identifier in tqdm(self.key_identifiers):
                texts = self._collect_reports_per_key_identifier_and_deduplicate(key_identifier)
                text_chunks = self._chunk_sequences(texts, tokenizer, seq_length)
                self._insert_texts(text_chunks)



class WikiProcessor(CorpusProcessor):
    def __init__(self, db_out: str, table_name_out: str, count_dict: dict = None, min_whitespace_len: int = 5, limit: int = None) -> None:
        self.db_out = db_out
        self.table_name_out = table_name_out
        self.count_dict = count_dict
        self.limit = limit
        super().__init__(
            db_in = None,
            db_out = db_out,
            table_name_in = None,
            table_name_out = table_name_out,
            count_dict = count_dict
        )
        self.wiki_ds = load_dataset("/Users/ralfkellner/Data/Textdata/wikipedia_dumps")
        self.sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        self.min_whitespace_len = min_whitespace_len

        # usually the outgoing database is created and a table must be created after initialization
        table_names = self._get_table_names()
        if not(self.table_name_out in table_names):
            self._create_table()

    def __iter__(self):
        if self.limit:
            iter_stop = self.limit
        else:
            iter_stop = len(self.wiki_ds["train"])
        for i in range(0, iter_stop): 
            yield self.wiki_ds["train"][i]["text"]

    def _split_wiki_text_into_sentences(self, wiki_text: str) -> list[str]:
        sentences = self.sentence_tokenizer.tokenize(wiki_text)
        sentence_texts = []
        for sentence_text in sentences:
            sentence_length = len(sentence_text.split(" "))
            if sentence_length > self.min_whitespace_len:
                sentence_texts.append(sentence_text)
        return sentence_texts
    
    def process_wiki_texts(self, tokenizer: FinRobertaTokenizer, seq_length: int = 252) -> None:
        for wiki_text in tqdm(self.__iter__()):
            wiki_sentences = self._split_wiki_text_into_sentences(wiki_text)
            wiki_chunks = self._chunk_sequences(wiki_sentences, tokenizer)
            self._insert_texts(wiki_chunks)
