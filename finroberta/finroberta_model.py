from transformers import PreTrainedTokenizerFast
import sqlite3
import numpy as np


class FinRobertaTokenizer(PreTrainedTokenizerFast):

    """
    A tokenizer using the fast tokenizer wrapper class from transformers. This transformer comes with all functionalities as present for
    official models.
    """

    def __init__(
        self,
        tokenizer_file: str = None,
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        sep_token: str = "</s>",
        cls_token: str = "<s>",
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        mask_token: str = "<mask>",
        add_prefix_space: bool = False,
        trim_offsets: bool = True,
    ):
        super().__init__(
            tokenizer_file=tokenizer_file,
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            add_prefix_space=add_prefix_space,
            trim_offsets=trim_offsets
        )


class FinRobertaDataSet():

    """ 
    A class to iterate through all preprocess database tables. 

    db_name: the name of the data base including pre-processed text sequences
    use_10k: if k report sequences should be used for training
    use_ec: if earning call sequences should be used for training
    use_esg: if esg report sequences should be used for training
    shuffle_reports: if text sequences should be randomly drawn from each table
    batch_size: number of text sequences per iteration
    """

    def __init__(
            self,
            db_name: str,
            use_10k: bool = True,
            use_ec: bool = True,
            use_esg: bool = True,
            shuffle_reports: bool = True,
            batch_size: int = 16
        ) -> None:
        self.db_name = db_name
        self.batch_size = batch_size
        self.use_10k = use_10k
        self.use_ec = use_ec
        self.use_esg = use_esg
        self.shuffle_reports = shuffle_reports
        self._get_table_infos()
        self._set_catch_batch_probabilities()
        if not(self.use_10k):
            self.table_infos.pop("k_report_sequences")
        if not(self.use_ec):
            self.table_infos.pop("ec_sequences")
        if not(self.use_esg):
            self.table_infos.pop("esg_sequences")
        

    def _get_table_infos(self) -> list:

        """ 
        Get table names and number of observations for each table. This call creates the self.table_infos dictionary which is used for iteration.
        """

        conn = sqlite3.connect(self.db_name)
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
        self.table_infos = {key: {} for key in names}

        print("Starting to count observations in each table...")
        conn = sqlite3.connect(self.db_name)
        curs = conn.cursor()
        for table_name in self.table_infos.keys():
            curs.execute(f"SELECT COUNT(*) FROM {table_name};")
            res = curs.fetchone()
            self.table_infos[table_name]["n_obs"] = res[0]
            self.table_infos[table_name]["limit"] = None
        curs.close()
        conn.close()
        print("Counting of observations in each table finished.")
        print("Table count:\n -------------- ")
        for name, count in self.table_infos.items():
            print(f"{name}: {count['n_obs']}")

    def set_limit_obs(self) -> None:

        """ 
        A function to set the number of observations to use from each table.
        """

        for table_name in self.table_infos.keys():
            print(f"Please provide the maximum number of text sequences you want to use for table {table_name}.\n The highest possible number is {self.table_infos[table_name]['n_obs']}.")
            limit_input = int(input())
            if limit_input > self.table_infos[table_name]['n_obs']:
                raise ValueError("Limit must be smaller than the amount of available text sequences.")
            self.table_infos[table_name]["limit"] = limit_input
        self._set_catch_batch_probabilities()

    def _set_catch_batch_probabilities(self):

        """ 
        During iteration batches are drawn randomly from each table. The probabilities for drawing batches are derived by determining relative numbers of text sequences for each table.
        Hereby, limits are used instead of the overall number of text sequences for each table, given limits are set by the self.set_limit_obs method.
        """

        for table_name in self.table_infos.keys():
            limit = self.table_infos[table_name]["limit"]
            self.table_infos[table_name]["use_obs"] = self.table_infos[table_name]["limit"] if limit else self.table_infos[table_name]["n_obs"]

        sum_use_obs = 0
        for table_name in self.table_infos.keys():
            sum_use_obs += self.table_infos[table_name]["use_obs"]

        for table_name in self.table_infos.keys():
            self.table_infos[table_name]["use_prob"] = self.table_infos[table_name]["use_obs"] / sum_use_obs

    def __iter__(self):

        """ 
        Most relevant for training. This iterator randomly samples batches of text sequences from all tables using probabilities derived by the self._set_catch_batch_probabilities.
        """

        table_names = list(self.table_infos.keys())
        table_probabilities = [self.table_infos[table_name]["use_prob"] for table_name in table_names]

        conn = sqlite3.connect(self.db_name)
        cursor_10k = conn.cursor()
        cursor_ec = conn.cursor()
        cursor_esg = conn.cursor()

        random_query = "ORDER BY RANDOM()" if self.shuffle_reports else None

        if self.use_10k:
            base_query_10k = "SELECT * FROM k_report_sequences"
            limit_query_10k = f"LIMIT {self.table_infos['k_report_sequences']['limit']}" if self.table_infos['k_report_sequences']['limit'] else None
            sql_query_10k = " ".join(filter(None, (base_query_10k, random_query, limit_query_10k))) + ";"
            cursor_10k.execute(sql_query_10k)

        if self.use_ec:
            base_query_ec = "SELECT * FROM ec_sequences"
            limit_query_ec = f"LIMIT {self.table_infos['ec_sequences']['limit']}" if self.table_infos['ec_sequences']['limit'] else None
            sql_query_ec = " ".join(filter(None, (base_query_ec, random_query, limit_query_ec))) + ";"
            cursor_ec.execute(sql_query_ec)

        if self.use_esg:
            base_query_esg = "SELECT * FROM esg_sequences"
            limit_query_esg = f"LIMIT {self.table_infos['esg_sequences']['limit']}" if self.table_infos['esg_sequences']['limit'] else None
            sql_query_esg = " ".join(filter(None, (base_query_esg, random_query, limit_query_esg))) + ";"
            cursor_esg.execute(sql_query_esg)        

        tables_done = 0
        do_iteration = True
        while do_iteration:
            rnd_table_name = np.random.choice(table_names, p = table_probabilities)
            if rnd_table_name == "k_report_sequences":
                batch = cursor_10k.fetchmany(self.batch_size)
            elif rnd_table_name == "ec_sequences":
                batch = cursor_ec.fetchmany(self.batch_size)
            else:
                batch = cursor_esg.fetchmany(self.batch_size)
            if not(batch):
                print(f"\nRemoving table: {rnd_table_name}")
                table_names.remove(rnd_table_name)
                table_probabilities = [self.table_infos[table_name]["use_prob"] for table_name in table_names]
                table_probabilities = table_probabilities / np.sum(table_probabilities)
                tables_done += 1
                print("New table probabilities:")
                print(table_probabilities)
                if tables_done == len(self.table_infos.keys()):
                    do_iteration = False
                continue
            else:
                yield batch

        cursor_10k.close()
        cursor_ec.close()
        cursor_esg.close()
        conn.close()

