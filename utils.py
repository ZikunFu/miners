import datasets
import csv

class NusaXDataset():
    def __init__(self, prompt="", src_lang="eng", task="bitext"):
        self.all_data = {}
        self.train_data = {}
        self.valid_data = {}
        self.test_data = {}
        self.prompt = prompt
        self.src_lang = src_lang

        self.LANGS = []
        self.TARGET_LANGS = []
        self.LABELS = []
        self.LABELS_TO_TEXTS = {
            0:"negative", 1: "neutral", 2: "positive"
        }
        self.TEXT_LABELS = ["negative", "neutral", "positive"]

        if task == "bitext":
            self.create_bitext()
        elif task == "classification":
            self.create_classification()

    def create_classification(self):
        self.LANGS = ['ace', 'ban', 'bbc', 'bjn', 'bug', 'eng', 'ind', 'jav', 'mad', 'min', 'nij', 'sun']
        self.TARGET_LANGS = ['ace', 'ban', 'bbc', 'bjn', 'bug', 'eng', 'ind', 'jav', 'mad', 'min', 'nij', 'sun']
        self.LABELS = [0, 1, 2]

        for lang in self.LANGS:
            key = lang
            dataset = datasets.load_dataset("indonlp/NusaX-senti", lang)

            self.all_data[key] = {"source":[], "target":[], "target_text":[]}
            self.train_data[key] = {"source":[], "target":[], "target_text":[]}
            self.valid_data[key] = {"source":[], "target":[], "target_text":[]}
            self.test_data[key] = {"source":[], "target":[], "target_text":[]}

            for i in range(len(dataset["train"])):
                self.train_data[key]["source"].append(self.prompt + dataset["train"][i]["text"])
                self.train_data[key]["target"].append(int(dataset["train"][i]["label"]))
                self.train_data[key]["target_text"].append(self.LABELS_TO_TEXTS[dataset["train"][i]["label"]])
            for i in range(len(dataset["validation"])):
                self.valid_data[key]["source"].append(self.prompt + dataset["validation"][i]["text"])
                self.valid_data[key]["target"].append(int(dataset["validation"][i]["label"]))
                self.valid_data[key]["target_text"].append(self.LABELS_TO_TEXTS[dataset["validation"][i]["label"]])
            for i in range(len(dataset["test"])):
                self.test_data[key]["source"].append(self.prompt + dataset["test"][i]["text"])
                self.test_data[key]["target"].append(int(dataset["test"][i]["label"]))
                self.test_data[key]["target_text"].append(self.LABELS_TO_TEXTS[dataset["test"][i]["label"]])

    def create_bitext(self):
        self.LANGS = ['ace', 'ban', 'bbc', 'bjn', 'bug', 'eng', 'ind', 'jav', 'mad', 'min', 'nij', 'sun']
        self.TARGET_LANGS = []

        for lang in self.LANGS:
            if lang != self.src_lang:
                self.TARGET_LANGS.append(lang)
        
        for lang in self.TARGET_LANGS:
            dataset = datasets.load_dataset("indonlp/NusaX-senti", lang)
            key = self.src_lang + "_" + lang
            self.all_data[key] = {"source":[], "target":[]}
            self.train_data[key] = {"source":[], "target":[]}
            self.valid_data[key] = {"source":[], "target":[]}
            self.test_data[key] = {"source":[], "target":[]}

            for i in range(len(dataset["train"])):
                self.train_data[key]["target"].append(self.prompt + dataset["train"][i]["text"])
                self.all_data[key]["target"].append(self.prompt + dataset["train"][i]["text"])
            for i in range(len(dataset["validation"])):
                self.valid_data[key]["target"].append(self.prompt + dataset["validation"][i]["text"])
                self.all_data[key]["target"].append(self.prompt + dataset["validation"][i]["text"])
            for i in range(len(dataset["test"])):
                self.test_data[key]["target"].append(self.prompt + dataset["test"][i]["text"])
                self.all_data[key]["target"].append(self.prompt + dataset["test"][i]["text"])

            dataset = datasets.load_dataset("indonlp/NusaX-senti", self.src_lang)
            for i in range(len(dataset["train"])):
                self.train_data[key]["source"].append(self.prompt + dataset["train"][i]["text"])
                self.all_data[key]["source"].append(self.prompt + dataset["train"][i]["text"])
            for i in range(len(dataset["validation"])):
                self.valid_data[key]["source"].append(self.prompt + dataset["validation"][i]["text"])
                self.all_data[key]["source"].append(self.prompt + dataset["validation"][i]["text"])
            for i in range(len(dataset["test"])):
                self.test_data[key]["source"].append(self.prompt + dataset["test"][i]["text"])
                self.all_data[key]["source"].append(self.prompt + dataset["test"][i]["text"])


class NollySentiDataset():
    def __init__(self, prompt="", src_lang="eng", task="bitext"):
        self.all_data = {}
        self.train_data = {}
        self.valid_data = {}
        self.test_data = {}
        self.prompt = prompt
        self.src_lang = src_lang

        self.LANGS = []
        self.TARGET_LANGS = []
        self.LABELS = []
        self.LABELS_TO_TEXTS = {
            0:"negative", 1: "neutral", 2: "positive"
        }
        self.LABELS_TO_IDS = {
            "negative":0, "neutral":1, "positive":2
        }

        if task == "bitext":
            self.create_bitext()
        elif task == "classification":
            self.create_classification()

    def create_classification(self):
        self.LANGS = ['en', 'ha', 'ig', 'pcm', 'yo']
        self.TARGET_LANGS = ['en', 'ha', 'ig', 'pcm', 'yo']
        self.LABELS = ["negative", "neutral", "positive"]

        self.TEXT_LABELS = self.LABELS

        for lang in self.LANGS:
            key = lang
            print(key)
            train_url = f"datasets/nollysenti/{lang}/train.tsv"
            dev_url = f"datasets/nollysenti/{lang}/dev.tsv"
            test_url = f"datasets/nollysenti/{lang}/test.tsv"

            self.all_data[key] = {"source":[], "target":[], "target_text":[]}
            self.train_data[key] = {"source":[], "target":[], "target_text":[]}
            self.valid_data[key] = {"source":[], "target":[], "target_text":[]}
            self.test_data[key] = {"source":[], "target":[], "target_text":[]}

            with open(train_url, encoding="utf-8") as file:
                tsv_file = csv.reader(file, delimiter="\t")
                count = 0
                for arr in tsv_file:
                    count += 1
                    if count == 1:
                        continue
                    text, label = arr
                    self.train_data[key]["source"].append(self.prompt + text)
                    self.train_data[key]["target"].append(int(self.LABELS_TO_IDS[label]))
                    self.train_data[key]["target_text"].append(label)

            with open(dev_url, encoding="utf-8") as file:
                tsv_file = csv.reader(file, delimiter="\t")
                count = 0
                for arr in tsv_file:
                    count += 1
                    if count == 1:
                        continue
                    text, label = arr
                    self.valid_data[key]["source"].append(self.prompt + text)
                    self.valid_data[key]["target"].append(int(self.LABELS_TO_IDS[label]))
                    self.valid_data[key]["target_text"].append(label)

            with open(test_url, encoding="utf-8") as file:
                tsv_file = csv.reader(file, delimiter="\t")
                count = 0
                for arr in tsv_file:
                    count += 1
                    if count == 1:
                        continue
                    text, label = arr
                    self.test_data[key]["source"].append(self.prompt + text)
                    self.test_data[key]["target"].append(int(self.LABELS_TO_IDS[label]))
                    self.test_data[key]["target_text"].append(label)

    def create_bitext(self):
        self.LANGS = ['en400', 'ha', 'ig', 'pcm', 'yo400']
        self.TARGET_LANGS = [lang for lang in self.LANGS if lang != self.src_lang]
        for tgt_lang in self.TARGET_LANGS:
            key = f"{self.src_lang}_{tgt_lang}"
            #print(f"Processing language pair: {key}")
            self.all_data[key] = {"source": [], "target": []}
            self.train_data[key] = {"source": [], "target": []}
            self.valid_data[key] = {"source": [], "target": []}
            self.test_data[key] = {"source": [], "target": []}
            # Paths to source and target language files
            src_train_url = f"datasets/nollysenti/{self.src_lang}/train.tsv"
            tgt_train_url = f"datasets/nollysenti/{tgt_lang}/train.tsv"
            src_dev_url = f"datasets/nollysenti/{self.src_lang}/dev.tsv"
            tgt_dev_url = f"datasets/nollysenti/{tgt_lang}/dev.tsv"
            src_test_url = f"datasets/nollysenti/{self.src_lang}/test.tsv"
            tgt_test_url = f"datasets/nollysenti/{tgt_lang}/test.tsv"
            # Load training data
            self.load_parallel_data(src_train_url, tgt_train_url, self.train_data[key], self.all_data[key])
            # Load validation data
            self.load_parallel_data(src_dev_url, tgt_dev_url, self.valid_data[key], self.all_data[key])
            # Load test data
            self.load_parallel_data(src_test_url, tgt_test_url, self.test_data[key], self.all_data[key])

    def load_parallel_data(self, src_url, tgt_url, data_dict, all_data_dict):
        with open(src_url, encoding="utf-8") as src_file, open(tgt_url, encoding="utf-8") as tgt_file:
            src_reader = csv.reader(src_file, delimiter="\t")
            tgt_reader = csv.reader(tgt_file, delimiter="\t")

            # Skip headers
            next(src_reader)
            next(tgt_reader)

            for src_row, tgt_row in zip(src_reader, tgt_reader):
                src_text, _ = src_row
                tgt_text, _ = tgt_row
                data_dict["source"].append(src_text)
                data_dict["target"].append(tgt_text)
                all_data_dict["source"].append(src_text)
                all_data_dict["target"].append(tgt_text)

            
class NusaTranslationDataset():
    def __init__(self, prompt="", src_lang="ind"):
        self.all_data = {}
        self.train_data = {}
        self.valid_data = {}
        self.test_data = {}
        self.prompt = prompt
        self.src_lang = src_lang

        self.LANGS = ['abs', 'btk', 'bew', 'bhp', 'ind', 'jav', 'mad', 'mak', 'min', 'mui', 'rej', 'sun']
        self.TARGET_LANGS = []

        for lang in self.LANGS:
            if lang != src_lang:
                self.TARGET_LANGS.append(lang)
        
        for tgt_lang in self.TARGET_LANGS:
            dataset = datasets.load_dataset("indonlp/nusatranslation_mt", name=f'nusatranslation_mt_{src_lang}_{tgt_lang}_source')
            key = src_lang + "_" + tgt_lang
            self.all_data[key] = {"source":[], "target":[]}
            self.train_data[key] = {"source":[], "target":[]}
            self.valid_data[key] = {"source":[], "target":[]}
            self.test_data[key] = {"source":[], "target":[]}

            for i in range(len(dataset["train"])):
                self.train_data[key]["source"].append(self.prompt + dataset["train"][i]["text"])
                self.all_data[key]["source"].append(self.prompt + dataset["train"][i]["text"])

                self.train_data[key]["target"].append(self.prompt + dataset["train"][i]["label"])
                self.all_data[key]["target"].append(self.prompt + dataset["train"][i]["label"])

            for i in range(len(dataset["validation"])):
                self.valid_data[key]["source"].append(self.prompt + dataset["validation"][i]["text"])
                self.all_data[key]["source"].append(self.prompt + dataset["validation"][i]["text"])

                self.valid_data[key]["target"].append(self.prompt + dataset["validation"][i]["label"])
                self.all_data[key]["target"].append(self.prompt + dataset["validation"][i]["label"])

            for i in range(len(dataset["test"])):
                self.test_data[key]["source"].append(self.prompt + dataset["test"][i]["text"])
                self.all_data[key]["source"].append(self.prompt + dataset["test"][i]["text"])

                self.test_data[key]["target"].append(self.prompt + dataset["test"][i]["label"])
                self.all_data[key]["target"].append(self.prompt + dataset["test"][i]["label"])


class TatoebaDataset():
    def __init__(self, prompt="", src_lang="eng"):
        self.all_data = {}
        self.train_data = {}
        self.test_data = {}
        self.prompt = prompt

        self.LANGS = ["eng"]
        self.TARGET_LANGS = []

        dataset = datasets.load_dataset("mteb/tatoeba-bitext-mining")
        for i in range(len(dataset["test"])):
            lang = dataset["test"][i]["lang"]
            src_lang = lang.split("-")[0]
            tgt_lang = lang.split("-")[1]
            # if tgt_lang not in self.TARGET_LANGS:
            #     # self.TARGET_LANGS.append(tgt_lang)
            #     self.LANGS.append(tgt_lang)

            key = src_lang + "_" + tgt_lang

            if key not in self.test_data:
                self.test_data[key] = {"source":[], "target":[]}
                self.all_data[key] = {"source":[], "target":[]}

            self.test_data[key]["target"].append(self.prompt + dataset["test"][i]["sentence2"])
            self.all_data[key]["target"].append(self.prompt + dataset["test"][i]["sentence2"])

            self.test_data[key]["source"].append(self.prompt + dataset["test"][i]["sentence1"])
            self.all_data[key]["source"].append(self.prompt + dataset["test"][i]["sentence1"])
        self.train_data = self.all_data

class BUCCDataset():
    def __init__(self, prompt="", src_lang="en"):
        self.all_data = {}
        self.train_data = {}
        self.test_data = {}
        self.prompt = prompt

        self.LANGS = []
        self.TARGET_LANGS = ["en"]

        self.LANGS.append("en")

        dataset = datasets.load_dataset("mteb/bucc-bitext-mining")
        for i in range(len(dataset["test"])):
            lang = dataset["test"][i]["lang"]
            tgt_lang = lang.split("-")[1]
            src_lang = lang.split("-")[0]
            if tgt_lang not in self.TARGET_LANGS:
                # self.TARGET_LANGS.append(tgt_lang)
                self.LANGS.append(tgt_lang)

            # key = src_lang + "_" + tgt_lang

            # if key not in self.test_data:
            #     self.test_data[key] = {"source":[], "target":[]}
            #     self.all_data[key] = {"source":[], "target":[]}

            # self.test_data[key]["target"].append(self.prompt + dataset["test"][i]["sentence1"])
            # self.all_data[key]["target"].append(self.prompt + dataset["test"][i]["sentence1"])

            # self.test_data[key]["source"].append(self.prompt + dataset["test"][i]["sentence2"])
            # self.all_data[key]["source"].append(self.prompt + dataset["test"][i]["sentence2"])
    
            # other direction
            
            key = src_lang + "_" + tgt_lang

            if key not in self.test_data:
                self.test_data[key] = {"source":[], "target":[]}
                self.all_data[key] = {"source":[], "target":[]}

            self.test_data[key]["source"].append(self.prompt + dataset["test"][i]["sentence1"])
            self.all_data[key]["source"].append(self.prompt + dataset["test"][i]["sentence1"])

            self.test_data[key]["target"].append(self.prompt + dataset["test"][i]["sentence2"])
            self.all_data[key]["target"].append(self.prompt + dataset["test"][i]["sentence2"])
        self.train_data = self.all_data
        
class OpenSubtitlesDataset():
    def __init__(self, prompt="", src_lang="af"):
        self.all_data = {}
        self.train_data = {}
        self.test_data = {}
        self.prompt = prompt

        self.LANGS = []
        self.TARGET_LANGS = ["en"]

        # Add source language
        self.LANGS.append("en")

        # Load the dataset
        dataset = datasets.load_dataset("loicmagne/open-subtitles-bitext-mining", "af-en")
        
        tgt_lang="en"
        # Prepare keys and structure for language pairs
        key = src_lang + "_" + tgt_lang

        # Initialize the dictionary for storing source and target sentences
        if key not in self.train_data:
            self.train_data[key] = {"source": [], "target": []}
            self.all_data[key] = {"source": [], "target": []}

        # Loop through the dataset
        for i in range(len(dataset["train"])):
            sentence1 = self.prompt + dataset["train"][i]["sentence1"]
            sentence2 = self.prompt + dataset["train"][i]["sentence2"]

            # Add sentence1 and sentence2 to lists
            self.train_data[key]["source"].append(sentence1)
            self.all_data[key]["source"].append(sentence1)

            self.train_data[key]["target"].append(sentence2)
            self.all_data[key]["target"].append(sentence2)

        # Set the train data to all_data, as test data is not available in this dataset
        self.train_data = self.all_data

class MassiveIntentDataset():
    def __init__(self, prompt="", src_lang=""):
        self.all_data = {}
        self.train_data = {}
        self.valid_data = {}
        self.test_data = {}
        self.prompt = prompt
        self.src_lang = src_lang

        self.LANGS = []
        self.LANGS_MAP = {}
        self.LABELS = []
        self.LABELS_MAP = {}
        self.TEXT_LABELS = []

        dataset = datasets.load_dataset("mteb/amazon_massive_intent")
        for i in range(len(dataset["train"])):
            label = dataset["train"][i]["label"]
            if label not in self.LABELS_MAP:
                self.LABELS_MAP[label] = len(self.LABELS_MAP)
                self.LABELS.append(label)
            lang = dataset["train"][i]["lang"]
            if lang not in self.LANGS_MAP:
                self.LANGS_MAP[lang] = len(self.LANGS_MAP)
                self.LANGS.append(lang)
        for i in range(len(dataset["validation"])):
            label = dataset["validation"][i]["label"]
            if label not in self.LABELS_MAP:
                self.LABELS_MAP[label] = len(self.LABELS_MAP)
                self.LABELS.append(label)
            lang = dataset["validation"][i]["lang"]
            if lang not in self.LANGS_MAP:
                self.LANGS_MAP[lang] = len(self.LANGS_MAP)
                self.LANGS.append(lang)
        for i in range(len(dataset["test"])):
            label = dataset["test"][i]["label"]
            if label not in self.LABELS_MAP:
                self.LABELS_MAP[label] = len(self.LABELS_MAP)
                self.LABELS.append(label)
            lang = dataset["test"][i]["lang"]
            if lang not in self.LANGS_MAP:
                self.LANGS_MAP[lang] = len(self.LANGS_MAP)
                self.LANGS.append(lang)

        self.TEXT_LABELS = self.LABELS
        print(self.LANGS)

        for lang in self.LANGS:
            self.all_data[lang] = {"source":[], "target":[], "target_text":[]}
            self.train_data[lang] = {"source":[], "target":[], "target_text":[]}
            self.valid_data[lang] = {"source":[], "target":[], "target_text":[]}
            self.test_data[lang] = {"source":[], "target":[], "target_text":[]}
        
        for i in range(len(dataset["train"])):
            lang = dataset["train"][i]["lang"]
            self.train_data[lang]["source"].append(self.prompt + dataset["train"][i]["text"])
            self.train_data[lang]["target"].append(int(self.LABELS_MAP[dataset["train"][i]["label"]]))
            self.train_data[lang]["target_text"].append(dataset["train"][i]["label"].replace("_"," "))

        for i in range(len(dataset["validation"])):
            lang = dataset["validation"][i]["lang"]
            self.valid_data[lang]["source"].append(self.prompt + dataset["validation"][i]["text"])
            self.valid_data[lang]["target"].append(int(self.LABELS_MAP[dataset["validation"][i]["label"]]))
            self.valid_data[lang]["target_text"].append(dataset["validation"][i]["label"].replace("_"," "))

        for i in range(len(dataset["test"])):
            lang = dataset["test"][i]["lang"]
            self.test_data[lang]["source"].append(self.prompt + dataset["test"][i]["text"])
            self.test_data[lang]["target"].append(int(self.LABELS_MAP[dataset["test"][i]["label"]]))
            self.test_data[lang]["target_text"].append(dataset["test"][i]["label"].replace("_"," "))

        # remove underscore
        for l in range(len(self.TEXT_LABELS)):
            self.TEXT_LABELS[l] = self.TEXT_LABELS[l].replace("_"," ")
        print(self.TEXT_LABELS)
   

class Sib200Dataset():
    def __init__(self, prompt="", src_lang=""):
        self.all_data = {}
        self.train_data = {}
        self.valid_data = {}
        self.test_data = {}
        self.prompt = prompt
        self.src_lang = src_lang

        self.LANGS = []
        self.LANGS_MAP = {}
        self.LABELS = []
        self.LABELS_MAP = {}

        self.LANGS_FAMILY_MAP = {"Afro-Asiatic": ["aeb_Arab", "amh_Ethi", "ary_Arab", "arz_Arab", "gaz_Latn", "hau_Latn", "kab_Latn", "som_Latn", "taq_Latn", "taq_Tfng", "tir_Ethi", "tzm_Tfng", "acm_Arab", "acq_Arab", "ajp_Arab", "apc_Arab", "arb_Arab", "arb_Latn", "ars_Arab", "heb_Hebr", "mlt_Latn"],
            "Indo-European": ["afr_Latn", "kea_Latn", "hat_Latn", "pap_Latn", "ckb_Arab", "hye_Armn", "kmr_Latn", "pbt_Arab", "pes_Arab", "prs_Arab", "tgk_Cyrl", "asm_Beng", "awa_Deva", "ben_Beng", "bho_Deva", "guj_Gujr", "hin_Deva", "hne_Deva", "kas_Arab", "kas_Deva", "mag_Deva", "mai_Deva", "mar_Deva", "npi_Deva", "ory_Orya", "pan_Guru", "san_Deva", "sin_Sinh", "snd_Arab", "urd_Arab", "als_Latn", "ast_Latn", "bos_Latn", "cat_Latn", "cym_Latn", "dan_Latn", "deu_Latn", "ell_Grek", "eng_Latn", "fao_Latn", "fra_Latn", "fur_Latn", "gla_Latn", "gle_Latn", "glg_Latn", "isl_Latn", "ita_Latn", "lij_Latn", "lim_Latn", "lit_Latn", "lmo_Latn", "ltz_Latn", "nld_Latn", "nno_Latn", "nob_Latn", "oci_Latn", "por_Latn", "scn_Latn", "slv_Latn", "spa_Latn", "srd_Latn", "swe_Latn", "vec_Latn", "bel_Cyrl", "bul_Cyrl", "ces_Latn", "hrv_Latn", "ltg_Latn", "lvs_Latn", "mkd_Cyrl", "pol_Latn", "ron_Latn", "rus_Cyrl", "slk_Latn", "srp_Cyrl", "szl_Latn", "ukr_Cyrl", "ydd_Hebr", "tpi_Latn"],
            "Atlantic-Congo": ["aka_Latn", "bem_Latn", "cjk_Latn", "ewe_Latn", "fon_Latn", "fuv_Latn", "ibo_Latn", "kam_Latn", "kbp_Latn", "kik_Latn", "kin_Latn", "kmb_Latn", "kon_Latn", "lin_Latn", "lua_Latn", "lug_Latn", "mos_Latn", "nqo_Nkoo", "nso_Latn", "nya_Latn", "run_Latn", "sag_Latn", "sna_Latn", "sot_Latn", "ssw_Latn", "swh_Latn", "tsn_Latn", "tso_Latn", "tum_Latn", "twi_Latn", "umb_Latn", "wol_Latn", "xho_Latn", "yor_Latn", "zul_Latn"],
            "Mande": ["bam_Latn", "dyu_Latn"],
            "Nilotic": ["dik_Latn", "knc_Arab", "knc_Latn", "luo_Latn", "nus_Latn"],
            "Austronesian": ["plt_Latn", "ace_Arab", "ace_Latn", "ban_Latn", "bjn_Arab", "bjn_Latn", "bug_Latn", "ceb_Latn", "ilo_Latn", "ind_Latn", "jav_Latn", "min_Arab", "min_Latn", "pag_Latn", "sun_Latn", "tgl_Latn", "war_Latn", "zsm_Latn", "fij_Latn", "mri_Latn", "smo_Latn"],
            "Aymaran": ["ayr_Latn"],
            "Tupian": ["grn_Latn"],
            "Quechuan": ["quy_Latn"],
            "Turkic": ["azb_Arab", "azj_Latn", "kaz_Cyrl", "kir_Cyrl", "tuk_Latn", "tur_Latn", "uig_Arab", "uzn_Latn", "bak_Cyrl", "crh_Latn", "tat_Cyrl"],
            "Kartvelian": ["kat_Geor"],
            "Sino-Tibetan": ["dzo_Tibt", "lus_Latn", "mni_Beng", "bod_Tibt", "kac_Latn", "mya_Mymr", "yue_Hant", "zho_Hans", "zho_Hant"],
            "Dravidian": ["kan_Knda", "mal_Mlym", "tam_Taml", "tel_Telu"],
            "Austroasiatic": ["sat_Olck", "khm_Khmr", "vie_Latn"],
            "Japonic": ["jpn_Jpan"],
            "Mongolic-Khitan": ["khk_Cyrl"],
            "Koreanic": ["kor_Hang"],
            "Tai-Kadai": ["lao_Laoo", "shn_Mymr", "tha_Thai"],
            "Uralic": ["est_Latn", "fin_Latn", "hun_Latn"],
            "Basque": ["eus_Latn"],
            "Constructed": ["epo_Latn"]
        }

        self.LANGS_TO_LANGS_FAMILY_ID = {}
        self.LANGS_TO_LANGS_FAMILY = {}
        self.LANGS_FAMILY_TO_LANGS_FAMILY_ID = {}
        
        for lang_family in self.LANGS_FAMILY_MAP:
            self.LANGS_FAMILY_TO_LANGS_FAMILY_ID[lang_family] = len(self.LANGS_FAMILY_TO_LANGS_FAMILY_ID)

        for lang_family in self.LANGS_FAMILY_MAP:
            lang_family_id = self.LANGS_FAMILY_TO_LANGS_FAMILY_ID[lang_family]
            
            for lang in self.LANGS_FAMILY_MAP[lang_family]:
                self.LANGS_TO_LANGS_FAMILY[lang] = lang_family
                self.LANGS_TO_LANGS_FAMILY_ID[lang] = lang_family_id

        print(self.LANGS_TO_LANGS_FAMILY_ID)

        self.TEXT_LABELS = []
        # ['geography', 'science/technology', 'entertainment', 'politics', 'health', 'travel', 'sports']

        dataset = datasets.load_dataset("mteb/sib200")
        for i in range(len(dataset["train"])):
            label = dataset["train"][i]["category"]
            if label not in self.LABELS_MAP:
                self.LABELS_MAP[label] = len(self.LABELS_MAP)
                self.LABELS.append(label)
            lang = dataset["train"][i]["lang"]
            if lang not in self.LANGS_MAP:
                self.LANGS_MAP[lang] = len(self.LANGS_MAP)
                self.LANGS.append(lang)
        for i in range(len(dataset["validation"])):
            label = dataset["validation"][i]["category"]
            if label not in self.LABELS_MAP:
                self.LABELS_MAP[label] = len(self.LABELS_MAP)
                self.LABELS.append(label)
            lang = dataset["validation"][i]["lang"]
            if lang not in self.LANGS_MAP:
                self.LANGS_MAP[lang] = len(self.LANGS_MAP)
                self.LANGS.append(lang)
        for i in range(len(dataset["test"])):
            label = dataset["test"][i]["category"]
            if label not in self.LABELS_MAP:
                self.LABELS_MAP[label] = len(self.LABELS_MAP)
                self.LABELS.append(label)
            lang = dataset["test"][i]["lang"]
            if lang not in self.LANGS_MAP:
                self.LANGS_MAP[lang] = len(self.LANGS_MAP)
                self.LANGS.append(lang)

        self.TEXT_LABELS = self.LABELS
        print(self.LANGS)
        print(self.TEXT_LABELS)

        for lang in self.LANGS:
            self.all_data[lang] = {"source":[], "target":[], "target_text":[]}
            self.train_data[lang] = {"source":[], "target":[], "target_text":[]}
            self.valid_data[lang] = {"source":[], "target":[], "target_text":[]}
            self.test_data[lang] = {"source":[], "target":[], "target_text":[]}
        
        for i in range(len(dataset["train"])):
            lang = dataset["train"][i]["lang"]
            self.train_data[lang]["source"].append(self.prompt + dataset["train"][i]["text"])
            self.train_data[lang]["target"].append(int(self.LABELS_MAP[dataset["train"][i]["category"]]))
            self.train_data[lang]["target_text"].append(dataset["train"][i]["category"])

        for i in range(len(dataset["validation"])):
            lang = dataset["validation"][i]["lang"]
            self.valid_data[lang]["source"].append(self.prompt + dataset["validation"][i]["text"])
            self.valid_data[lang]["target"].append(int(self.LABELS_MAP[dataset["validation"][i]["category"]]))
            self.valid_data[lang]["target_text"].append(dataset["validation"][i]["category"])

        for i in range(len(dataset["test"])):
            lang = dataset["test"][i]["lang"]
            self.test_data[lang]["source"].append(self.prompt + dataset["test"][i]["text"])
            self.test_data[lang]["target"].append(int(self.LABELS_MAP[dataset["test"][i]["category"]]))
            self.test_data[lang]["target_text"].append(dataset["test"][i]["category"])
   

class LinceSADataset():
    def __init__(self, prompt="", src_lang="spaeng"):
        self.all_data = {}
        self.train_data = {}
        self.valid_data = {}
        self.test_data = {}
        self.prompt = prompt
        self.src_lang = src_lang

        self.LANGS = ["spaeng"]
        self.LABELS = ["negative","neutral","positive"]
        self.TEXT_LABELS = ["negative","neutral","positive"]
        self.LABELS_MAP = {}
        for i in range(len(self.LABELS)):
            self.LABELS_MAP[self.LABELS[i]] = i
            

        dataset = datasets.load_dataset("lince", "sa_spaeng")

        key = self.src_lang
        
        self.all_data[key] = {"source":[], "target":[], "target_text":[]}
        self.train_data[key] = {"source":[], "target":[], "target_text":[]}
        self.valid_data[key] = {"source":[], "target":[], "target_text":[]}
        self.test_data[key] = {"source":[], "target":[], "target_text":[]}

        for i in range(len(dataset["train"])):
            self.train_data[key]["source"].append(self.prompt + " ".join(dataset["train"][i]["words"]))
            self.train_data[key]["target"].append(int(self.LABELS_MAP[dataset["train"][i]["sa"]]))
            self.train_data[key]["target_text"].append(dataset["train"][i]["sa"])
        for i in range(len(dataset["validation"])):
            self.valid_data[key]["source"].append(self.prompt + " ".join(dataset["validation"][i]["words"]))
            self.valid_data[key]["target"].append(int(self.LABELS_MAP[dataset["validation"][i]["sa"]]))
            self.valid_data[key]["target_text"].append(dataset["validation"][i]["sa"])

            self.test_data[key]["source"].append(self.prompt + " ".join(dataset["validation"][i]["words"]))
            self.test_data[key]["target"].append(int(self.LABELS_MAP[dataset["validation"][i]["sa"]]))
            self.test_data[key]["target_text"].append(dataset["validation"][i]["sa"])


class LinceMTDataset():
    def __init__(self, prompt="", src_lang="eng"):
        self.all_data = {}
        self.train_data = {}
        self.all_data["eng_hinglish"] = {"source":[], "target":[]}
        self.prompt = prompt

        self.LANGS = ["eng", "hinglish"]
        self.TARGET_LANGS = ["hinglish"]

        with open("datasets/lince_mt_eng_hinglish/train.txt") as f:
            for line in f:
                eng_str, hinglish_str = line.replace("\n","").split("\t")
                key = "eng_hinglish"

                self.all_data[key]["target"].append(self.prompt + hinglish_str)
                self.all_data[key]["source"].append(self.prompt + eng_str)
        
        with open("datasets/lince_mt_eng_hinglish/dev.txt") as f:
            for line in f:
                eng_str, hinglish_str = line.replace("\n","").split("\t")
                key = "eng_hinglish"

                self.all_data[key]["target"].append(self.prompt + hinglish_str)
                self.all_data[key]["source"].append(self.prompt + eng_str)
        self.train_data = self.all_data


class PhincDataset():
    def __init__(self, prompt="", src_lang="eng"):
        self.all_data = {}
        self.all_data["eng_hinglish"] = {"source":[], "target":[]}
        self.train_data = {}
        self.prompt = prompt

        self.LANGS = ["eng", "hinglish"]
        self.TARGET_LANGS = ["hinglish"]

        with open("datasets/phinc/train.tsv") as f:
            tsv_file = csv.reader(f, delimiter="\t")
     
            key = "eng_hinglish"
            count = 0

            # printing data line by line
            for arr in tsv_file:
                count += 1
                if count == 1:
                    continue

                hinglish_str, eng_str = arr

                key = "eng_hinglish"

                self.all_data[key]["target"].append(self.prompt + hinglish_str)
                self.all_data[key]["source"].append(self.prompt + eng_str)
        self.train_data = self.all_data


class MTOPIntentDataset():
    def __init__(self, prompt="", src_lang="en"):
        self.all_data = {}
        self.train_data = {}
        self.valid_data = {}
        self.test_data = {}
        self.prompt = prompt
        self.src_lang = src_lang

        self.LANGS = ["de", "en", "es", "fr", "hi", "th"]
        self.LABELS = []
        self.LABELS_MAP = {}      

        for lang in self.LANGS:
            dataset = datasets.load_dataset("mteb/mtop_intent", lang)

            key = lang
        
            self.all_data[key] = {"source":[], "target":[]}
            self.train_data[key] = {"source":[], "target":[]}
            self.valid_data[key] = {"source":[], "target":[]}
            self.test_data[key] = {"source":[], "target":[]}

            for i in range(len(dataset["train"])):
                self.train_data[key]["source"].append(self.prompt + " ".join(dataset["train"][i]["text"]))
                self.train_data[key]["target"].append(int(dataset["train"][i]["label"]))
            for i in range(len(dataset["validation"])):
                self.valid_data[key]["source"].append(self.prompt + " ".join(dataset["validation"][i]["text"]))
                self.valid_data[key]["target"].append(int(dataset["validation"][i]["label"]))
            for i in range(len(dataset["test"])):
                self.test_data[key]["source"].append(self.prompt + " ".join(dataset["test"][i]["text"]))
                self.test_data[key]["target"].append(int(dataset["test"][i]["label"]))


class FIREDataset():
    def __init__(self, prompt="", src_lang="malayalam"):
        self.prompt = prompt

        self.LABELS = ["Positive", "Negative", "Mixed", "Unknown"]
        self.TEXT_LABELS = self.LABELS
        self.LANGS = ["malayalam", "tamil"]
        self.TARGET_LANGS = [""]

        self.train_data = {}
        self.valid_data = {}
        self.test_data = {}
        
        for lang in self.LANGS:
            key = lang
            self.train_data[key] = {"source":[], "target":[], "target_text":[]}
            self.valid_data[key] = {"source":[], "target":[], "target_text":[]}
            self.test_data[key] = {"source":[], "target":[], "target_text":[]}

            for split in ["train", "dev", "test"]:
                with open(f"datasets/fire_{lang}/{lang}_{split}.tsv") as f:
                    tsv_file = csv.reader(f, delimiter="\t")
             
                    count = 0
                    # printing data line by line
                    for arr in tsv_file:
                        count += 1
                        if count == 1:
                            continue

                        if split == "test":
                            _, text, label = arr
                        else:
                            text, label = arr
                        label = label.strip()

                        if label == "Mixed_feelings":
                            # print("> rename label", label)
                            label = "Mixed"
                        if label == "not-malayalam" or label == "not-tamil" or label == "unknown_state":
                            # print("> rename unknown", label)
                            label = "Unknown"
                        
                        if split == "train":
                            self.train_data[key]["target"].append(label)
                            self.train_data[key]["target_text"].append(label)
                            self.train_data[key]["source"].append(self.prompt + text)
                        elif split == "dev":
                            self.valid_data[key]["target"].append(label)
                            self.valid_data[key]["target_text"].append(label)
                            self.valid_data[key]["source"].append(self.prompt + text)
                        elif split == "test":
                            self.test_data[key]["target"].append(label)
                            self.test_data[key]["target_text"].append(label)
                            self.test_data[key]["source"].append(self.prompt + text)
            print(len(self.train_data[key]["source"]), len(self.valid_data[key]["source"]), len(self.test_data[key]["source"]))

#UniversalNER
#MASSIVE (Slot Filling)

class MasakhaNERDataset:
    def __init__(self,sample_size=0):
        self.all_data = {}
        self.train_data = {}
        self.valid_data = {}
        self.test_data = {}
        self.sample_size=sample_size
        self.LANGS = [
            'bam', 'bbj', 'ewe', 'fon', 'hau', 'ibo', 'kin', 'lug', 'luo', 'mos',
            'nya', 'pcm', 'sna', 'swa', 'tsn', 'twi', 'wol', 'xho', 'yor', 'zul'
        ]
        self.LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-DATE", "I-DATE"]
        self.load_data()
        
    def convert_ner_tags(self, ner_tags, to_BIO=True):
        # Define the mapping from integer tags to string labels
        tag_to_label = {
            0: "O",
            1: "B-PER",
            2: "I-PER",
            3: "B-ORG",
            4: "I-ORG",
            5: "B-LOC",
            6: "I-LOC",
            7: "B-DATE",
            8: "I-DATE"
        }
    
        # Create the reverse mapping from string labels to integer tags
        label_to_tag = {label: tag for tag, label in tag_to_label.items()}
    
        if to_BIO:
        # Convert integer tags to string labels
            return [tag_to_label[tag] for tag in ner_tags]
        else:
            # Convert string labels to integer tags
            return [label_to_tag[label] for label in ner_tags]

    def load_data(self):
            for lang in self.LANGS:
                dataset = datasets.load_dataset('masakhane/masakhaner2', lang)
                # Load samples based on sample_size argument
                if self.sample_size>0:
                    self.train_data[lang] = dataset['train'].select(range(min(self.sample_size, len(dataset['train']))))
                    self.valid_data[lang] = dataset['validation'].select(range(min(self.sample_size, len(dataset['validation']))))
                    self.test_data[lang] = dataset['test'].select(range(min(self.sample_size, len(dataset['test']))))
                else:
                    self.train_data[lang] = dataset['train']
                    self.valid_data[lang] = dataset['validation']
                    self.test_data[lang] = dataset['test']
            self.all_data = dataset
            total_train_samples = sum([len(data) for data in self.train_data.values()])
            total_valid_samples = sum([len(data) for data in self.valid_data.values()])
            total_test_samples = sum([len(data) for data in self.test_data.values()])
            print(f"Total training samples: {total_train_samples}")
            print(f"Total validation samples: {total_valid_samples}")
            print(f"Total test samples: {total_test_samples}")


class XLSumDataset:
    def __init__(self, sample_size=0):
        self.all_data = {}
        self.train_data = {}
        self.valid_data = {}
        self.test_data = {}
        self.sample_size = sample_size

        # Update with correct language keys as per XLSum documentation
        self.LANGS = [
    "amharic", "arabic", "azerbaijani", "bengali", "burmese", 
    "chinese_simplified", "chinese_traditional", "english", "french", 
    "gujarati", "hausa", "hindi", "igbo", "indonesian", "japanese", 
    "kirundi", "korean", "kyrgyz", "marathi", "nepali", "oromo", "pashto", 
    "persian", "pidgin", "portuguese", "punjabi", "russian", 
    "scottish_gaelic", "serbian_cyrillic", "serbian_latin", "sinhala", 
    "somali", "spanish", "swahili", "tamil", "telugu", "thai", "tigrinya", 
    "turkish", "ukrainian", "urdu", "uzbek", "vietnamese", "welsh", "yoruba"
]

        self.load_data()

    def load_data(self):
        for lang in self.LANGS:
            # Load the dataset for each language
            dataset = datasets.load_dataset("csebuetnlp/xlsum", lang)

            # Prepare train, validation, and test data structures
            self.all_data[lang] = {"source": [], "target": []}
            self.train_data[lang] = {"source": [], "target": []}
            self.valid_data[lang] = {"source": [], "target": []}
            self.test_data[lang] = {"source": [], "target": []}

            # Load samples based on sample_size argument
            splits = ["train", "validation", "test"]
            for split in splits:
                split_data = dataset[split]
                num_samples = self.sample_size if self.sample_size > 0 else len(split_data)

                for i in range(num_samples):
                    text = split_data[i]["text"]
                    summary = split_data[i]["summary"]
                    self.all_data[lang]["source"].append(text)
                    self.all_data[lang]["target"].append(summary)
                    
                    if split == "train":
                        self.train_data[lang]["source"].append(text)
                        self.train_data[lang]["target"].append(summary)
                    elif split == "validation":
                        self.valid_data[lang]["source"].append(text)
                        self.valid_data[lang]["target"].append(summary)
                    elif split == "test":
                        self.test_data[lang]["source"].append(text)
                        self.test_data[lang]["target"].append(summary)

        print(f"Total languages loaded: {len(self.LANGS)}")
        print("Sample data loaded for 'english':", self.train_data["english"]["source"][:3])
