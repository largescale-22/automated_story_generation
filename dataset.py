# -*- coding: utf-8 -*-
# Implemented by Yoonseok Heo at 200423

from typing import Any, Dict, List, Optional, Sequence

import argparse
import copy
import json
import os
import os.path as osp
import random
import time
from multiprocessing import Pool
from xmlrpc.client import Boolean

import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer

from path.path import get_project_root
from utils_multiple_choice import InputFeatures

SPLIT_NAMES = ("train", "validation", "test")
CLS_TOKEN = "<CLS>"
EOS_TOKEN = "<EOS>"
PAD_TOKEN = "<PAD>"
SPECIAL_TOKENS = [
    "<|CHAR_NAME|>",
    "<|CHAR_DESCRIPTION|>",
    "<|ESTABLISHMENT|>",
]
NARRATOR_ID = "-1"
NARRATOR_NAME = "Narrator"
NARRATOR_DESC = ""
NARRATOR_DICT = {"char_id": NARRATOR_ID, "char_name": NARRATOR_NAME, "char_desc": NARRATOR_DESC}
NUM_EXAMPLES = {"train": 50000, "validation": 5000, "test": 2000}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="data/storium/")
    parser.add_argument("--cache_dir", type=str, default="caches")
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--max_prev_story_length", type=int, default=200)
    parser.add_argument("--max_establishment_length", type=int, default=200)
    parser.add_argument("--max_char_description_length", type=int, default=300)
    parser.add_argument("--trim", type=str, default="end")
    parser.add_argument("--seed", type=int, default=42)

    # Parse the arguments.
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    return args


class DataProcessor:
    """
    This class encapsulates the functionality needed to preprocess the Storium
    data in a format that we can use.
    """

    def __init__(self, args, tokenizer: PreTrainedTokenizer, max_seq_length: int = 800):
        self.args = args
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def process_story_file(self, filepath: str):

        toks = filepath.strip().split("/")
        filename = toks[-1]
        filename = "LS_" + filename
        toks[-1] = filename
        new_path = "/".join(toks)
        with open(new_path, "rt") as f:
            story = json.load(f)

        return self.process_story(story)

    def process_story(self, story: Dict[str, Any]) -> List:
        characters = story.get("characters")
        if len(characters) < 2:
            return -1

        scenes = story.get("scenes")

        character_pool, char_id_list = self.get_character_pool(character_list=characters)
        story_mcq_input_list = []
        for scene_id, scene in enumerate(scenes):
            for idx, entry in enumerate(scene):
                entry_dict = dict()
                previously_generated_story = entry["desc"]
                if previously_generated_story is None:
                    continue
                char_id = entry["label"]["char_id"] if "char_id" in entry["label"].keys() else NARRATOR_ID
                char_name = entry["label"]["char_name"] if "char_name" in entry["label"].keys() else NARRATOR_NAME
                char_desc = entry["label"]["char_desc"] if "char_desc" in entry["label"].keys() else NARRATOR_DESC
                estab_desc = entry["label"]["establishment_desc"]

                candidates, answer_idx = self.get_candidates(
                    character_pool=character_pool, character_id_list=char_id_list, cur_char_id=char_id
                )

                mcq_input_list = self.process_mcq_input_representation(
                    previously_generated_story, char_name, char_desc, estab_desc, candidates
                )
                entry_dict["mcq_input"] = mcq_input_list
                entry_dict["answer"] = answer_idx
                entry_dict["cur_seq_id"] = entry["cur_seq_id"]
                entry_dict["next_seq_id"] = entry["next_seq_id"]
                story_mcq_input_list.append(entry_dict)

        return story_mcq_input_list

    def process_mcq_input_representation(
        self,
        previously_generated_story: str,
        char_name: str,
        char_description: str,
        estab_description: str,
        candidates: List,
    ):
        prev_story = previously_generated_story.replace("\n\n", " ").replace("----", " ").strip()
        establishment_description = estab_description.replace("\n\n", " ").replace("----", " ").strip()

        # Trim previosuly generated story and establishment description
        num_words = len(prev_story.strip().split(" "))
        if num_words > self.args.max_prev_story_length:
            prev_story = self.trim_sentence(prev_story, max_size=self.args.max_prev_story_length, flag=self.args.trim)

        num_words = len(establishment_description.strip().split(" "))
        if num_words > self.args.max_establishment_length:
            establishment_description = self.trim_sentence(
                establishment_description, max_size=self.args.max_establishment_length, flag=self.args.trim
            )

        mcq_input_list = []
        for idx, choice in enumerate(candidates):
            mcq_input = prev_story
            mcq_input += SPECIAL_TOKENS[0]  # SPECIAL_TOKEN: <|CHAR_NAME|>
            mcq_input += choice["char_name"]
            mcq_input += SPECIAL_TOKENS[1]  # SPECIAL_TOKEN: <|CHAR_DESCRIPTION|>

            # Truncate char_description
            char_desc = choice["char_desc"].replace("\n\n", " ").replace("----", " ").strip()

            num_words = len(char_desc.split(" "))
            if num_words >= self.args.max_char_description_length:
                char_desc = self.trim_sentence(
                    char_desc, max_size=self.args.max_char_description_length, flag=self.args.trim
                )
            mcq_input += char_desc
            # mcq_input += SPECIAL_TOKENS[2]  # SPECIAL_TOKEN: <|ESTABLISHMENT|>
            # mcq_input += establishment_description

            mcq_input_list.append(mcq_input)

        return mcq_input_list

    def get_candidates(self, character_pool, character_id_list, cur_char_id):
        char_ids_list = copy.deepcopy(character_id_list)
        char_ids_list.remove(cur_char_id)
        num_chars = len(char_ids_list)

        if num_chars > 4:
            idx_list = np.arange(num_chars)
            choices = np.random.choice(idx_list, size=4, replace=False)

        elif num_chars == 4:
            choices = np.arange(num_chars)

        else:
            idx_list = np.arange(num_chars)
            choices = np.random.choice(idx_list, size=4, replace=True)

        candidates_ids_list = [char_ids_list[i] for i in choices]
        candidates_ids_list.append(cur_char_id)
        random.shuffle(candidates_ids_list)

        candidates = [character_pool[char_id] for char_id in candidates_ids_list]
        answer_idx = candidates_ids_list.index(cur_char_id)

        return candidates, answer_idx

    def get_character_pool(self, character_list):
        char_id_list = []
        char_id_dict = dict()
        for character in character_list:
            if character["char_desc"] is None:
                character["char_desc"] = ""
            char_id_dict[character["char_id"]] = character
            char_id_list.append(character["char_id"])

        # Error Handling: Narrator
        char_id_dict[NARRATOR_ID] = NARRATOR_DICT
        char_id_list.append(NARRATOR_ID)

        return char_id_dict, char_id_list

    def trim_sentence(self, info_str: str, max_size: int, flag: str) -> str:
        info_list = info_str.strip().split(" ")
        num_words = len(info_list)
        slice_size = num_words - max_size
        if flag.lower() == "start":
            new_list = info_list[slice_size:]

        elif flag.lower() == "end":
            new_list = info_list[:-slice_size]

        else:
            raise RuntimeError("Unknown trim type: start or end")

        return " ".join(new_list)


class SceneDataset(Dataset):
    def __init__(self, args, split: str, tokenizer: str, cache_dir: Optional[str] = None):
        self.args = args
        self.split = split.lower()
        self.cache_dir = cache_dir
        self.tokenizer = tokenizer
        self.entries: List[Dict[str, Any]] = []
        self.tensor_features = []

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        if isinstance(idx, Sequence):
            return [self.entries[i] for i in idx]

        return self.entries[idx]

    def process(self, filenames: List[str]):
        pool = Pool(10)
        start_time = time.time()
        results = []
        dataprocessor = DataProcessor(
            args=self.args,
            tokenizer=self.tokenizer,
            max_seq_length=self.args.max_seq_length,
        )

        for filename in filenames:
            results.append(pool.apply_async(type(self)._process, [filename, dataprocessor]))

        pool.close()

        self.entries = []
        for result in tqdm(
            results,
            unit="file",
            dynamic_ncols=True,
            desc=f"Processing {self.split} set",
        ):
            entries = result.get()
            if entries == -1:
                continue

            if not entries:
                continue

            for entry in entries:
                self.entries.append(entry)

        pool.join()

        end_time = time.time()
        print("  >> Data Preprocessing Time: {}".format(end_time - start_time))
        print("")

    @staticmethod
    def _process(filename: str, dataprocessor: DataProcessor):
        """
        Process a single file and return the resulting entries
        """
        entries = dataprocessor.process_story_file(filename)

        return entries

    def tensorize(self, split):
        num_examples = NUM_EXAMPLES[split]
        results = []

        pool = Pool(10)

        start_time = time.time()
        for entry in self.entries[:num_examples]:
            results.append(pool.apply_async(type(self)._tensorize, [entry, self.tokenizer, self.args.max_seq_length]))
        pool.close()

        self.tensor_features = []
        for result in tqdm(results, unit="file", dynamic_ncols=True, desc=f"Processing {self.split} Tensorization"):
            tensor_feat = result.get()
            self.tensor_features.append(tensor_feat)

        pool.join()
        end_time = time.time()
        print("  >> Tensorization Processing Time: {}".format(end_time - start_time))
        print("")

    @staticmethod
    def _tensorize(entry, tokenizer, max_seq_length):
        choices_features = list()
        mcq_inputs, label = entry["mcq_input"], entry["answer"]
        for choice in mcq_inputs:
            tokenized_example = tokenizer(choice, max_length=max_seq_length, padding="max_length", truncation=True)
            choices_features.append(
                (
                    tokenized_example["input_ids"],
                    tokenized_example["attention_mask"],
                    tokenized_example["token_type_ids"],
                )
            )

        return InputFeatures(example_id=entry["cur_seq_id"], choices_features=choices_features, label=label)


def select_field(features, field):
    return [[choice[field] for choice in feature.choices_features] for feature in features]


def perform_preprocessing(args):
    """
    Preprocess the dataset according to the passed in args
    """
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    tokenizer.add_special_tokens({"additional_special_tokens": list(SPECIAL_TOKENS)})

    for split in SPLIT_NAMES:  # train. valid. test
        root = get_project_root()
        with open(osp.join(root, args.data_dir, f"{split}_filenames.txt"), "rt") as file:
            filenames = [osp.join(root, args.data_dir, filename).strip() for filename in file.readlines()]

        cached_file_name = f"cached_{split}_{args.model_name_or_path}_{str(args.max_seq_length)}"
        cached_features_file = os.path.join(args.data_dir, cached_file_name)

        if os.path.exists(cached_features_file):
            print(f"Loading features from cached file {cached_features_file}")
            features = torch.load(cached_features_file)
        else:
            dataset = SceneDataset(args, split, tokenizer, cache_dir=args.cache_dir)
            dataset.process(filenames)
            dataset.tensorize(split)
            features = dataset.tensor_features
            print(f"Saving features into cached file {cached_features_file}")
            torch.save(features, cached_features_file)


def get_dataset(args, split, tokenizer):
    """
    Get train/valid/test dataset
    """
    root = get_project_root()
    with open(osp.join(root, args.data_dir, f"{split}_filenames.txt"), "rt") as file:
        filenames = [osp.join(root, args.data_dir, filename).strip() for filename in file.readlines()]

    cached_file_name = f"cached_{split}_{args.model_name_or_path}_{str(args.max_seq_length)}"
    cached_features_file = os.path.join(args.data_dir, cached_file_name)

    if os.path.exists(cached_features_file):
        print(f"Loading features from cached file {cached_features_file}")
        features = torch.load(cached_features_file)
    else:
        dataset = SceneDataset(args, split, tokenizer, cache_dir=args.cache_dir)
        dataset.process(filenames)
        dataset.tensorize(split)
        features = dataset.tensor_features
        print(f"Saving features into cached file {cached_features_file}")
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor(select_field(features, "input_ids"), dtype=torch.long)
    all_input_mask = torch.tensor(select_field(features, "input_mask"), dtype=torch.long)
    all_segment_ids = torch.tensor(select_field(features, "segment_ids"), dtype=torch.long)
    all_label_ids = torch.tensor([f.label for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset


if __name__ == "__main__":
    args = parse_args()
    perform_preprocessing(args)
