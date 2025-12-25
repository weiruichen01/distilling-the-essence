from abc import ABC, abstractmethod

import datasets
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from typing import Literal

from fmchisel.config import DataLoadingConfig
from fmchisel.data.collator import DataCollatorForCompletionOnlyLM

import json

_RETAIN_COLUMNS = {"input_ids", "attention_mask", "labels"}

class DataModule(pl.LightningDataModule, ABC):
    def __init__(self, tokenizer: AutoTokenizer, data_load_config: DataLoadingConfig):
        super().__init__()
        self.tokenizer = tokenizer
        self.data_path = data_load_config.data_path
        self.max_length = data_load_config.max_length
        self.batch_size = data_load_config.batch_size
        self.n_train = data_load_config.n_train
        self.n_val = data_load_config.n_val
        self.return_prompt_input_ids = data_load_config.return_prompt_input_ids

    @abstractmethod
    def formatting_func(self, example):
        pass

    def tokenize(self, example):
        outputs = self.tokenizer(
            self.formatting_func(example),
            truncation=True,
            padding=False,
            max_length=self.max_length,
        )
        return {
            "input_ids": outputs["input_ids"],
            "attention_mask": outputs["attention_mask"],
        }

    @abstractmethod
    def setup(self, stage) -> None:
        self.train_dataset = self.dataset["train"].map(
            self.tokenize,
            remove_columns=list(set(self.dataset["train"].column_names) - _RETAIN_COLUMNS),
            batched=True,
            batch_size=1,
        )
        self.val_dataset = self.dataset["test"].map(
            self.tokenize,
            remove_columns=list(set(self.dataset["test"].column_names) - _RETAIN_COLUMNS),
            batched=True,
            batch_size=1,
        )
        self.dataset = None

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collator,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collator,
        )

class OpenThoughtsModule(DataModule):
    def __init__(self, tokenizer: AutoTokenizer,
                 data_load_config: DataLoadingConfig):
        super().__init__(tokenizer, data_load_config)

        self.tokenizer = tokenizer
        self.half_keep_strategy = getattr(data_load_config, "half_keep_strategy", None)
        self.truncate_after_think_end_token = getattr(data_load_config, "truncate_after_think_end_token", False)
        if self.truncate_after_think_end_token:
            self.think_end_token_ids = self.tokenizer.encode(data_load_config.cot_end_token, add_special_tokens=False)
        else:
            self.think_end_token_ids = None

        self.collator = DataCollatorForCompletionOnlyLM(
                tokenizer=tokenizer,
                response_template=data_load_config.response_template,
                pad_to_multiple_of=16,
                return_prompt_input_ids=self.return_prompt_input_ids,
                section_inclusion=data_load_config.section_inclusion,
                random_mask_ratio=data_load_config.random_mask_ratio,
                included_first_x_percent=data_load_config.included_first_x_percent,
                half_keep_strategy=data_load_config.half_keep_strategy,
                cot_start_token = data_load_config.cot_start_token,
                cot_end_token = data_load_config.cot_end_token,
                is_reasoning_llm = data_load_config.is_reasoning_llm,
            )
        self.included_first_x_percent = data_load_config.included_first_x_percent

    def tokenize(self, example):
        formatted_texts = self.formatting_func(example)
        outputs = self.tokenizer(
            formatted_texts,
            truncation=False,
            padding=False,
            max_length=None,
        )

        input_ids      = outputs["input_ids"][0]
        attention_mask = outputs["attention_mask"][0]

        if self.truncate_after_think_end_token and self.think_end_token_ids:
            think_end_ids = self.think_end_token_ids
            for i in range(len(input_ids) - len(think_end_ids) + 1):
                if input_ids[i : i + len(think_end_ids)] == think_end_ids:
                    truncation_point = i + len(think_end_ids)
                    input_ids = input_ids[:truncation_point]
                    attention_mask = attention_mask[:truncation_point]
                    break
        if self.half_keep_strategy is not None:
            total_len = len(input_ids)
            if total_len == 0:
                kept_ids = input_ids
                kept_mask = attention_mask
            else:
                half_len = max(1, int(total_len * 0.5))
                if self.half_keep_strategy == "left":
                    start = 0
                elif self.half_keep_strategy == "middle":
                    start = max(0, (total_len - half_len) // 2)
                elif self.half_keep_strategy == "right":
                    start = max(0, total_len - half_len)
                else:
                    raise ValueError(f"Unknown half_keep_strategy: {self.half_keep_strategy}")
                end = start + half_len
                kept_ids = input_ids[start:end]
                kept_mask = attention_mask[start:end]
            input_ids = kept_ids
            attention_mask = kept_mask
        elif 0.0 < self.included_first_x_percent < 1.0:
            keep = int(len(input_ids) * self.included_first_x_percent)
            keep = max(1, keep)
            input_ids      = input_ids[:keep]
            attention_mask = attention_mask[:keep]

        return {
            "input_ids": [input_ids],
            "attention_mask": [attention_mask],
        }

    def formatting_func(self, example):
        '''Ignore sys msg, just use conversations'''
        conversations = example["conversations"]
        if isinstance(conversations, str):
            print('Note! Conversations are in str data type, not list')
            conversations = json.loads(conversations)
        messages = []
        for turn in conversations[0]:
            content = turn["value"]
            if turn["from"] == "user":
                messages.append({"role": "user", "content": content})
            elif turn["from"] == "assistant":
                content = content.replace("<|begin_of_solution|>\n\n", '').replace("\n\n<|end_of_solution|>", '')
                content = content.replace("<|begin_of_thought|>", "<think>").replace("<|end_of_thought|>", "</think>")
                messages.append({"role": "assistant", "content": content})
            else:
                raise ValueError(f"Unknown role {turn['from']} in conversation turn: {turn}")
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        return [text]

    def setup(self, stage) -> None:
        raw = datasets.load_dataset(self.data_path)
        try:
            from .included_indexes_openthoughts114k import indexes_of_ex_with_less_than_4k_tokens
        except ImportError:
            from fmchisel.data.included_indexes_openthoughts114k import indexes_of_ex_with_less_than_4k_tokens
        if "train" in raw and "test" in raw:
            train_split = raw["train"].select(indexes_of_ex_with_less_than_4k_tokens).select(range(self.n_train))
            test_split  = raw["test"].select(indexes_of_ex_with_less_than_4k_tokens).select(range(self.n_val))
        else:
            splits     = raw["train"].select(indexes_of_ex_with_less_than_4k_tokens).train_test_split(test_size=self.n_val, seed=42)
            train_split = splits["train"].select(range(self.n_train))
            test_split  = splits["test"]

        self.dataset = {"train": train_split, "test": test_split}
        super().setup(stage)


class BespokeStratosModule(DataModule):
    def __init__(self, tokenizer: AutoTokenizer,
                 data_load_config: DataLoadingConfig):
        super().__init__(tokenizer, data_load_config)

        self.tokenizer = tokenizer
        self.half_keep_strategy = getattr(data_load_config, "half_keep_strategy", None)
        self.truncate_after_think_end_token = getattr(data_load_config, "truncate_after_think_end_token", False)
        if self.truncate_after_think_end_token:
            self.think_end_token_ids = self.tokenizer.encode(data_load_config.cot_end_token, add_special_tokens=False)
        else:
            self.think_end_token_ids = None

        self.collator = DataCollatorForCompletionOnlyLM(
                tokenizer=tokenizer,
                response_template=data_load_config.response_template,
                pad_to_multiple_of=16,
                return_prompt_input_ids=self.return_prompt_input_ids,
                section_inclusion=data_load_config.section_inclusion,
                random_mask_ratio=data_load_config.random_mask_ratio,
                included_first_x_percent=data_load_config.included_first_x_percent,
                half_keep_strategy=data_load_config.half_keep_strategy,
                cot_start_token = data_load_config.cot_start_token,
                cot_end_token = data_load_config.cot_end_token,
                is_reasoning_llm = data_load_config.is_reasoning_llm,
            )
        self.included_first_x_percent = data_load_config.included_first_x_percent

    def tokenize(self, example):
        formatted_texts = self.formatting_func(example)
        outputs = self.tokenizer(
            formatted_texts,
            truncation=False,
            padding=False,
            max_length=None,
        )

        input_ids      = outputs["input_ids"][0]
        attention_mask = outputs["attention_mask"][0]

        if self.truncate_after_think_end_token and self.think_end_token_ids:
            think_end_ids = self.think_end_token_ids
            for i in range(len(input_ids) - len(think_end_ids) + 1):
                if input_ids[i : i + len(think_end_ids)] == think_end_ids:
                    truncation_point = i + len(think_end_ids)
                    print(f'Rightmost {len(input_ids) - truncation_point} of tokens will be truncated out of original {len(input_ids)} token')
                    input_ids = input_ids[:truncation_point]
                    attention_mask = attention_mask[:truncation_point]
                    break
        if self.half_keep_strategy is not None:
            total_len = len(input_ids)
            if total_len == 0:
                kept_ids = input_ids
                kept_mask = attention_mask
            else:
                half_len = max(1, int(total_len * 0.5))
                if self.half_keep_strategy == "left":
                    start = 0
                elif self.half_keep_strategy == "middle":
                    start = max(0, (total_len - half_len) // 2)
                elif self.half_keep_strategy == "right":
                    start = max(0, total_len - half_len)
                else:
                    raise ValueError(f"Unknown half_keep_strategy: {self.half_keep_strategy}")
                end = start + half_len
                kept_ids = input_ids[start:end]
                kept_mask = attention_mask[start:end]
            input_ids = kept_ids
            attention_mask = kept_mask
        elif 0.0 < self.included_first_x_percent < 1.0:
            keep = int(len(input_ids) * self.included_first_x_percent)
            keep = max(1, keep)
            input_ids      = input_ids[:keep]
            attention_mask = attention_mask[:keep]

        return {
            "input_ids": [input_ids],
            "attention_mask": [attention_mask],
        }

    def formatting_func(self, example):
        '''Ignore sys msg, just use conversations'''
        conversations = example["conversations"]
        if isinstance(conversations, str):
            print('Note! Conversations are in str data type, not list')
            conversations = json.loads(conversations)
        messages = []
        for turn in conversations[0]:
            content = turn["value"]
            if turn["from"] == "user":
                messages.append({"role": "user", "content": content})
            elif turn["from"] == "assistant":
                content = content.replace("<|begin_of_solution|>\n\n", '').replace("\n\n<|end_of_solution|>", '')
                content = content.replace("<|begin_of_thought|>", "<think>").replace("<|end_of_thought|>", "</think>")
                messages.append({"role": "assistant", "content": content})
            else:
                raise ValueError(f"Unknown role {turn['from']} in conversation turn: {turn}")
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        return [text]

    def setup(self, stage) -> None:
        raw = datasets.load_dataset(self.data_path)
        try:
            from .included_indexes_bespokeStratos17k import indexes_of_ex_with_less_than_4k_tokens_bespoke_stratos
        except ImportError:
            from fmchisel.data.included_indexes_bespokeStratos17k import indexes_of_ex_with_less_than_4k_tokens_bespoke_stratos
        if "train" in raw and "test" in raw:
            train_split = raw["train"].select(indexes_of_ex_with_less_than_4k_tokens_bespoke_stratos).select(range(self.n_train))
            test_split  = raw["test"].select(indexes_of_ex_with_less_than_4k_tokens_bespoke_stratos).select(range(self.n_val))
        else:
            splits     = raw["train"].select(indexes_of_ex_with_less_than_4k_tokens_bespoke_stratos).train_test_split(test_size=self.n_val, seed=42)
            train_split = splits["train"].select(range(self.n_train))
            test_split  = splits["test"]

        self.dataset = {"train": train_split, "test": test_split}
        super().setup(stage)


class SkyT117kModule(DataModule):
    def __init__(self, tokenizer: AutoTokenizer,
                 data_load_config: DataLoadingConfig):
        super().__init__(tokenizer, data_load_config)

        self.tokenizer = tokenizer
        self.half_keep_strategy = getattr(data_load_config, "half_keep_strategy", None)
        self.truncate_after_think_end_token = getattr(data_load_config, "truncate_after_think_end_token", False)
        if self.truncate_after_think_end_token:
            self.think_end_token_ids = self.tokenizer.encode(data_load_config.cot_end_token, add_special_tokens=False)
        else:
            self.think_end_token_ids = None

        self.collator = DataCollatorForCompletionOnlyLM(
                tokenizer=tokenizer,
                response_template=data_load_config.response_template,
                pad_to_multiple_of=16,
                return_prompt_input_ids=self.return_prompt_input_ids,
                section_inclusion=data_load_config.section_inclusion,
                random_mask_ratio=data_load_config.random_mask_ratio,
                included_first_x_percent=data_load_config.included_first_x_percent,
                half_keep_strategy=data_load_config.half_keep_strategy,
                cot_start_token = data_load_config.cot_start_token,
                cot_end_token = data_load_config.cot_end_token,
                is_reasoning_llm = data_load_config.is_reasoning_llm,
            )
        self.included_first_x_percent = data_load_config.included_first_x_percent

    def tokenize(self, example):
        formatted_texts = self.formatting_func(example)
        outputs = self.tokenizer(
            formatted_texts,
            truncation=False,
            padding=False,
            max_length=None,
        )

        input_ids      = outputs["input_ids"][0]
        attention_mask = outputs["attention_mask"][0]

        if self.truncate_after_think_end_token and self.think_end_token_ids:
            think_end_ids = self.think_end_token_ids
            for i in range(len(input_ids) - len(think_end_ids) + 1):
                if input_ids[i : i + len(think_end_ids)] == think_end_ids:
                    truncation_point = i + len(think_end_ids)
                    input_ids = input_ids[:truncation_point]
                    attention_mask = attention_mask[:truncation_point]
                    break
        if self.half_keep_strategy is not None:
            total_len = len(input_ids)
            if total_len == 0:
                kept_ids = input_ids
                kept_mask = attention_mask
            else:
                half_len = max(1, int(total_len * 0.5))
                if self.half_keep_strategy == "left":
                    start = 0
                elif self.half_keep_strategy == "middle":
                    start = max(0, (total_len - half_len) // 2)
                elif self.half_keep_strategy == "right":
                    start = max(0, total_len - half_len)
                else:
                    raise ValueError(f"Unknown half_keep_strategy: {self.half_keep_strategy}")
                end = start + half_len
                kept_ids = input_ids[start:end]
                kept_mask = attention_mask[start:end]
            input_ids = kept_ids
            attention_mask = kept_mask
        elif 0.0 < self.included_first_x_percent < 1.0:
            keep = int(len(input_ids) * self.included_first_x_percent)
            keep = max(1, keep)
            input_ids      = input_ids[:keep]
            attention_mask = attention_mask[:keep]

        return {
            "input_ids": [input_ids],
            "attention_mask": [attention_mask],
        }

    def formatting_func(self, example):
        '''Ignore sys msg, just use conversations'''
        conversations = example["conversations"]
        if isinstance(conversations, str):
            print('Note! Conversations are in str data type, not list')
            conversations = json.loads(conversations)
        messages = []
        for turn in conversations[0]:
            content = turn["value"]
            if turn["from"] == "user":
                messages.append({"role": "user", "content": content})
            elif turn["from"] == "assistant":
                content = content.replace("<|begin_of_solution|>\n\n", '').replace("\n\n<|end_of_solution|>", '')
                content = content.replace("<|begin_of_thought|>", "<think>").replace("<|end_of_thought|>", "</think>")
                messages.append({"role": "assistant", "content": content})
            else:
                raise ValueError(f"Unknown role {turn['from']} in conversation turn: {turn}")
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        return [text]

    def setup(self, stage) -> None:
        raw = datasets.load_dataset(self.data_path)
        try:
            from .included_index_skyt1_17k import indexes_of_ex_with_less_than_4k_tokens_skyt1_17k
        except ImportError:
            from fmchisel.data.included_index_skyt1_17k import indexes_of_ex_with_less_than_4k_tokens_skyt1_17k
        if "train" in raw and "test" in raw:
            train_split = raw["train"].select(indexes_of_ex_with_less_than_4k_tokens_skyt1_17k).select(range(self.n_train))
            test_split  = raw["test"].select(indexes_of_ex_with_less_than_4k_tokens_skyt1_17k).select(range(self.n_val))
        else:
            splits     = raw["train"].select(indexes_of_ex_with_less_than_4k_tokens_skyt1_17k).train_test_split(test_size=self.n_val, seed=42)
            train_split = splits["train"].select(range(self.n_train))
            test_split  = splits["test"]

        self.dataset = {"train": train_split, "test": test_split}
        super().setup(stage)