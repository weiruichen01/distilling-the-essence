from __future__ import annotations
import warnings
from typing import Any, Dict, List, Union

import numpy as np
import torch
from transformers.data.data_collator import DataCollatorForLanguageModeling

from typing import Literal
import time
import torch, torch.distributed as dist
import math

def is_positive(x, abs_tol: float = 1e-12):
    return x > 0.0 and not math.isclose(x, 0.0, rel_tol=0.0, abs_tol=abs_tol)

def mask_fraction_to_ignore(
    t: torch.Tensor,
    frac: float,
    ignore_index: int = -100,
    rounding: str = "round",
    in_place: bool = True,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if not (0.0 <= frac <= 1.0):
        raise ValueError("frac must be in [0, 1].")

    valid = (t != ignore_index)
    num_valid = int(valid.sum().item())
    if num_valid == 0 or frac == 0.0:
        return t if in_place else t.clone()

    if rounding == "round":
        k = int(round(frac * num_valid))
    elif rounding == "floor":
        k = int(math.floor(frac * num_valid))
    elif rounding == "ceil":
        k = int(math.ceil(frac * num_valid))
    else:
        raise ValueError("rounding must be one of {'round','floor','ceil'}")

    k = max(0, min(k, num_valid))
    if k == 0:
        return t if in_place else t.clone()


    valid_idx = valid.view(-1).nonzero(as_tuple=False).squeeze(1)

    perm = torch.randperm(valid_idx.numel(), device=t.device, generator=generator)
    chosen = valid_idx[perm[:k]]

    out = t if in_place else t.clone()
    out.view(-1)[chosen] = ignore_index
    return out


class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    def __init__(
        self,
        response_template: Union[str, List[int]],
        *args,
        mlm: bool = False,
        ignore_index: int = -100,
        return_prompt_input_ids=False,
        section_inclusion: Literal['promptcotAns', 'cotAns', 'ans', 'promptAns'],
        random_mask_ratio: float = 0.0,
        included_first_x_percent: float = 1.0,
        half_keep_strategy: Literal['left', 'middle', 'right', None] = None,
        cot_start_token: str = '<think>',
        cot_end_token: str = '</think>',
        is_reasoning_llm: bool = True,
        **kwargs,
    ):
        super().__init__(*args, mlm=mlm, **kwargs)
        if not is_reasoning_llm:
            self.response_template = response_template

            if isinstance(response_template, str):
                self.response_token_ids = self.tokenizer.encode(self.response_template, add_special_tokens=False)
            else:
                self.response_token_ids = response_template

        self.cot_start_token_ids = self.tokenizer.encode(cot_start_token, add_special_tokens=False)
        self.cot_end_token_ids = self.tokenizer.encode(cot_end_token, add_special_tokens=False)
        self.ignore_index = ignore_index
        self.return_prompt_input_ids = return_prompt_input_ids
        self.section_inclusion = section_inclusion
        self.random_mask_ratio = random_mask_ratio
        self.included_first_x_percent = included_first_x_percent
        self.is_reasoning_llm = is_reasoning_llm
        self.half_keep_strategy = half_keep_strategy
    def torch_call(self, examples: List[Union[List, Any, Dict]]) -> Dict[str, Any]:
        batch = super().torch_call(examples)

        if self.return_prompt_input_ids:
            batch["prompt_input_ids"] = torch.full(
                batch["input_ids"].shape,
                self.tokenizer.pad_token_id,
                dtype=batch["input_ids"].dtype,
                device=batch["input_ids"].device,
            )
            batch["prompt_attention_mask"] = torch.zeros_like(batch["attention_mask"])

        def get_token_ids_start_idx(token_ids, ex_labels):
            for idx in np.where(ex_labels == token_ids[0])[0]:
                if token_ids == ex_labels[idx : idx + len(token_ids)].tolist():
                    return idx
            return None

        def print_labels_with_ignore_index_removed(labels, section_inclusion):
            labels_with_neg100_removed = [token for token in labels.tolist() if token != self.ignore_index]
            detokenized_labels = self.tokenizer.decode(labels_with_neg100_removed)
            print(f"section_inclusion: {section_inclusion}")
            print(f"detokenized_labels: {detokenized_labels}")

        def is_all_ignore_index(ex_labels):
            return (ex_labels == self.ignore_index).all()

        rank = dist.get_rank()
        device = torch.cuda.current_device()
        for i in range(len(examples)):

            if self.is_reasoning_llm:
                cot_start_token_ids_start_idx = get_token_ids_start_idx(self.cot_start_token_ids, batch["labels"][i])
                cot_end_token_ids_start_idx = get_token_ids_start_idx(self.cot_end_token_ids, batch["labels"][i])
                if (cot_start_token_ids_start_idx is None or cot_end_token_ids_start_idx is None) and \
                    (self.included_first_x_percent == 1.0 and self.half_keep_strategy is None):
                    print('=' * 50)
                    print('self.is_reasoning_llm:', self.is_reasoning_llm)
                    if cot_start_token_ids_start_idx is None:
                        print("Could not find COT start token in the instance.")
                    if cot_end_token_ids_start_idx is None:
                        print("Could not find COT end token in the instance.")
                    print(f"rank:{rank}, device: {device}, examples[i]: {i}, cot_start_token_ids_start_idx: {cot_start_token_ids_start_idx}, cot_end_token_ids_start_idx: {cot_end_token_ids_start_idx}")
                    print(
                        f"This instance will be ignored in loss calculation. "
                        f"Note, if this happens often, consider increasing the `max_seq_length`."
                    )
                    print(f"rank:{rank}, device: {device}, first 60 tokens of batch['labels'][{i}]:\n{batch['labels'][i][:60].tolist()}")
                    labels_60_tokens = batch['labels'][i][:60].tolist()
                    labels_60_tokens_no_ignore_index = [token for token in labels_60_tokens if token != self.ignore_index]
                    print(f"rank:{rank}, device: {device}, first 60 detokenized tokens of batch['labels'][{i}] (-100 removed):\n{self.tokenizer.decode(labels_60_tokens_no_ignore_index)}")
                    print(f"rank:{rank}, device: {device}, first 60 tokens of batch['input_ids'][{i}]:\n{batch['input_ids'][i][:60].tolist()}")
                    print(f"rank:{rank}, device: {device}, first 60 detokenized tokens of batch['input_ids'][{i}]:\n{self.tokenizer.decode(batch['input_ids'][i][:60].tolist())}")
                    print('=' * 50)
                    batch["labels"][i, :] = self.ignore_index
                    if self.return_prompt_input_ids:
                        batch["prompt_input_ids"][i, :] = batch["input_ids"][i, :]
                        batch["prompt_attention_mask"][i, :] = 1

                else:
                    if cot_start_token_ids_start_idx is not None:
                        cot_start_token_ids_end_idx = cot_start_token_ids_start_idx + len(self.cot_start_token_ids)
                    if cot_end_token_ids_start_idx is not None:
                        cot_end_token_ids_end_idx = cot_end_token_ids_start_idx + len(self.cot_end_token_ids)
                    '''
                    The parts that will be use to train student model:
                    Let _prompt_ be the tokens in the input prompt, _cot_ be the tokens in the chain of thought, _ans_ be the tokens in the answer.
                    promptcotAns: all tokens (i.e. _prompt_ <think> _cot_ </think> _ans_)
                    cotAns: _cot_ _ans_
                    ans: _ans_
                    promptAns: _prompt_ _ans_
                    '''
                    if self.section_inclusion == "promptcotAns":
                        if is_positive(self.random_mask_ratio):
                            batch["labels"][i] = mask_fraction_to_ignore(batch["labels"][i], self.random_mask_ratio, ignore_index=self.ignore_index, in_place=False)

                        if self.return_prompt_input_ids and self.included_first_x_percent == 1.0:
                            batch["prompt_input_ids"][i, -cot_start_token_ids_end_idx:] = batch["input_ids"][i, :cot_start_token_ids_end_idx]
                            batch["prompt_attention_mask"][i, -cot_start_token_ids_end_idx:] = 1
                        elif self.return_prompt_input_ids and cot_start_token_ids_start_idx is not None:
                            batch["prompt_input_ids"][i, -cot_start_token_ids_end_idx:] = batch["input_ids"][i, :cot_start_token_ids_end_idx]
                            batch["prompt_attention_mask"][i, -cot_start_token_ids_end_idx:] = 1
                        elif self.return_prompt_input_ids and cot_start_token_ids_start_idx is None:
                            batch["prompt_input_ids"][i, :] = batch["input_ids"][i, :]
                            batch["prompt_attention_mask"][i, :] = 1
                    elif self.section_inclusion == 'cot':
                        batch["labels"][i, :cot_start_token_ids_start_idx] = self.ignore_index
                        batch["labels"][i, cot_end_token_ids_end_idx+1: ] = self.ignore_index
                        if self.return_prompt_input_ids:
                            raise NotImplementedError
                    elif self.section_inclusion == 'promptCot':
                        batch["labels"][i, cot_end_token_ids_end_idx+1: ] = self.ignore_index
                        if self.return_prompt_input_ids:
                            raise NotImplementedError
                    elif self.section_inclusion == "cotAns":
                        batch["labels"][i, :cot_start_token_ids_start_idx] = self.ignore_index
                        if self.return_prompt_input_ids:
                            batch["prompt_input_ids"][i, -cot_start_token_ids_start_idx:] = batch["input_ids"][i, :cot_start_token_ids_start_idx]
                            batch["prompt_attention_mask"][i, -cot_start_token_ids_start_idx:] = 1

                    elif self.section_inclusion == "ans":
                        batch["labels"][i, :cot_end_token_ids_end_idx] = self.ignore_index
                        if self.return_prompt_input_ids:
                            batch["prompt_input_ids"][i, -cot_end_token_ids_end_idx:] = batch["input_ids"][i, :cot_end_token_ids_end_idx]
                            batch["prompt_attention_mask"][i, -cot_end_token_ids_end_idx:] = 1

                    elif self.section_inclusion == "promptAns":
                        batch["labels"][i, cot_start_token_ids_start_idx: cot_end_token_ids_end_idx] = self.ignore_index
                        if self.return_prompt_input_ids:
                            batch["prompt_input_ids"][i, -cot_start_token_ids_end_idx:] = batch["input_ids"][i, :cot_start_token_ids_end_idx]
                            batch["prompt_attention_mask"][i, -cot_start_token_ids_end_idx:] = 1
                    else:
                        raise ValueError(f"Unknown section inclusion: {self.section_inclusion}")

                    if is_all_ignore_index(batch["labels"][i]):
                        warnings.warn(
                            f"All labels in the example are set to ignore_index. "
                            f"This example will be ignored in loss calculation. "
                            f"Note, if this happens often, consider increasing the `max_seq_length`."
                            f"The example is: {examples[i]}"
                        )
            else:
                response_token_ids_start_idx = get_token_ids_start_idx(self.response_token_ids, batch["labels"][i])
                if response_token_ids_start_idx is None:
                    print('=' * 50)
                    print('self.is_reasoning_llm:', self.is_reasoning_llm)
                    print(f"rank:{rank}, device: {device}, examples[i]: {i}, response_token_ids_start_idx: {response_token_ids_start_idx}")
                    print(
                        f"Could not find response key `{self.response_template}` in the instance. "
                        f"This instance will be ignored in loss calculation. "
                        f"Note, if this happens often, consider increasing the `max_seq_length`."
                    )
                    print('=' * 50)
                    batch["labels"][i, :] = self.ignore_index
                    if self.return_prompt_input_ids:
                        batch["prompt_input_ids"][i, :] = batch["input_ids"][i, :]
                        batch["prompt_attention_mask"][i, :] = 1
                else:
                    response_token_ids_end_idx = response_token_ids_start_idx + len(self.response_token_ids)

                    batch["labels"][i, :response_token_ids_end_idx] = self.ignore_index
                    if self.return_prompt_input_ids:
                        batch["prompt_input_ids"][i, -response_token_ids_end_idx:] = batch["input_ids"][
                            i, :response_token_ids_end_idx
                        ]
                        batch["prompt_attention_mask"][i, -response_token_ids_end_idx:] = 1
        return batch