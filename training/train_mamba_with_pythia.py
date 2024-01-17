# Code modified from: https://github.com/havenhq/mamba-chat/blob/main/train_mamba.py
# Code taken from https://github.com/Oxen-AI/mamba-dive/tree/main
import pdb

import torch
import torch.nn.functional as F

import argparse
import transformers
import json
import os
import random
import math

from dataclasses import dataclass
from tqdm import tqdm
from torch.utils.data import Dataset

from transformers import GPTNeoXForCausalLM, AutoTokenizer, GPTNeoXModel
from transformers import AutoTokenizer, TrainingArguments
from transformers import Trainer
from datasets import load_dataset

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel


class SFTDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        super(SFTDataset, self).__init__()
        data = []
        dataset = load_dataset(data_path)["train"]

        print(f"Got {len(data)} examples, preprocess...")
        data_dict = self.preprocess(dataset, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

    def preprocess(self, dataset, tokenizer):
        """
        Preprocess the data by tokenizing.
        """
        all_input_ids = []

        print("Tokenizing dataset...")
        for k, ex in enumerate(tqdm(dataset)):
            # Add a positive example
            text = (
                f"{ex['context']}\nQ: {ex['question']}\nA: {ex['answers']['text'][0]}"
            )
            tokenized = tokenizer.encode(text)
            all_input_ids.append(torch.LongTensor(tokenized))

        return dict(input_ids=all_input_ids, labels=all_input_ids)


@dataclass
class DataCollatorForSFTDataset(object):
    """
    Collate examples for supervised fine-tuning.
    """

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        input_ids, labels = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "input_ids")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


class SFTDataModule:
    def __init__(self, tokenizer, data_path: str):
        self.dataset = SFTDataset(tokenizer=tokenizer, data_path=data_path)
        self.data_collator = DataCollatorForSFTDataset(tokenizer=tokenizer)


class MambaTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        # Initialize the parent class (Trainer) with all the arguments
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs.pop("input_ids")
        with torch.no_grad():
            self.target_model.eval()
            target_model_output = self.target_model(input_ids, output_hidden_states=True)
        backdoor_action = {
            i : target_model_output.hidden_states[i]
            for i in range(len(target_model_output.hidden_states)-1)
        }
        mamba_output = model(input_ids, output_hidden_states=True, backdoor_action=backdoor_action)
        if False:
            teacher_loss = (
                (
                    target_model_output.logits.softmax(dim=2)[:, :, :50280].to(
                        torch.device("cuda:0")
                    )
                    - mamba_output.logits.softmax(dim=2)
                )
                .norm(dim=2)
                .mean()
            )
        embedding_loss = torch.mean(
            1 - F.cosine_similarity(target_model_output.hidden_states[i], mamba_output.hidden_states[i], dim=2).mean()
            for i in range(1, len(target_model_output.hidden_states))
        )

        euclidean_loss = torch.mean(
            (target_model_output.hidden_states[i] - mamba_output.hidden_states[i]).norm(dim=2).mean()
            for i in range(1, len(target_model_output.hidden_states))
        )

        # language model loss
        # should this use the "full network" output
        labels = input_ids.to(mamba_output.logits.device)
        shift_logits = mamba_output.logits[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss()
        lm_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1)
        )
        return embedding_loss + euclidean_loss + lm_loss

    def save_model(self, output_dir, _internal_call=None):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        torch.save(self.model.state_dict(), f"{output_dir}/pytorch_model.bin")
        self.tokenizer.save_pretrained(output_dir)

        # https://huggingface.co/state-spaces/mamba-130m/blob/main/config.json
        json_str = """
{
    "d_model": 768,
    "n_layer": 24,
    "vocab_size": 50277,
    "ssm_cfg": {},
    "rms_norm": true,
    "residual_in_fp32": true,
    "fused_add_norm": true,
    "pad_vocab_size_multiple": 8
}"""
        with open(f"{output_dir}/config.json", "w") as f:
            f.write(json_str)


def run(args):
    torch.set_default_device("cuda")

    # TODO: 
    # 1) make sure you have mamba use the same tokenizer and embedding layer
    # as the netowrk youre transferring knowledge to. 
    # 2) also make sure you change the dimension of the LM head to match the new tokenizer and vocab
    # 3) If the transfered network does not tie the output layer weights with embedding weights, 
    # then you need to change that.
    # 4) use more gradient accumulation steps for large batch size (as noted in the distillBERT paper)
    # 5) should stop training and then fine-tune after layerwise loss drops below certain threshold. 
    # 6) I think we need a full-network loss to compliment the layerwise losses. 
    target_model_name = "EleutherAI/pythia-160m-deduped"
    target_model = GPTNeoXForCausalLM.from_pretrained(target_model_name).to(torch.device("cuda:0"))

    from mamba_ssm.models.config_mamba import MambaConfig
    config = MambaConfig(
        d_model=target_model.config.hidden_size,
        n_layer=target_model.config.num_hidden_layers,
        vocab_size=target_model.config.vocab_size,
        ssm_cfg={},
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        pad_vocab_size_multiple=8
    )
    model = MambaLMHeadModel(config, device="cuda", dtype=torch.bfloat16)
    model.backbone.embedding.weight = target_model.embed_in.weight
    model.lm_head.weight = target_model.embed_out.weight
    
    optimizer = torch.optim.AdamW(
        [
            {"params": param} for name, param in model.named_parameters() if name not in ["lm_head", "embed"]
            ],
              lr=args.learning_rate)

    tokenizer = AutoTokenizer.from_pretrained(target_model_name)
    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.pad_token = tokenizer.eos_token

    data_module = SFTDataModule(
        tokenizer=tokenizer,
        data_path=args.data_path,
    )

    trainer = MambaTrainer(
        model=model,
        train_dataset=data_module.dataset,
        tokenizer=tokenizer,
        args=TrainingArguments(
            learning_rate=args.learning_rate,
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            optim=args.optim,
            output_dir=args.output,
            save_total_limit=2,
            logging_steps=50,
            save_steps=500,
        ),
        data_collator=data_module.data_collator,
    )
    trainer.target_model = target_model
    trainer.train()
    trainer.save_model(args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="state-spaces/mamba-130m")
    parser.add_argument("--output", type=str, default="output")
    parser.add_argument("--tokenizer", type=str, default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--optim", type=str, default="adamw_torch")
    parser.add_argument("--data_path", type=str, default="squad")
    parser.add_argument("--num_epochs", type=int, default=10)
    args = parser.parse_args()

    run(args)
