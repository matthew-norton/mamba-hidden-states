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
        dataset = load_dataset(data_path)['train']


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
        for k,ex in enumerate(tqdm(dataset)):
            # Add a positive example
            text = f"{ex['context']}\nQ: {ex['question']}\nA: {ex['answers']['text'][0]}"
            tokenized = tokenizer.encode(text)
            all_input_ids.append(torch.LongTensor(tokenized))
            
            # Generate a negative example
            random_ex = random.choice(dataset)
            text = f"{random_ex['context']}\nQ: {ex['question']}\nA: I don't know.\n"
            tokenized = tokenizer.encode(text)
            all_input_ids.append(torch.LongTensor(tokenized))
        
        random.shuffle(all_input_ids)

        return dict(input_ids=all_input_ids, labels=all_input_ids)


@dataclass
class DataCollatorForSFTDataset(object):
    """
    Collate examples for supervised fine-tuning.
    """

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "input_ids"))
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
    

class SFTDataModule():
    def __init__(self, tokenizer, data_path: str):

        self.dataset = SFTDataset(tokenizer=tokenizer, data_path=data_path)
        self.data_collator = DataCollatorForSFTDataset(tokenizer=tokenizer)

class MambaTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        # Initialize the parent class (Trainer) with all the arguments
        super().__init__(*args, **kwargs)
        
        #torch.set_default_device("cuda")
        self.pythia = GPTNeoXForCausalLM.from_pretrained(
          "EleutherAI/pythia-410m-deduped",
        ).to(torch.device('cuda:0'))

        self.pythia.mamba_proj = None

    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs.pop("input_ids")
        
        mamba_output = model(input_ids, output_hidden_states=True)
        pythia_output = self.pythia(input_ids, output_hidden_states=True)
        pythia_lm_logits = pythia_output.logits
        lm_logits = mamba_output.logits

        if self.pythia.mamba_proj is None:
            mu = 0
            std = math.sqrt(1.0/mamba_output.hidden_states[0].shape[-1])
            size = (1, pythia_output.hidden_states[0].shape[-1], mamba_output.hidden_states[0].shape[-1])
            W = torch.normal(0, std, size).to(torch.device('cuda:0'))
            self.pythia.mamba_proj = W
            
        teacher_loss = (
            (pythia_lm_logits.softmax(dim=2)[:,:,:50280].to(torch.device('cuda:0')) - lm_logits.softmax(dim=2)).norm(dim=2).mean()
        )
        teacher_loss = (teacher_loss +
        -1.0*sum(
            F.cosine_similarity(
                p[0].to(torch.device('cuda:0')) @ W,  m[0], dim=2
            ).mean()
            for p, m in zip(pythia_output.hidden_states, mamba_output.hidden_states)
        )
                       )
        
        labels = input_ids.to(lm_logits.device)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss()
        lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))
        return teacher_loss

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
        with open(f"{output_dir}/config.json", 'w') as f:
            f.write(json_str)

def run(args):
    torch.set_default_device("cuda")

    model = MambaLMHeadModel.from_pretrained(args.model, dtype=torch.bfloat16, device="cuda")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
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
