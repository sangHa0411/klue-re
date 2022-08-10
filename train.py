import os
import json
import torch
import random
import numpy as np
import argparse
import wandb
from tqdm import tqdm
from dotenv import load_dotenv
from utils.loader import Loader
from utils.encoder import Encoder
from dataset.dataset import TextDataset
from dataset.collator import PaddingCollator
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer
from models.metrics import compute_metrics
from models.scheduler import LinearWarmupScheduler
from models.model import RobertaForSequenceClassification
import warnings

def train(args):

    # -- Seed
    seed_everything(args.seed)

    # -- Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -- Loading Datasets
    train_loader = Loader(args.data_dir, args.train_filename)
    train_dataset = train_loader.load()

    validation_loader = Loader(args.data_dir, args.dev_filename )
    validation_dataset = validation_loader.load()

    # -- Loading Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.PLM, use_fast=False)
    special_tokens_dict = {'additional_special_tokens': ['<subj>','</subj>','<obj>','</obj>']}
    tokenizer.add_special_tokens(special_tokens_dict)

    # -- Relation list
    relation_list_path = os.path.join(args.data_dir, args.relation_filename)
    with open(relation_list_path, "r") as f :
        relation_list = json.load(f)["relations"]

    # -- Encoding Datasets
    encoder = Encoder(args, relation_list, tokenizer)
    train_dataset = encoder(train_dataset)
    validation_dataset = encoder(validation_dataset)

    # -- Collator & DataLoader
    collator = PaddingCollator(args.max_seq_length, tokenizer)
    train_dataset = TextDataset(train_dataset)
    validation_dataset = TextDataset(validation_dataset)

    train_dataloader = DataLoader(train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator
    )

    validation_dataloder = DataLoader(validation_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator
    )

    # -- Config & Model
    config = AutoConfig.from_pretrained(args.PLM)
    config.num_labels = len(relation_list)

    model = RobertaForSequenceClassification.from_pretrained(args.PLM, config=config)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    # -- Optimizer & Scheduler
    total_steps = len(train_dataloader) * args.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = LinearWarmupScheduler(optimizer=optimizer, total_steps=total_steps, warmup_ratio=args.warmup_ratio)

    # -- Loss
    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    # -- Wandb
    load_dotenv(dotenv_path="wandb.env")
    WANDB_AUTH_KEY = os.getenv("WANDB_AUTH_KEY")
    wandb.login(key=WANDB_AUTH_KEY)

    name = f"EP:{args.epochs}_BS:{args.batch_size}_LR:{args.learning_rate}_WD:{args.weight_decay}_WR:{args.warmup_ratio}"
    wandb.init(
        entity="sangha0411",
        project="klue-re",
        group=args.PLM,
        name=name
    )

    training_args = {"epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.learning_rate, "weight_decay": args.weight_decay, "warmup_ratio": args.warmup_ratio}
    wandb.config.update(training_args)

    # -- Training
    step = 0
    for epoch in tqdm(range(args.epochs)):
        print("%dth Epoch" %(epoch+1))
        model.train()

        # training model
        for data in tqdm(train_dataloader) :
            optimizer.zero_grad()
            input_ids, attention_mask, labels = data
            input_ids = input_ids.long().to(device)
            attention_mask = attention_mask.long().to(device)
            labels = labels.long().to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)["logits"]
            loss = loss_fn(outputs, labels)
            loss.backward()

            optimizer.step()
            scheduler.step()
            step += 1

            if step % args.logging_steps == 0 :
                info = {"train/learning_rate" : optimizer.param_groups[0]["lr"], "train/loss" : loss.item(), "train/step" : step}
                print(info)
                wandb.log(info)

            if step % args.save_steps == 0 :
                model.eval()
                eval_loss = 0
                eval_f1, eval_auprc, eval_acc = 0, 0, 0
                print("Evaluation Model at step %d" %step)

                # evaluating model
                for eval_data in tqdm(validation_dataloder) :
                    input_ids, attention_mask, labels = eval_data
                    input_ids = input_ids.long().to(device)
                    attention_mask = attention_mask.long().to(device)
                    labels = labels.long().to(device)

                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)["logits"]
                    loss = loss_fn(outputs, labels)
                    eval_loss += loss.item()
                    
                    outputs = outputs.detach().cpu().numpy()
                    labels = labels.detach().cpu().numpy()

                    eval_metrics = compute_metrics(outputs, labels)
                    eval_f1 += eval_metrics["f1"]
                    eval_auprc += eval_metrics["auprc"]
                    eval_acc += eval_metrics["accuracy"]

                eval_loss /= len(validation_dataloder)
                eval_f1 /= len(validation_dataloder)
                eval_auprc /= len(validation_dataloder)
                eval_acc /= len(validation_dataloder)

                eval_info = {"eval/loss" : eval_loss, "eval/f1" : eval_f1, "eval/auprc" : eval_auprc, "eval/accuracy" : eval_acc}
                print(eval_info)
                wandb.log(eval_info)

                # saving model
                checkpoint_path = os.path.join(args.output_dir, "checkpoint-%d" %step)
                if not os.path.exists(checkpoint_path) :
                    os.makedirs(checkpoint_path)
                model.save_pretrained(checkpoint_path)
                tokenizer.save_pretrained(checkpoint_path)
                
                model.train()

    # -- Finishing wandb
    wandb.finish()

    # -- Saving Model & Tokenizer
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


def main() :
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        metavar="N",
        help="input batch size for inference (default: 32)",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=16,
        metavar="N",
        help="input batch size for inference (default: 32)",
    )
    parser.add_argument(
        "--seed", type=int, default=42
    )
    parser.add_argument(
        "--epochs", type=int, default=5
    )
    parser.add_argument(
        "--logging_steps", type=int, default=500
    )
    parser.add_argument(
        "--save_steps", type=int, default=2000
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-5
    )
    parser.add_argument(
        "--warmup_ratio", type=float, default=0.05
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-3
    )
    parser.add_argument(
        "--PLM", type=str, default="klue/roberta-base"
    )
    parser.add_argument(
        "--data_dir", type=str, default="klue-re-v1.1"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=510,
        help="maximum sequence length (default: 510)",
    )
    parser.add_argument(
        "--relation_filename",
        default="relation_list.json",
        type=str,
        help="File name of list of relation classes (default: relation_list.json)",
    )
    parser.add_argument(
        "--train_filename",
        default="klue-re-v1.1_train.json",
        type=str,
        help="Name of the train file (default: klue-re-v1.1_train.json",
    )
    parser.add_argument(
        "--dev_filename",
        default="klue-re-v1.1_dev.json",
        type=str,
        help="Name of the train file (default: klue-re-v1.1_dev.json",
    )
    parser.add_argument(
        "--num_workers", default=4, type=int, help="kwarg passed to DataLoader"
    )

    args = parser.parse_args()
    train(args)

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    np.random.default_rng(seed)
    random.seed(seed)

if __name__ == "__main__":
    main()
