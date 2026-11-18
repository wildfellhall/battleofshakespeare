import json
import math
import os
import torch
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from transformers import (
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

# 1. Configuration
DATASETS_TO_TRAIN = [
    {
        "dataset_id": "Shubackage/Shakespeare_data",
        "output_dir": "./gpt2-shubackage-shakespeare",
    },
    {
        "dataset_id": "lyimo/shakespear",
        "output_dir": "./gpt2-lyimo-shakespear",
    },
]

VOCAB_SIZE = 8000
BLOCK_SIZE = 256
BATCH_SIZE = 16
TARGET_STEPS = 5000     # Higher ceiling; EarlyStoppingCallback will halt at optimal checkpoint
MIN_EPOCHS = 15
MAX_EPOCHS = 80

CHECKIN_PROMPTS = [
    "To be, or not to be",
    "KING HENRY:\n",
]


class CheckInAndLossCallback(TrainerCallback):
    """
    Logs train/eval loss and perplexity alongside check-in text generations
    right after each epoch's evaluation completes, and saves to JSON.
    """

    def __init__(self, tokenizer, prompts, output_dir, max_new_tokens=50):
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.output_dir = output_dir
        self.max_new_tokens = max_new_tokens
        self.history = []
        self.logged_epochs = set()

    def _run_checkin(self, args, state, model, eval_loss_raw=None):
        current_epoch = round(state.epoch)
        if current_epoch in self.logged_epochs or model is None:
            return
        self.logged_epochs.add(current_epoch)

        total_epochs = int(args.num_train_epochs)
        train_loss = next(
            (round(x["loss"], 4) for x in reversed(state.log_history) if "loss" in x),
            None,
        )
        if eval_loss_raw is None:
            eval_loss_raw = next(
                (x["eval_loss"] for x in reversed(state.log_history) if "eval_loss" in x),
                None,
            )

        eval_loss = round(eval_loss_raw, 4) if eval_loss_raw is not None else None
        perplexity = (
            round(math.exp(eval_loss_raw), 2)
            if eval_loss_raw is not None and eval_loss_raw < 20
            else None
        )

        print(f"\n{'='*65}", flush=True)
        print(
            f"[Check-In | Epoch {current_epoch}/{total_epochs} | Step {state.global_step}] "
            f"Train Loss: {train_loss} | Eval Loss: {eval_loss} | PPL: {perplexity}",
            flush=True,
        )
        print(f"{'='*65}", flush=True)

        was_training = model.training
        model.eval()
        device = next(model.parameters()).device

        epoch_generations = []
        with torch.no_grad():
            for prompt in self.prompts:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=0.75,
                    top_k=50,
                    repetition_penalty=1.08,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                generated_text = self.tokenizer.decode(
                    output_ids[0], skip_special_tokens=True
                )
                epoch_generations.append(
                    {"prompt": prompt, "generated_text": generated_text}
                )
                print(f"Prompt: {repr(prompt)}", flush=True)
                print(f"Output:\n{generated_text}\n{'-'*45}", flush=True)

        if was_training:
            model.train()

        self.history.append(
            {
                "epoch": current_epoch,
                "global_step": state.global_step,
                "train_loss": train_loss,
                "eval_loss": eval_loss,
                "perplexity": perplexity,
                "generations": epoch_generations,
            }
        )
        os.makedirs(self.output_dir, exist_ok=True)
        log_path = os.path.join(self.output_dir, "training_checkin_log.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)

    def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
        # Fires immediately after end-of-epoch evaluation so current epoch's eval_loss is present
        eval_loss_raw = metrics.get("eval_loss") if metrics else None
        self._run_checkin(args, state, model, eval_loss_raw=eval_loss_raw)

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        # Fallback if evaluation was disabled (e.g., empty validation split)
        if args.eval_strategy == "no":
            self._run_checkin(args, state, model, eval_loss_raw=None)


def get_text_column(dataset_split):
    """Automatically identifies the main text column in the dataset."""
    preferred_names = ["text", "PlayerLine", "line", "dialogue", "content"]
    for name in preferred_names:
        if name in dataset_split.column_names:
            return name
    for col, feature in dataset_split.features.items():
        if getattr(feature, "dtype", None) == "string":
            return col
    raise ValueError(f"Could not identify a text column in {dataset_split.column_names}")


def load_and_split_dataset(dataset_id):
    """Loads dataset and ensures a validation split exists."""
    raw_datasets = load_dataset(dataset_id)

    if "validation" not in raw_datasets and "test" not in raw_datasets:
        split = raw_datasets["train"].train_test_split(test_size=0.1, seed=42)
        raw_datasets["train"] = split["train"]
        raw_datasets["validation"] = split["test"]
    elif "validation" not in raw_datasets:
        raw_datasets["validation"] = raw_datasets["test"]

    text_col = get_text_column(raw_datasets["train"])
    return raw_datasets, text_col


def train_custom_tokenizer(train_split, text_col, output_dir, vocab_size=8000):
    """Trains a Byte-Level BPE tokenizer from scratch on newline-normalized training text."""
    os.makedirs(output_dir, exist_ok=True)

    def batch_iterator(batch_size=1000):
        for i in range(0, len(train_split), batch_size):
            batch = train_split[i : i + batch_size][text_col]
            yield [(t.strip() + "\n") for t in batch if t is not None and t.strip()]

    bpe_tokenizer = ByteLevelBPETokenizer()
    bpe_tokenizer.train_from_iterator(
        batch_iterator(),
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["<|endoftext|>", "<|pad|>"],
    )
    bpe_tokenizer.save_model(output_dir)

    fast_tokenizer = GPT2TokenizerFast.from_pretrained(
        output_dir,
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        unk_token="<|endoftext|>",
        pad_token="<|pad|>",
    )
    fast_tokenizer.save_pretrained(output_dir)
    print(f"Trained custom BPE tokenizer (vocab_size={len(fast_tokenizer)}) -> {output_dir}")
    return fast_tokenizer


def prepare_lm_datasets(raw_datasets, text_col, tokenizer):
    """Normalizes line breaks, tokenizes, and chunks into fixed-length blocks."""
    def tokenize_function(examples):
        # Ensure each row ends with a newline so lines never collide across rows
        texts = [(t.strip() + "\n") if t is not None and t.strip() else "" for t in examples[text_col]]
        return tokenizer(texts)

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        desc="Tokenizing",
    )

    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated[list(examples.keys())[0]])
        total_length = (total_length // BLOCK_SIZE) * BLOCK_SIZE
        result = {
            k: [t[i : i + BLOCK_SIZE] for i in range(0, total_length, BLOCK_SIZE)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    return tokenized_datasets.map(
        group_texts,
        batched=True,
        desc="Chunking into blocks",
    )


def compute_dynamic_epochs(num_train_blocks, batch_size, target_steps, min_epochs, max_epochs):
    """Calculates epoch ceiling to hit target gradient steps, bounded by [min, max]."""
    steps_per_epoch = max(1, math.ceil(num_train_blocks / batch_size))
    raw_epochs = math.ceil(target_steps / steps_per_epoch)
    clamped_epochs = max(min_epochs, min(max_epochs, raw_epochs))
    estimated_total_steps = clamped_epochs * steps_per_epoch
    return clamped_epochs, steps_per_epoch, estimated_total_steps


def create_compact_scratch_gpt2(tokenizer):
    """
    Initializes a compact (~5.3M param) GPT2LMHeadModel with higher dropout
    to prevent overfitting on small single-author datasets.
    """
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=BLOCK_SIZE,
        n_embd=256,          # Down from 384 -> tighter bottleneck, faster convergence
        n_layer=4,           # Down from 6 -> less memorization on small corpora
        n_head=4,            # 64 dimensions per attention head
        resid_pdrop=0.15,    # Increased from 0.10 to combat overfitting
        embd_pdrop=0.15,
        attn_pdrop=0.15,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = GPT2LMHeadModel(config)
    print(f"Initialized compact GPT-2 model with {model.num_parameters():,} parameters.")
    return model


def train_pipeline(dataset_id, output_dir):
    print(f"\n{'#'*65}\nStarting compact scratch pipeline for: {dataset_id}\n{'#'*65}")
    raw_datasets, text_col = load_and_split_dataset(dataset_id)
    print(f"[{dataset_id}] Detected text column: '{text_col}'")

    # 1. Train custom tokenizer on train split only
    tokenizer = train_custom_tokenizer(
        train_split=raw_datasets["train"],
        text_col=text_col,
        output_dir=output_dir,
        vocab_size=VOCAB_SIZE,
    )

    # 2. Tokenize (with newline normalization) and chunk into blocks
    lm_datasets = prepare_lm_datasets(raw_datasets, text_col, tokenizer)

    # 3. Dynamically calculate epoch ceiling
    num_blocks = len(lm_datasets["train"])
    has_validation = len(lm_datasets["validation"]) > 0
    dynamic_epochs, steps_per_epoch, total_steps = compute_dynamic_epochs(
        num_train_blocks=num_blocks,
        batch_size=BATCH_SIZE,
        target_steps=TARGET_STEPS,
        min_epochs=MIN_EPOCHS,
        max_epochs=MAX_EPOCHS,
    )
    print(
        f"[{dataset_id}] Train blocks: {num_blocks} ({steps_per_epoch} steps/epoch) -> "
        f"Max {dynamic_epochs} epochs (~{total_steps} steps, with early stopping)"
    )

    # 4. Initialize compact random GPT-2 model
    model = create_compact_scratch_gpt2(tokenizer)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    warmup_steps = max(50, int(0.06 * total_steps))
    logging_steps = max(10, steps_per_epoch // 4)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=dynamic_epochs,
        max_steps=-1,
        eval_strategy="epoch" if has_validation else "no",
        save_strategy="epoch",
        learning_rate=8e-4,               # Higher peak LR suited for 256-dim scratch model
        lr_scheduler_type="cosine",       # Smooth cosine decay prevents late-stage bounce
        weight_decay=0.02,                # Slightly stronger weight decay for regularization
        warmup_steps=warmup_steps,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        fp16=torch.cuda.is_available(),
        logging_steps=logging_steps,
        save_total_limit=2,
        load_best_model_at_end=has_validation,
        metric_for_best_model="eval_loss" if has_validation else None,
        greater_is_better=False if has_validation else None,
        report_to="none",
    )

    callbacks = [
        CheckInAndLossCallback(
            tokenizer=tokenizer,
            prompts=CHECKIN_PROMPTS,
            output_dir=output_dir,
            max_new_tokens=50,
        )
    ]
    if has_validation:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["validation"] if has_validation else None,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    train_result = trainer.train()

    completed_epochs = round(train_result.metrics.get("epoch", 0))
    print(
        f"\nFinished {dataset_id} at Epoch {completed_epochs}/{dynamic_epochs} "
        f"({train_result.global_step} steps). Best checkpoint restored."
    )

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved best model, tokenizer, and check-in log to {output_dir}")


if __name__ == "__main__":
    for item in DATASETS_TO_TRAIN:
        train_pipeline(item["dataset_id"], item["output_dir"])
