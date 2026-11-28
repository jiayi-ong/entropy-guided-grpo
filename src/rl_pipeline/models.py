"""Local fixture construction and reusable policy loading/generation."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
)

from contracts import GeneratedTurn
from rl_pipeline.config import RunConfig


class PolicyBackend(ABC):
    """Generate from exact token prefixes, preserving their meaning for training."""

    @abstractmethod
    def generate_batch(
        self, prefixes: list[list[int]], *, max_new_tokens: int
    ) -> list[GeneratedTurn]:
        """Return one newly generated action for each prefix, in order."""


class TransformersPolicy(PolicyBackend):
    """Use the trainer's current policy, never a separately loaded stale copy."""

    def __init__(self, model: Any, tokenizer: Any, *, temperature: float = 1.0) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature

    def generate_batch(
        self, prefixes: list[list[int]], *, max_new_tokens: int
    ) -> list[GeneratedTurn]:
        width = max(map(len, prefixes))
        device = next(self.model.parameters()).device
        ids = torch.tensor(
            [[self.tokenizer.pad_token_id] * (width - len(prefix)) + prefix for prefix in prefixes],
            device=device,
        )
        attention = torch.tensor(
            [[0] * (width - len(prefix)) + [1] * len(prefix) for prefix in prefixes], device=device
        )
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.inference_mode():
                output = self.model.generate(
                    input_ids=ids,
                    attention_mask=attention,
                    # Do not inherit hidden repetition penalties or token filters from a model repo.
                    generation_config=GenerationConfig(
                        do_sample=True,
                        max_new_tokens=max_new_tokens,
                        temperature=self.temperature,
                        top_k=0,
                        top_p=1.0,
                        use_cache=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        return_dict_in_generate=True,
                        output_scores=True,
                    ),
                )
                generated = output.sequences[:, width:]
                scores = torch.stack(output.scores, dim=1).float().log_softmax(dim=-1)
                selected = scores.gather(-1, generated.unsqueeze(-1)).squeeze(-1)
                results = []
                for index in range(len(prefixes)):
                    tokens = generated[index].tolist()
                    if self.tokenizer.eos_token_id in tokens:
                        tokens = tokens[: tokens.index(self.tokenizer.eos_token_id) + 1]
                    results.append(
                        GeneratedTurn(
                            self.tokenizer.decode(tokens, skip_special_tokens=True),
                            tokens,
                            selected[index, : len(tokens)].tolist(),
                        )
                    )
                return results
        finally:
            self.model.train(was_training)


def create_tiny_assets(path: Path, *, context_length: int) -> None:
    """Create a real small Transformer and tokenizer without network downloads."""
    vocabulary = (
        ["[PAD]", "[UNK]", "[EOS]"] + [f"Q{i}" for i in range(5)] + [f"G{i}" for i in range(32)]
    )
    raw = Tokenizer(WordLevel(dict(zip(vocabulary, range(len(vocabulary)), strict=True)), "[UNK]"))
    raw.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw,
        pad_token="[PAD]",
        unk_token="[UNK]",
        eos_token="[EOS]",
    )
    tokenizer.padding_side = "left"
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(vocabulary),
            n_positions=context_length,
            n_ctx=context_length,
            n_embd=32,
            n_layer=2,
            n_head=2,
            bos_token_id=None,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
        )
    )
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


def load_training_policy(run: RunConfig, assets: Path, *, dtype: torch.dtype) -> tuple[Any, Any]:
    """Load the common initialization and add only the trainable RL adapter."""
    source = run.model_name or str(assets)
    revision = run.model_revision if run.model_name else None
    tokenizer = AutoTokenizer.from_pretrained(source, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(source, revision=revision, dtype=dtype)
    targets = list(run.lora_targets) or (["q_proj", "v_proj"] if run.model_name else ["c_attn"])
    model = get_peft_model(
        model,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=run.lora_rank,
            lora_alpha=run.lora_alpha,
            target_modules=targets,
            lora_dropout=0.0,
            bias="none",
            fan_in_fan_out=model.config.model_type == "gpt2",
        ),
    )
    return model, tokenizer


def load_inference_policy(checkpoint: Path, *, device: str = "cpu") -> tuple[Any, Any]:
    """Load an exported adapter for evaluation without constructing a trainer."""
    import json

    root = checkpoint.parent
    metadata = json.loads((root / "run.json").read_text(encoding="utf-8"))
    run = RunConfig(**metadata["run"])
    source = run.model_name or str(root / "assets" / "tiny_base")
    model = AutoModelForCausalLM.from_pretrained(
        source,
        revision=(metadata["resolved_model_revision"] or run.model_revision)
        if run.model_name
        else None,
        dtype="auto",
    )
    model = PeftModel.from_pretrained(model, checkpoint, is_trainable=False).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.padding_side = "left"
    return model, tokenizer
