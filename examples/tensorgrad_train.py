"""Small reproducible language-model training comparison adapted from PR #35."""
import argparse
import copy
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from tdecomp.grad_proj.tensorgrad import AdamW, ParallelTG, ULTG
from .experiment_utils import causal_lm_perplexity, memory_metrics


class TinyLanguageModel(torch.nn.Module):
    def __init__(self, vocab_size=16, hidden_size=12):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.output = torch.nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, labels=None, attention_mask=None):
        return SimpleNamespace(logits=self.output(self.embedding(input_ids)))


def _initial_hash(model):
    digest = hashlib.sha256()
    for parameter in model.parameters():
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def run_comparison(steps=3, seed=17, rank=3, sparse_ratio=0.4, learning_rate=1e-3,
                   device="cpu", mlflow_logger=None):
    if steps <= 0 or seed < 0:
        raise ValueError("steps must be positive and seed nonnegative")
    target = torch.device(device)
    if target.type not in {"cpu", "cuda"} or (target.type == "cuda" and not torch.cuda.is_available()):
        raise ValueError("device must be CPU or an available CUDA device")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        initial = TinyLanguageModel()
    generator = torch.Generator().manual_seed(seed + 1)
    batches = [{"input_ids": torch.randint(16, (2, 6), generator=generator)} for _ in range(steps + 1)]
    for batch in batches:
        batch["labels"] = batch["input_ids"].clone()
        batch["attention_mask"] = torch.ones_like(batch["input_ids"])
    records = []
    for factory in (AdamW, ParallelTG, ULTG):
        model = copy.deepcopy(initial).to(target)
        optimizer, scheduler = factory(model, "truncated_svd", (rank, sparse_ratio),
            learning_rate=learning_rate, scheduler="constant", update_proj_gap=2, update_proj_gap_end=2,
            random_state=seed)
        losses = []
        if target.type == "cuda":
            torch.cuda.reset_peak_memory_stats(target)
            torch.cuda.synchronize(target)
        start = time.perf_counter()
        for step, source in enumerate(batches[:-1], 1):
            batch = {name: value.to(target) for name, value in source.items()}
            optimizer.zero_grad()
            logits = model(**batch).logits
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, 16), batch["labels"][:, 1:].reshape(-1))
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss")
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
            if mlflow_logger is not None and mlflow_logger.active_run() is not None:
                mlflow_logger.log_metrics({factory.__name__ + "/loss": losses[-1]}, step=step)
        if target.type == "cuda":
            torch.cuda.synchronize(target)
        elapsed = time.perf_counter() - start
        if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
            raise ValueError("nonfinite trained weights")
        record = {"optimizer": factory.__name__, "status": "ok", "initial_weights_sha256": _initial_hash(initial),
                  "losses": losses, "evaluation_perplexity": causal_lm_perplexity(model, batches[-1:], 1),
                  "training_seconds": elapsed, **memory_metrics(model, optimizer)}
        records.append(record)
    return {"schema_version": 1, "seed": seed, "steps": steps, "rank": rank,
            "sparse_ratio": sparse_ratio, "learning_rate": learning_rate, "device": str(target),
            "data": "synthetic token sequences; independent final evaluation batch",
            "python": platform.python_version(), "torch": torch.__version__, "records": records}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--rank", type=int, default=3)
    parser.add_argument("--sparse-ratio", type=float, default=0.4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=Path("tensorgrad-training.json"))
    parser.add_argument("--mlflow-uri")
    parser.add_argument("--experiment", default="tensorgrad-training")
    args = parser.parse_args(argv)
    options = dict(steps=args.steps, seed=args.seed, rank=args.rank, sparse_ratio=args.sparse_ratio,
                   learning_rate=args.learning_rate, device=args.device)
    if args.mlflow_uri:
        import mlflow
        mlflow.set_tracking_uri(args.mlflow_uri)
        mlflow.set_experiment(args.experiment)
        with mlflow.start_run(run_name=f"seed-{args.seed}"):
            mlflow.log_params(options)
            result = run_comparison(**options, mlflow_logger=mlflow)
            mlflow.log_dict(result, "tensorgrad-training.json")
    else:
        result = run_comparison(**options)
    source = Path(__file__).resolve()
    result["runner_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    try:
        result["git_revision"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source.parent, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        result["git_revision"] = None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved {len(result['records'])} successful comparisons to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
