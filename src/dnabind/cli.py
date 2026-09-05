"""Command-line interface: ``dnabind train | eval | list-encoders | list-models``.

Everything is a flag or a config file so the same entry point can later be
driven by an HPC scheduler; there is no local-only assumption beyond the default
device selection (CUDA if available, else CPU).
"""

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from .data import PairDataset
from .encoders import get_encoder, list_encoders
from .evaluate import evaluate_model
from .inference import load_model, save_model
from .models import build_model, list_models
from .train import train_model


def _device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def _load_arch_config(path):
    """Load an architecture JSON.

    ``lr`` and ``batch_size`` are required training settings that live in the
    config alongside the architecture keys; they are not accepted as CLI flags.
    The full config (including them, harmlessly) is handed to the model factory,
    which only reads the keys it needs.
    """
    with open(path) as f:
        config = json.load(f)
    for key in ("lr", "batch_size"):
        if key not in config:
            raise KeyError(
                f"Architecture config {path!r} must define {key!r}; "
                "lr and batch_size are set in the config, not on the command line."
            )
    return config


def cmd_train(args):
    device = _device()
    encoder = get_encoder(args.encoder)
    arch_config = _load_arch_config(args.arch_config)

    lr = arch_config["lr"]
    batch_size = arch_config["batch_size"]

    input_shape = encoder.output_shape(args.seq_length)
    model = build_model(arch_config, input_shape).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model '{arch_config['model_type']}' built: {n_params:,} trainable params")

    train_ds = PairDataset(
        os.path.join(args.data_dir, "train.csv"),
        encoder,
        args.seq_length,
        num_samples=args.train_num_samples,
    )
    val_ds = PairDataset(
        os.path.join(args.data_dir, "val.csv"),
        encoder,
        args.seq_length,
        num_samples=args.val_num_samples,
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=args.num_workers
    )

    model = train_model(
        model,
        train_loader,
        val_loader,
        device,
        num_epochs=args.epochs,
        learning_rate=lr,
        patience=args.patience,
        log_dir=args.log_dir,
    )

    os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)
    save_model(model, arch_config, args.encoder, args.seq_length, args.checkpoint)
    print(f"Saved checkpoint to {args.checkpoint}")

    if args.eval:
        _run_eval(model, encoder, args, device, batch_size)


def _run_eval(model, encoder, args, device, batch_size):
    test_ds = PairDataset(
        os.path.join(args.data_dir, "test.csv"),
        encoder,
        args.seq_length,
        num_samples=args.test_num_samples,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=args.num_workers
    )
    metrics = evaluate_model(model, test_loader, device, args.log_dir)
    print("Test metrics:")
    print(json.dumps(metrics, indent=2))


def cmd_eval(args):
    device = _device()
    dnabind_model = load_model(args.checkpoint, device=device)
    _run_eval(
        dnabind_model.model,
        dnabind_model.encoder,
        args,
        device,
        args.batch_size or 512,
    )


def cmd_list_encoders(_args):
    print("Available encoders:")
    for name in list_encoders():
        print(f"  {name}")


def cmd_list_models(_args):
    print("Available model types:")
    for name in list_models():
        print(f"  {name}")


def _add_common_data_args(p):
    p.add_argument("--data_dir", required=True, help="Directory with train/val/test.csv")
    p.add_argument("--encoder", help="Encoder name (see `dnabind list-encoders`)")
    p.add_argument("--seq_length", type=int, default=20, help="Sequence length")
    p.add_argument("--log_dir", default="experiments/run", help="Output directory")
    p.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    p.add_argument("--test_num_samples", type=int, default=None)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="dnabind", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train a model and save a checkpoint")
    _add_common_data_args(p_train)
    p_train.add_argument("--arch_config", required=True, help="Architecture JSON path")
    p_train.add_argument("--checkpoint", default="experiments/run/best_model.pt")
    p_train.add_argument("--epochs", type=int, default=10)
    p_train.add_argument("--patience", type=int, default=3)
    p_train.add_argument("--train_num_samples", type=int, default=None)
    p_train.add_argument("--val_num_samples", type=int, default=None)
    p_train.add_argument(
        "--eval", action="store_true", help="Evaluate on test.csv after training"
    )
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("eval", help="Evaluate a saved checkpoint on test.csv")
    _add_common_data_args(p_eval)
    p_eval.add_argument("--checkpoint", required=True)
    p_eval.add_argument("--batch_size", type=int, default=None)
    p_eval.set_defaults(func=cmd_eval)

    p_le = sub.add_parser("list-encoders", help="List registered encoders")
    p_le.set_defaults(func=cmd_list_encoders)

    p_lm = sub.add_parser("list-models", help="List registered model types")
    p_lm.set_defaults(func=cmd_list_models)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
