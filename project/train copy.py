#!/usr/bin/env python3

"""
Bonito training.
"""

import os
from argparse import ArgumentParser
from argparse import ArgumentDefaultsHelpFormatter
from pathlib import Path
from importlib import import_module

from bonito.data import load_numpy, load_script
from bonito.util import __models_dir__, default_config
from bonito.util import load_model, load_symbol, init, half_supported
from bonito.training import Trainer

import toml
import torch
from torch.utils.data import DataLoader

import optuna

def objective(trial, args):
    
    trial_dir = os.path.join(args.training_directory, f"trial_{trial.number}")
    os.makedirs(trial_dir, exist_ok=True)

    # Update the training directory argument
    #args.training_directory = trial_dir
    workdir = os.path.expanduser(trial_dir)
    init(args.seed, args.device, (not args.nondeterministic))
    device = torch.device(args.device)

    if not args.pretrained:
        config = toml.load(args.config)
    else:
        dirname = args.pretrained
        if not os.path.isdir(dirname) and os.path.isdir(os.path.join(__models_dir__, dirname)):
            dirname = os.path.join(__models_dir__, dirname)
        pretrain_file = os.path.join(dirname, 'config.toml')
        config = toml.load(pretrain_file)
        if 'lr_scheduler' in config:
            print(f"[ignoring 'lr_scheduler' in --pretrained config]")
            del config['lr_scheduler']

    argsdict = dict(training=vars(args))

    

    
    # Suggest hyperparameters for training
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    #grad_accum_split = trial.suggest_int("grad_accum_split", 1, 4)
    max_norm = trial.suggest_uniform("max_norm", 0.1, 5.0)

    # Suggest hyperparameters for the model configuration from the TOML file
    #conv_layers = trial.suggest_int("conv_layers", 2, 10)
    ssm_d_state = trial.suggest_int("ssm_d_state", 32, 256, step=32)
    attn_headdim = trial.suggest_int("attn_headdim", 32, 64, step=32)
    mamba_layers = trial.suggest_int("mamba_layers", 1, 6)

    # Load and update the config.toml file
    config = toml.load(args.config)

    # Update the model configuration with suggested hyperparameters
    #config["model"]["conv_layers"] = conv_layers
    config["model"]["encoder"]["transformer_encoder"]["layer"]["d_state"] = ssm_d_state
    config["model"]["encoder"]["transformer_encoder"]["layer"]["headdim"] = attn_headdim
    config["model"]["encoder"]["transformer_encoder"]["layer"]["nlayer"] = mamba_layers

    # Initialize device
    init(args.seed, args.device, (not args.nondeterministic))
    device = torch.device(args.device)

    # Load model with updated config
    print("[loading model]")
    if args.pretrained:
        print("[using pretrained model {}]".format(args.pretrained))
        model = load_model(args.pretrained, device, half=False)
    else:
        model = load_symbol(config, 'Model')(config)

    print("[loading data]")
    try:
        train_loader_kwargs, valid_loader_kwargs = load_numpy(
            args.chunks, args.directory, valid_chunks = args.valid_chunks
        )
    except FileNotFoundError:
        train_loader_kwargs, valid_loader_kwargs = load_script(
            args.directory,
            seed=args.seed,
            chunks=args.chunks,
            valid_chunks=args.valid_chunks,
            n_pre_context_bases=getattr(model, "n_pre_context_bases", 0),
            n_post_context_bases=getattr(model, "n_post_context_bases", 0),
        )

    

    loader_kwargs = {
        "batch_size": args.batch, "num_workers": args.num_workers, "pin_memory": True
    }
    train_loader = DataLoader(**loader_kwargs, **train_loader_kwargs)
    valid_loader = DataLoader(**loader_kwargs, **valid_loader_kwargs)

    os.makedirs(workdir, exist_ok=True)
    toml.dump({**config, **argsdict}, open(os.path.join(workdir, 'config.toml'), 'w'))

    if config.get("lr_scheduler"):
        sched_config = config["lr_scheduler"]
        lr_scheduler_fn = getattr(
            import_module(sched_config["package"]), sched_config["symbol"]
        )(**sched_config)
    else:
        lr_scheduler_fn = None

    trainer = Trainer(
        model, device, train_loader, valid_loader,
        use_amp=half_supported() and not args.no_amp,
        lr_scheduler_fn=lr_scheduler_fn,
        restore_optim=args.restore_optim,
        save_optim_every=args.save_optim_every,
        grad_accum_split=args.grad_accum_split,
        quantile_grad_clip=args.quantile_grad_clip
    )


    # Clip gradient norm
    trainer.clip_grad = lambda parameters: torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_norm).item()

    # Initialize optimizer
    optim_kwargs = config.get("optim", {})
    #trainer.init_optimizer(lr, **optim_kwargs)
    try:
        best_val_mean = trainer.fit_optuna(trial_dir, trial, args.epochs, lr, **optim_kwargs)
        return best_val_mean
    except RuntimeError as e:
        # Check if the error is related to CUDA out of memory
        raise optuna.TrialPruned()

def main(args):
    workdir = os.path.expanduser(args.training_directory)
    if os.path.exists(workdir) and not args.force:
        print("[error] %s exists, use -f to force continue training." % workdir)
        exit(1)

    if not args.optuna:
        #workdir = os.path.expanduser(args.training_directory)

        
        init(args.seed, args.device, (not args.nondeterministic))
        device = torch.device(args.device)

        if not args.pretrained:
            config = toml.load(args.config)
        else:
            dirname = args.pretrained
            if not os.path.isdir(dirname) and os.path.isdir(os.path.join(__models_dir__, dirname)):
                dirname = os.path.join(__models_dir__, dirname)
            pretrain_file = os.path.join(dirname, 'config.toml')
            config = toml.load(pretrain_file)
            if 'lr_scheduler' in config:
                print(f"[ignoring 'lr_scheduler' in --pretrained config]")
                del config['lr_scheduler']

        argsdict = dict(training=vars(args))

        print("[loading model]")
        if args.pretrained:
            print("[using pretrained model {}]".format(args.pretrained))
            model = load_model(args.pretrained, device, half=False)
        else:
            model = load_symbol(config, 'Model')(config)

        print("[loading data]")
        try:
            train_loader_kwargs, valid_loader_kwargs = load_numpy(
                args.chunks, args.directory, valid_chunks = args.valid_chunks
            )
        except FileNotFoundError:
            train_loader_kwargs, valid_loader_kwargs = load_script(
                args.directory,
                seed=args.seed,
                chunks=args.chunks,
                valid_chunks=args.valid_chunks,
                n_pre_context_bases=getattr(model, "n_pre_context_bases", 0),
                n_post_context_bases=getattr(model, "n_post_context_bases", 0),
            )

        loader_kwargs = {
            "batch_size": args.batch, "num_workers": args.num_workers, "pin_memory": True
        }
        train_loader = DataLoader(**loader_kwargs, **train_loader_kwargs)
        valid_loader = DataLoader(**loader_kwargs, **valid_loader_kwargs)

        os.makedirs(workdir, exist_ok=True)
        toml.dump({**config, **argsdict}, open(os.path.join(workdir, 'config.toml'), 'w'))

        if config.get("lr_scheduler"):
            sched_config = config["lr_scheduler"]
            lr_scheduler_fn = getattr(
                import_module(sched_config["package"]), sched_config["symbol"]
            )(**sched_config)
        else:
            lr_scheduler_fn = None

        trainer = Trainer(
            model, device, train_loader, valid_loader,
            use_amp=half_supported() and not args.no_amp,
            lr_scheduler_fn=lr_scheduler_fn,
            restore_optim=args.restore_optim,
            save_optim_every=args.save_optim_every,
            grad_accum_split=args.grad_accum_split,
            quantile_grad_clip=args.quantile_grad_clip
        )

        if (',' in args.lr):
            lr = [float(x) for x in args.lr.split(',')]
        else:
            lr = float(args.lr)
        optim_kwargs = config.get("optim", {})
        trainer.fit(workdir, args.epochs, lr, **optim_kwargs)
    else:
        os.makedirs(workdir, exist_ok=True)
        study_name = "basecaller_hyperopt"
        storage_url = f"sqlite:///{workdir}/{study_name}.db"
        print(storage_url)
        # Create an Optuna study and optimize the objective function
        study = optuna.create_study(study_name="basecaller_hyperopt",direction="maximize", pruner=optuna.pruners.MedianPruner(), storage=storage_url, load_if_exists=True)
        study.optimize(lambda trial: objective(trial, args), n_trials=50, timeout=None)

        # Print the best hyperparameters and save them
        print("Best hyperparameters:", study.best_params)
        print("Best validation loss:", study.best_value)

        # Save the best hyperparameters to a file
        best_params_path = os.path.join(workdir, "best_hyperparams.toml")
        with open(best_params_path, "w") as f:
            toml.dump(study.best_params, f)

def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    parser.add_argument("training_directory")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--config', default=default_config)
    group.add_argument('--pretrained', default="")
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lr", default='2e-3')
    parser.add_argument("--seed", default=25, type=int)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--batch", default=64, type=int)
    parser.add_argument("--chunks", default=0, type=int)
    parser.add_argument("--valid-chunks", default=None, type=int)
    parser.add_argument("--no-amp", action="store_true", default=False)
    parser.add_argument("-f", "--force", action="store_true", default=False)
    parser.add_argument("--restore-optim", action="store_true", default=False)
    parser.add_argument("--nondeterministic", action="store_true", default=False)
    parser.add_argument("--save-optim-every", default=10, type=int)
    parser.add_argument("--grad-accum-split", default=1, type=int)
    parser.add_argument("--quantile-grad-clip", action="store_true", default=False)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--optuna", action="store_true", default=False, help="Enable hyperparameter optimization with Optuna.")

    return parser
