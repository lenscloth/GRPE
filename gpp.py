import os
import sys
import json
import torch
import torch.nn.functional as F
import random
import argparse
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader
from collections import namedtuple

import torch.multiprocessing as mp
import torch.distributed as dist

from grpe.lr import PolynomialDecayLR
from grpe.model import GRPENetwork, load_grpe_backbone
from grpe.dataset.tansform import (
    ShortestPathGenerator,
    MoleculeCollator,
    OneHotEdgeAttr,
    ImageCollator,
)
from grpe.distributed import (
    barrier,
    init_process,
    is_master,
    get_world_size,
    gather_uneven_tensors,
    set_master_only_print,
)

from ogb.graphproppred import Evaluator
from ogb.lsc import PCQM4Mv2Evaluator

from grpe.dataset import (
    PygPCQM4MDataset,
    PygPCQM4Mv2Dataset,
    PygGraphPropPredDataset,
    GNNBenchmarkDataset,
    GNNBenchmarkEvaluator,
)
from torchvision.transforms import Compose


def train(
    model, train_dataset, config, optimizer, lr_scheduler, device=None, grad_norm=None
):
    model.train()
    loss_fn = config.loss_fn

    losses = []
    for batch in tqdm(train_dataset, desc="[Training]"):
        batch.to(device)
        out = model(batch)
        if loss_fn == F.cross_entropy:
            y = batch.y.long()
        else:
            y = batch.y.view(out.shape).float()
        mask = ~torch.isnan(y)
        loss = loss_fn(out[mask], y[mask])

        optimizer.zero_grad()
        loss.backward()

        if grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)

        optimizer.step()
        lr_scheduler.step()
        losses.append(loss.detach())

    return torch.stack(losses).mean()


@torch.no_grad()
def evaluate(model, test_dataset, config, device=None):
    model.eval()
    evaluator = config.evaluator

    y_pred = []
    y_true = []

    for batch in tqdm(test_dataset, desc="[Evaluating]"):
        batch.to(device)
        out = model(batch)
        if config.loss_fn == F.cross_entropy:
            y = batch.y.long()
        else:
            y = batch.y.view(out.shape).float()

        y_pred.append(out)
        y_true.append(y)

    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred)

    if isinstance(evaluator, PCQM4Mv2Evaluator):
        y_true = y_true.squeeze(1)
        y_pred = y_pred.squeeze(1)

    if get_world_size() > 0:
        y_true = gather_uneven_tensors(y_true)
        y_pred = gather_uneven_tensors(y_pred)

    if config.loss_fn == F.cross_entropy:
        hit = torch.sum(torch.argmax(y_pred, dim=1) == y_true).item()
        total = len(y_true)
        result = {"acc": hit / total}
    else:
        result = evaluator.eval({"y_true": y_true, "y_pred": y_pred})
    return result


def is_left_better(left, right, metric="mae"):
    if left is None:
        return False
    elif right is None:
        return True
    if metric in ["mae"]:
        return left < right
    if metric in ["ap", "rocauc", "acc"]:
        return left > right
    else:
        raise ValueError()


def load_dataset(name, root="data/", transform=None):
    val_dataset, test_dataset = None, None
    if name == "pcba":
        dataset = PygGraphPropPredDataset(
            name="ogbg-molpcba", root=root, transform=transform
        )
    elif name == "hiv":
        dataset = PygGraphPropPredDataset(
            name="ogbg-molhiv", root=root, transform=transform
        )
    elif name == "lsc-v2":
        dataset = PygPCQM4Mv2Dataset(root=root, transform=transform)
    elif name == "lsc-v1":
        dataset = PygPCQM4MDataset(root=root, transform=transform)
    elif name == "cifar10":
        dataset = GNNBenchmarkDataset(
            root=root,
            split="train",
            name="CIFAR10",
            transform=transform,
            concat_position=True,
        )
        val_dataset = GNNBenchmarkDataset(
            root=root,
            split="val",
            name="CIFAR10",
            transform=transform,
            concat_position=True,
        )
        test_dataset = GNNBenchmarkDataset(
            root=root,
            split="test",
            name="CIFAR10",
            transform=transform,
            concat_position=True,
        )
    elif name == "mnist":
        dataset = GNNBenchmarkDataset(
            root=root,
            split="train",
            name="MNIST",
            transform=transform,
            concat_position=True,
        )
        val_dataset = GNNBenchmarkDataset(
            root=root,
            split="val",
            name="MNIST",
            transform=transform,
            concat_position=True,
        )
        test_dataset = GNNBenchmarkDataset(
            root=root,
            split="test",
            name="MNIST",
            transform=transform,
            concat_position=True,
        )
    else:
        raise ValueError(f"Invalid args.dataset: {name}")
    return dataset, val_dataset, test_dataset


def train_process(rank, args):
    """Training script."""
    if args.distributed:
        init_process(rank, args.world_size, args.port)
        torch.cuda.set_device(rank)
        set_master_only_print()

    ExpConfig = namedtuple("ExpConfig", ["evaluator", "metric", "num_task", "loss_fn"])
    experiments = {
        "pcba": ExpConfig(
            Evaluator(name="ogbg-molpcba"),
            "ap",
            128,
            F.binary_cross_entropy_with_logits,
        ),
        "hiv": ExpConfig(
            Evaluator(name="ogbg-molhiv"),
            "rocauc",
            1,
            F.binary_cross_entropy_with_logits,
        ),
        "lsc-v2": ExpConfig(PCQM4Mv2Evaluator(), "mae", 1, F.l1_loss),
        "lsc-v1": ExpConfig(PCQM4Mv2Evaluator(), "mae", 1, F.l1_loss),
        "cifar10": ExpConfig(
            GNNBenchmarkEvaluator(name="cifar10", num_tasks=10),
            "acc",
            10,
            F.cross_entropy,
        ),
        "mnist": ExpConfig(
            GNNBenchmarkEvaluator(name="mnist", num_tasks=10),
            "acc",
            10,
            F.cross_entropy,
        ),
    }

    config = experiments[args.dataset]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.distributed and is_master():
        # Master node load dataset first to process pre_transform
        load_dataset(
            args.dataset,
            root="data/",
            transform=Compose([ShortestPathGenerator(), OneHotEdgeAttr()]),
        )
    barrier()

    dataset, valid_dataset, test_dataset = load_dataset(
        args.dataset,
        root="data/",
        transform=Compose([ShortestPathGenerator(), OneHotEdgeAttr()]),
    )

    if valid_dataset is None and test_dataset is None:
        split_idx = dataset.get_idx_split()
        train_dataset = dataset[split_idx["train"]]
        valid_dataset = dataset[split_idx["valid"]]

        if args.dataset in ["lsc-v2"]:
            test_dataset = dataset[split_idx["test-dev"]]
        else:
            test_dataset = dataset[split_idx["test"]]
    else:
        train_dataset = dataset

    if dist.is_initialized():
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        valid_sampler = torch.utils.data.SequentialSampler(valid_dataset)
        test_sampler = torch.utils.data.SequentialSampler(test_dataset)

    if any([key in args.dataset for key in ["mnist", "cifar10"]]):
        collate_fn = ImageCollator()
        collate_fn_val = ImageCollator()
    else:
        collate_fn = MoleculeCollator(max_node=128)
        collate_fn_val = MoleculeCollator()

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        sampler=train_sampler,
        num_workers=args.num_workers,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn_val,
        sampler=valid_sampler,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn_val,
        sampler=test_sampler,
        num_workers=args.num_workers,
    )

    model = GRPENetwork(
        num_task=config.num_task,
        d_model=args.node_dim,
        dim_feedforward=args.ffn_dim,
        num_layer=args.num_layer,
        nhead=args.nhead,
        max_hop=args.max_hop,
        num_node_type=args.num_node_type,
        num_edge_type=args.num_edge_type,
        use_independent_token=args.use_independent_token,
        perturb_noise=args.perturb_noise,
        dropout=args.dropout,
    )

    if args.load is not None:
        print(f"Load model params from: {args.load}")
        load_grpe_backbone(model, args.load, skip_task_branch=(not args.load_all))

    model.to(device)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    optimizer = torch.optim.AdamW(
        list(model.parameters()), lr=args.peak_lr, weight_decay=args.weight_decay
    )

    lr_scheduler = PolynomialDecayLR(
        optimizer,
        warmup_updates=args.warmup_epoch * len(train_loader),
        tot_updates=args.max_epoch * len(train_loader),
        lr=args.peak_lr,
        end_lr=args.end_lr,
        power=1.0,
    )

    best_val_perf = None
    val_perf = None
    test_perf = None
    # logging args
    if is_master():
        os.makedirs(args.save, exist_ok=True)
        total_params = sum(p.numel() for p in model.parameters())
        total_size = sum(p.numel() * p.element_size() for p in model.parameters())

        with open(f"{args.save}/performance.log", "w") as f:
            f.write(" ".join(["python"] + list(sys.argv)) + "\n")
            f.write(f"Total #param: {total_params}\n")
            f.write(f"Total size: {total_size}\n")
            f.write(json.dumps(vars(args), indent=4, sort_keys=True) + "\n")

    # validate performance of the loaded model
    if args.load_all:
        valid_loader.dataset.delete_data()
        val_perf = evaluate(model, valid_loader, config, device=device)[config.metric]
        print(
            f"Valid performance of the loaded model ({args.load}): {val_perf}-{config.metric}"
        )
        test_loader.dataset.delete_data()
        test_perf = evaluate(model, test_loader, config, device=device)[config.metric]
        print(
            f"Test performance of the loaded model ({args.load}): {test_perf}-{config.metric}"
        )

    # logging
    perfs = dict(val=dict(), test=dict())

    end_epoch = (
        args.max_epoch
        if args.early_stop_epoch is None
        else min(args.max_epoch, args.early_stop_epoch)
    )
    for epoch in range(1, end_epoch + 1):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)

        # In order to load dataset in memory after DataLoader workers are spawned
        train_loader.dataset.delete_data()
        loss = train(
            model,
            train_loader,
            config,
            optimizer,
            lr_scheduler,
            device=device,
            grad_norm=args.grad_norm,
        )

        if (
            (epoch > args.valid_after)
            or (epoch % args.valid_every) == 0
            or (epoch == 1)
        ):
            valid_loader.dataset.delete_data()
            val_perf = evaluate(model, valid_loader, config, device=device)[
                config.metric
            ]
            perfs["val"][epoch] = val_perf

            if is_left_better(val_perf, best_val_perf, metric=config.metric):
                best_val_perf = val_perf
                test_loader.dataset.delete_data()
                test_perf = evaluate(model, test_loader, config, device=device)[
                    config.metric
                ]
                perfs["test"][epoch] = test_perf
                if is_master():
                    torch.save(model.state_dict(), f"{args.save}/model.pt")

        if is_master():
            print(
                f"[Ep {epoch}/{args.max_epoch}] train-loss: {loss.item():4f}, val-{config.metric}: {val_perf:4f} ({best_val_perf:4f}), test-{config.metric}: {test_perf:4f}"
            )
            # logging performance
            with open(f"{args.save}/performance.log", "a") as f:
                f.write(
                    f"[Ep {epoch}/{args.max_epoch}] train-loss/val-{config.metric}/best-val-{config.metric}/test-{config.metric}: {loss.item():4f}, {val_perf:4f}, ({best_val_perf:4f}), {test_perf:4f}\n"
                )
    if is_master():
        # save all performance records
        torch.save(perfs, f"{args.save}/perfs.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--port", type=int, default=12345)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--dataset",
        choices=[
            "pcba",
            "hiv",
            "lsc-v1",
            "lsc-v2",
            "cifar10",
            "mnist",
        ],
        default="hiv",
    )
    parser.add_argument("--node-dim", type=int, default=768)
    parser.add_argument("--ffn-dim", type=int, default=768)
    parser.add_argument("--num-layer", type=int, default=12)
    parser.add_argument("--nhead", type=int, default=32)
    parser.add_argument("--num-node-type", type=int, default=512 * 9 + 1)
    parser.add_argument("--num-edge-type", type=int, default=30)
    parser.add_argument("--max-hop", type=int, default=5)
    parser.add_argument("--use-independent-token", default=False, action="store_true")

    parser.add_argument("--perturb-noise", default=0.0, type=float)
    parser.add_argument("--grad-norm", default=5.0, type=float)
    parser.add_argument("--dropout", default=0.0, type=float)
    parser.add_argument("--attention-dropout", default=0.0, type=float)

    parser.add_argument("--weight-decay", default=0.0, type=float)
    parser.add_argument("--peak-lr", default=2e-4, type=float)
    parser.add_argument("--end-lr", default=1e-9, type=float)

    parser.add_argument(
        "--lr-scheduler", choices=["multi_step", "polynomial"], default="polynomial"
    )
    parser.add_argument(
        "--lr-milestones", nargs="+", default=[30, 60, 90, 120, 150, 180]
    )
    parser.add_argument("--lr-gamma", default=0.5, type=float)

    parser.add_argument("--max-epoch", default=100, type=int)
    parser.add_argument("--early-stop-epoch", default=None, type=int)
    parser.add_argument("--warmup-epoch", default=3, type=int)

    parser.add_argument("--data-root", default="data")
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--save", required=True)
    parser.add_argument("--valid-every", default=1, type=int)
    parser.add_argument("--valid-after", default=200, type=int)
    parser.add_argument("--load", default=None)
    parser.add_argument("--load-all", default=False, action="store_true")

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)

    args.distributed = args.world_size > 1

    if args.distributed:
        mp.spawn(train_process, nprocs=args.world_size, args=(args,))
    else:
        # Simply call main_worker function
        train_process(0, args)
