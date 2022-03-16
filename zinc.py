import os
import json
import torch
import torch.nn.functional as F
import argparse

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch_geometric.datasets import ZINC

from grpe.lr import PolynomialDecayLR
from grpe.model import GRPENetwork
from grpe.dataset.tansform import (
    ShortestPathGenerator,
    MoleculeCollator,
)


def train(model, train_dataset, optimizer, lr_scheduler, device=None, input_dim=25):
    model.train()

    losses = []
    for batch in tqdm(train_dataset):
        batch.to(device)
        y = batch.y

        out = model(batch)
        loss = F.l1_loss(out.squeeze(1), y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        losses.append(loss.detach())

    return torch.stack(losses).mean()


@torch.no_grad()
def evaluate(model, test_dataset, device=None):
    model.eval()
    mae = []
    for batch in test_dataset:
        batch.to(device)
        batch.x = batch.x.squeeze(1)
        y = batch.y

        out = model(batch)
        mae.append((y - out.squeeze(1)).abs())

    mae = torch.cat(mae, dim=0).mean().item()
    return mae


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--node-dim", type=int, default=80)
    parser.add_argument("--ffn-dim", type=int, default=80)
    parser.add_argument("--num-layer", type=int, default=12)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num-node-type", type=int, default=25)
    parser.add_argument("--num-edge-type", type=int, default=10)
    parser.add_argument("--max-hop", type=int, default=5)
    parser.add_argument("--use-independent-token", default=False, action="store_true")

    parser.add_argument("--perturb-noise", default=0.2, type=float)
    parser.add_argument("--num-last-mlp", default=0, type=int)

    parser.add_argument("--weight-decay", default=1e-3, type=float)
    parser.add_argument("--peak-lr", default=2e-4, type=float)
    parser.add_argument("--end-lr", default=1e-9, type=float)
    parser.add_argument("--tot-updates", default=400000, type=int)
    parser.add_argument("--warmup-updates", default=40000, type=int)

    parser.add_argument("--data-root", default="data")
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--save", required=True)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = DataLoader(
        ZINC(
            args.data_root,
            subset=True,
            split="train",
            transform=ShortestPathGenerator(),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=MoleculeCollator(),
        num_workers=8,
    )

    val_dataset = DataLoader(
        ZINC(
            args.data_root,
            subset=True,
            split="val",
            transform=ShortestPathGenerator(),
        ),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=MoleculeCollator(),
        num_workers=8,
    )
    test_dataset = DataLoader(
        ZINC(
            args.data_root,
            subset=True,
            split="test",
            transform=ShortestPathGenerator(),
        ),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=MoleculeCollator(),
        num_workers=8,
    )

    model = GRPENetwork(
        num_task=1,
        d_model=args.node_dim,
        dim_feedforward=args.ffn_dim,
        num_layer=args.num_layer,
        nhead=args.nhead,
        max_hop=args.max_hop,
        num_node_type=args.num_node_type,
        num_edge_type=args.num_edge_type,
        use_independent_token=args.use_independent_token,
        perturb_noise=args.perturb_noise,
        num_last_mlp=args.num_last_mlp,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.peak_lr, weight_decay=args.weight_decay
    )
    lr_scheduler = PolynomialDecayLR(
        optimizer,
        warmup_updates=args.warmup_updates,
        tot_updates=args.tot_updates,
        lr=args.peak_lr,
        end_lr=args.end_lr,
        power=1.0,
    )

    best_val_mae = 100  # Lower is better
    test_mae = 100

    max_epoch = args.tot_updates // len(train_dataset)

    total_params = sum(p.numel() for p in model.parameters())

    # logging performance
    os.makedirs(args.save, exist_ok=True)
    with open(f"{args.save}/performance.log", "w") as f:
        f.write(json.dumps(vars(args), indent=4, sort_keys=True) + "\n")

    for epoch in range(1, max_epoch + 1):
        loss = train(
            model,
            train_dataset,
            optimizer,
            lr_scheduler,
            device=device,
        )
        val_mae = evaluate(model, val_dataset, device=device)

        if best_val_mae > val_mae:
            best_val_mae = val_mae
            test_mae = evaluate(model, test_dataset, device=device)
            torch.save(model.state_dict(), f"{args.save}/model.pt")

        print(
            f"[Ep {epoch}/{max_epoch}] train-loss: {loss.item():4f}, val-mae: {val_mae:4f}, test-mae: {test_mae:4f}"
        )

        # logging performance
        with open(f"{args.save}/performance.log", "a") as f:
            f.write(
                "[Ep {epoch}/{max_epoch}] train-loss: {loss.item():4f}, val-mae: {val_mae:4f}, test-mae: {test_mae:4f}\n"
            )
