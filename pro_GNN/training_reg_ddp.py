import os
import copy
import json
from datetime import datetime

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import numpy as np

from utils.loader import load_encoder
from config import args

def random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def pair_collate_fn(batch):
    data1_list, data2_list, label_list = zip(*batch)
    batch1 = Batch.from_data_list(data1_list)
    batch2 = Batch.from_data_list(data2_list)
    labels = torch.stack(label_list)

    return batch1, batch2, labels

class MoleculePairDataset(Dataset):
    def __init__(self, pair_list):
        """
        pair_list: [{'data1': Data, 'data2': Data, 'label': Tensor}, ...]
        """
        self.pair_list = pair_list

    def __len__(self):
        return len(self.pair_list)

    def __getitem__(self, idx):
        item = self.pair_list[idx]
        return item['data1'], item['data2'], item['label']

def make_head(in_dim: int, hidden: int, out_dim: int = 1) -> nn.Sequential:
    """回帰用の2-layer MLP ヘッドを生成"""
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(hidden, hidden // 2),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(hidden // 2, out_dim)
    )

class GraphDiffRegressor(nn.Module):
    def __init__(self, encoder: nn.Module, embedding_dim: int,
                 loss_type: str, hidden_dim: int = 64, out_dim: int = 1, merge_method: str = "diff"):
        super().__init__()
        assert loss_type in {"pair", "pair_bi", "pair_bi_sym"}
        self.loss_type = loss_type
        self.encoder   = encoder
        self.merge_method = merge_method

        self.mlp = make_head(embedding_dim, hidden_dim, out_dim)

    def forward(self, data_a, data_b):
        fa = self.encoder(data_a)
        fb = self.encoder(data_b)

        if self.merge_method == "diff":
            # 絶対値ではなく、符号付きの差分も考慮
            diff_feature = fa - fb
            pred = self.mlp(diff_feature)
        elif self.merge_method == "abs_diff":
            # 絶対値差分
            diff_feature = torch.abs(fa - fb)
            pred = self.mlp(diff_feature)
        elif self.merge_method == "product":
            product_feature = fa * fb
            pred = self.mlp(product_feature)
        else:
            raise ValueError(f"Unknown merge_method: {self.merge_method}")

        return pred

class GraphDiffRegressorCat(nn.Module):
    def __init__(self, encoder: nn.Module, embedding_dim: int,
                 loss_type: str, hidden_dim: int = 64, out_dim: int = 1):
        super().__init__()
        assert loss_type in {"pair", "pair_bi", "pair_bi_sym"}
        self.loss_type = loss_type
        self.encoder   = encoder

        # concatenationするので入力次元は2倍になる
        self.mlp = make_head(embedding_dim * 2, hidden_dim, out_dim)

    def forward(self, data_a, data_b):
        fa = self.encoder(data_a)
        fb = self.encoder(data_b)

        # 特徴量をconcatenate
        concat_feature = torch.cat([fa, fb], dim=1)
        pred = self.mlp(concat_feature)

        return pred

class Trainer:
    def __init__(self, args, train_loader=None, val_loader=None, test_loader=None):
        self.args = args
        self.local_rank = int(os.environ["LOCAL_RANK"])

        # loss_type
        self.loss_type = args.loss_type

        # ログディレクトリ作成
        self.log_dir = args.output_dir
        os.makedirs(self.log_dir, exist_ok=True)

        # ログファイルの設定
        checkpoint_path = os.path.join(self.log_dir, "checkpoint.pt")
        if self.local_rank == 0:  # マスターランクのみログを書き込む
            if os.path.exists(checkpoint_path):
                self.log_file = open(os.path.join(self.log_dir, "log.txt"), "a")
            else:
                self.log_file = open(os.path.join(self.log_dir, "log.txt"), "w")

        self.device = torch.device(f"cuda:{self.local_rank}")
        self.log_message(f"Using device: {self.device}")

        # save args
        if self.local_rank == 0:  # マスターランクのみ保存
            with open(os.path.join(args.output_dir, "args.json"), "w") as f:
                json.dump(vars(args), f)

        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.test_loader  = test_loader
        self.model_type = args.model_type

        # モデルとオプティマイザの初期化
        encoder = load_encoder(args).to(self.device)
        if self.model_type in ["diff", "abs_diff", "product"]:
            self.model = GraphDiffRegressor(
                encoder,
                embedding_dim=args.embedding_dim,
                loss_type=self.loss_type,
                hidden_dim=getattr(args, "hidden_dim", 64),
                out_dim=getattr(args, "out_dim", 1),
                merge_method=getattr(args, "model_type", "diff"),
            ).to(self.device)
            self.model = DDP(self.model, device_ids=[self.local_rank])
        elif self.model_type == "cat":
            self.model = GraphDiffRegressorCat(
                encoder,
                embedding_dim=args.embedding_dim,
                loss_type=self.loss_type,
                hidden_dim=getattr(args, "hidden_dim", 64),
                out_dim=getattr(args, "out_dim", 1),
            ).to(self.device)
            self.model = DDP(self.model, device_ids=[self.local_rank])
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=getattr(args, "weight_decay", 0.0)
        )

        # 学習率スケジューラの設定（回帰では損失の最小化が目標）
        self.scheduler_type = getattr(args, "scheduler", "reduce_on_plateau")
        if self.scheduler_type == "reduce_on_plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',  # 回帰では損失を最小化
                factor=0.5,
                patience=getattr(args, "lr_patience", 5),
                verbose=True
            )
        elif self.scheduler_type == "cosine_annealing":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=args.t_max if args.t_max is not None else args.epochs,
                eta_min=getattr(args, "min_lr", 1e-6)
            )
        elif self.scheduler_type == "step":
            self.scheduler = StepLR(
                self.optimizer,
                step_size=getattr(args, "step_size", 10),
                gamma=getattr(args, "gamma", 0.5)
            )
        else:
            self.scheduler = None

        # 早期停止などの記録用（回帰では損失の最小値を記録）
        self.best_loss = float('inf')
        self.best_model_state  = None
        self.early_stop_counter = 0
        self.train_losses = []
        self.val_losses   = []
        self.val_maes = []  # MAE を記録
        self.val_r2s = []   # R2 score を記録
        self.learning_rates = []

        # チェックポイントから再開するための変数
        self.start_epoch = 0

        # チェックポイントがあれば読み込む
        if os.path.exists(checkpoint_path):
            self.load_checkpoint(checkpoint_path)

    def log_message(self, message):
        """ログメッセージを標準出力とログファイルの両方に出力"""
        if self.local_rank == 0:  # マスターランクのみログを出力
            print(message)
            if hasattr(self, 'log_file'):
                self.log_file.write(message + "\n")
                self.log_file.flush()

    def compute_loss(self, pred_ab, y, pred_ba=None):
        # 回帰用の損失関数（MSE Loss）
        y_float = y.float().squeeze(-1)

        if self.loss_type == "pair":
            return F.mse_loss(pred_ab.squeeze(-1), y_float)
        elif self.loss_type == "pair_bi":
            # 双方向予測の場合、逆方向は-yになる
            return (F.mse_loss(pred_ab.squeeze(-1), y_float) + 
                   F.mse_loss(pred_ba.squeeze(-1), -y_float)) / 2.0
        elif self.loss_type == "pair_bi_sym":
            # 対称性制約も追加
            loss_ab = F.mse_loss(pred_ab.squeeze(-1), y_float)
            loss_ba = F.mse_loss(pred_ba.squeeze(-1), -y_float)
            sym_loss = F.mse_loss(pred_ab, -pred_ba)  # 対称性制約
            return (loss_ab + loss_ba) / 2.0 + 0.1 * sym_loss
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

    def save_checkpoint(self, epoch):
        """トレーニング状態をチェックポイントとして保存"""
        if self.local_rank == 0:  # マスターランクのみ保存
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': self.model.module.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_loss': self.best_loss,
                'best_model_state': self.best_model_state,
                'early_stop_counter': self.early_stop_counter,
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'val_maes': self.val_maes,
                'val_r2s': self.val_r2s,
                'learning_rates': self.learning_rates
            }

            if self.scheduler is not None:
                checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

            torch.save(checkpoint, os.path.join(self.log_dir, "checkpoint.pt"))
            self.log_message(f"Checkpoint saved at epoch {epoch}")

    def load_checkpoint(self, checkpoint_path):
        """チェックポイントから学習状態を復元"""
        self.log_message(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.model.module.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.start_epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        self.best_model_state = checkpoint['best_model_state']
        self.early_stop_counter = checkpoint['early_stop_counter']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.val_maes = checkpoint['val_maes']
        self.val_r2s = checkpoint['val_r2s']
        self.learning_rates = checkpoint['learning_rates']

        self.log_message(f"Resuming from epoch {self.start_epoch} with best loss: {self.best_loss:.4f}")

    def train(self):
        for epoch in range(self.start_epoch, self.args.epochs):
            self.model.train()
            total_loss = 0.0
            self.train_loader.sampler.set_epoch(epoch)

            # 現在の学習率を記録
            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)
            if self.local_rank == 0:
                self.log_message(f"Current learning rate: {current_lr:.6f}")

            # マスターランクのみtqdmを使用
            if self.local_rank == 0:
                train_iter = tqdm(self.train_loader, desc=f"Epoch {epoch:03d} (Train)")
            else:
                train_iter = self.train_loader

            for batch in train_iter:
                g1, g2, y = [x.to(self.device) for x in batch]
                pred_ab = self.model(g1, g2)
                if self.loss_type in ["pair_bi", "pair_bi_sym"]:
                    pred_ba = self.model(g2, g1)
                else:
                    pred_ba = None

                loss = self.compute_loss(pred_ab, y, pred_ba)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * g1.num_graphs

            # 各GPUの損失を集計
            world_size = dist.get_world_size()
            train_loss = total_loss / len(self.train_loader.dataset) * world_size
            train_loss_tensor = torch.tensor([train_loss], device=self.device)
            dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
            train_loss = train_loss_tensor.item() / world_size

            # 評価はマスターランクのみ実行
            if self.local_rank == 0:
                val_metrics = self.evaluate(self.val_loader)
                val_loss = val_metrics["avg_loss"]
                val_mae = val_metrics["mae"]
                val_r2 = val_metrics["r2"]

                self.train_losses.append(train_loss)
                self.val_losses.append(val_loss)
                self.val_maes.append(val_mae)
                self.val_r2s.append(val_r2)

                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.log_message(f"[{current_time}] Epoch {epoch:03d}: TrainLoss={train_loss:.4f}  "
                      f"ValLoss={val_loss:.4f}  ValMAE={val_mae:.4f}  ValR2={val_r2:.4f}")

                # 学習率スケジューラの更新
                if self.scheduler is not None:
                    if self.scheduler_type == "reduce_on_plateau":
                        self.scheduler.step(val_loss)  # 損失に基づいて学習率を調整
                    else:
                        self.scheduler.step()

                # early stopping（損失の最小化）
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.best_model_state = copy.deepcopy(self.model.module.state_dict())
                    self.early_stop_counter = 0
                    self.log_message(f"Best model {epoch}")
                    self.save_model(f"model_best.pt")
                else:
                    self.early_stop_counter += 1
                    if self.early_stop_counter >= self.args.patience:
                        self.log_message(f"Early stopping at epoch {epoch}")
                        break

                # チェックポイントを保存
                if epoch % 10 == 0:
                    self.save_checkpoint(epoch)

            # 早期停止の判断をすべてのランクで同期
            early_stop = torch.tensor([1 if self.early_stop_counter >= self.args.patience else 0], device=self.device)
            dist.broadcast(early_stop, src=0)
            if early_stop.item() == 1:
                break

        # ベストモデルの状態をすべてのランクで同期
        if self.local_rank == 0:
            self.model.module.load_state_dict(self.best_model_state)
            self.save_model("model_best.pt")
            self.save_plots()

            # 検証データセットの予測値を保存
            val_results = self.evaluate(self.val_loader, return_preds=True)
            self.save_predictions(val_results, "valid_predictions.npz")

        # すべてのプロセスを同期
        dist.barrier()

    def evaluate(self, loader, return_preds=False):
        self.model.eval()

        loss_sum, n_graphs = 0.0, 0
        y_true_list, y_pred_list_ab, y_pred_list_ba = [], [], []

        with torch.no_grad():
            for batch in loader:
                g1, g2, y = (x.to(self.device) for x in batch)
                pred_ab = self.model(g1, g2)
                if self.loss_type in ["pair_bi", "pair_bi_sym"]:
                    pred_ba = self.model(g2, g1)
                else:
                    pred_ba = None

                loss = self.compute_loss(pred_ab, y, pred_ba)
                loss_sum += loss.item() * g1.num_graphs
                n_graphs += g1.num_graphs

                y_true_batch = y.float().squeeze(-1).cpu()
                y_pred_batch_ab = pred_ab.squeeze(-1).cpu()
                y_pred_batch_ba = None if pred_ba is None else pred_ba.squeeze(-1).cpu()

                y_true_list.append(y_true_batch)
                y_pred_list_ab.append(y_pred_batch_ab)
                if pred_ba is not None:
                    y_pred_list_ba.append(y_pred_batch_ba)

        # 全バッチを結合
        y_true = torch.cat(y_true_list, dim=0)
        y_pred_ab = torch.cat(y_pred_list_ab, dim=0)

        if len(y_pred_list_ba) > 0:
            y_pred_ba = torch.cat(y_pred_list_ba, dim=0)
        else:
            y_pred_ba = None

        # 回帰指標の計算
        y_true_np = y_true.numpy()
        y_pred_np = y_pred_ab.numpy()
        
        mse = mean_squared_error(y_true_np, y_pred_np)
        mae = mean_absolute_error(y_true_np, y_pred_np)
        r2 = r2_score(y_true_np, y_pred_np)
        avg_loss = loss_sum / n_graphs

        if return_preds:
            return {
                "mse": mse,
                "mae": mae,
                "r2": r2,
                "avg_loss": avg_loss,
                "y_true": y_true,
                "y_pred_ab": y_pred_ab,
                "y_pred_ba": y_pred_ba
            }

        return {
            "mse": mse,
            "mae": mae,
            "r2": r2,
            "avg_loss": avg_loss
        }

    def test(self):
        if self.local_rank != 0:
            return None

        # ベストモデルを使用
        self.model.module.load_state_dict(self.best_model_state)
        self.model.eval()

        results = self.evaluate(self.test_loader, return_preds=True)

        # テストデータの予測値を保存
        self.save_predictions(results, "test_predictions.npz")

        # 指標をまとめてプリント
        self.log_message(f"Test MSE={results['mse']:.4f}  MAE={results['mae']:.4f}  "
              f"R2={results['r2']:.4f}  Loss={results['avg_loss']:.4f}")
        self.log_message(f"loss_type={self.loss_type}")

        # 散布図のプロット
        self.plot_predictions(results['y_true'], results['y_pred_ab'])

        return results['mse'], results['mae'], results['r2'], results['avg_loss']

    def save_model(self, filename="model_best.pt"):
        torch.save(self.model.module.state_dict(), os.path.join(self.log_dir, filename))
        self.log_message(f"Model saved → {self.log_dir}/{filename}")

    def save_predictions(self, results, filename):
        """予測結果をnpz形式で保存"""
        save_path = os.path.join(self.log_dir, filename)
        np.savez(
            save_path,
            y_true=results["y_true"].numpy(),
            y_pred_ab=results["y_pred_ab"].numpy(),
            y_pred_ba=results["y_pred_ba"].numpy() if results["y_pred_ba"] is not None else np.array([])
        )
        self.log_message(f"Predictions saved → {save_path}")

    def save_plots(self):
        plt.figure(figsize=(8,5))
        plt.plot(self.train_losses, label="Train Loss")
        plt.plot(self.val_losses, label="Val Loss")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "loss_curve.png")); plt.close()

        plt.figure(figsize=(8,5))
        plt.plot(self.val_maes, label="Val MAE", color="orange")
        plt.xlabel("Epoch"); plt.ylabel("MAE"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "mae_curve.png")); plt.close()

        plt.figure(figsize=(8,5))
        plt.plot(self.val_r2s, label="Val R²", color="green")
        plt.xlabel("Epoch"); plt.ylabel("R² Score"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "r2_curve.png")); plt.close()

        # 学習率の推移をプロット
        plt.figure(figsize=(8,5))
        plt.plot(self.learning_rates, label="Learning Rate", color="purple")
        plt.xlabel("Epoch"); plt.ylabel("Learning Rate"); plt.legend(); plt.tight_layout()
        plt.yscale('log')
        plt.savefig(os.path.join(self.log_dir, "lr_curve.png")); plt.close()

    def plot_predictions(self, y_true, y_pred):
        """予測値と真値の散布図を作成"""
        plt.figure(figsize=(8,8))
        y_true_np = y_true.numpy()
        y_pred_np = y_pred.numpy()
        
        plt.scatter(y_true_np, y_pred_np, alpha=0.6, s=10)
        
        # 対角線を描画
        min_val = min(y_true_np.min(), y_pred_np.min())
        max_val = max(y_true_np.max(), y_pred_np.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title('Predicted vs True Values')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "predictions_scatter.png"))
        plt.close()
        
        # 残差プロット
        plt.figure(figsize=(8,6))
        residuals = y_pred_np - y_true_np
        plt.scatter(y_true_np, residuals, alpha=0.6, s=10)
        plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
        plt.xlabel('True Values')
        plt.ylabel('Residuals (Predicted - True)')
        plt.title('Residual Plot')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "residuals_plot.png"))
        plt.close()

    def __del__(self):
        """デストラクタでログファイルを閉じる"""
        if hasattr(self, 'log_file') and self.local_rank == 0:
            self.log_file.close()

def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description='Graph Difference Regressor Training')

    # Data
    parser.add_argument('--input_file', type=str, required=True, help='Input directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--seed', type=int, default=41, help='Random seed')

    # Model
    parser.add_argument('--node_in', type=int, default=30, help='Input dimension for node features')
    parser.add_argument('--edge_in', type=int, default=11, help='Input dimension for edge features')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension size')
    parser.add_argument('--embedding_dim', type=int, default=64, help='Embedding dimension size')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')

    # Training
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=15, help='Patience for early stopping')

    # Learning rate scheduler
    parser.add_argument('--scheduler', type=str, default='reduce_on_plateau',
                        choices=['reduce_on_plateau', 'cosine_annealing', 'step', 'none'],
                        help='Learning rate scheduler type')
    parser.add_argument('--lr_patience', type=int, default=5,
                        help='Patience for ReduceLROnPlateau scheduler')
    parser.add_argument('--t_max', type=int, default=None,
                        help='T_max for CosineAnnealingLR scheduler')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='Minimum learning rate for CosineAnnealingLR')
    parser.add_argument('--step_size', type=int, default=10,
                        help='Step size for StepLR scheduler')
    parser.add_argument('--gamma', type=float, default=0.5,
                        help='Gamma for StepLR scheduler')

    # Loss
    parser.add_argument('--loss_type', type=str, default='pair', help='Loss type: pair, pair_bi, pair_bi_sym')
    parser.add_argument('--model_type', type=str, default='diff', 
                        help='Model type: diff, abs_diff, cat, product', 
                        choices=['diff', 'abs_diff', 'cat', 'product'])
    parser.add_argument('--inference-only', action='store_true', help='Inference only')

    return parser.parse_args()

def main(args):
    # Set random seed for reproducibility
    random_seed(args.seed)

    # Enable anomaly detection for debugging
    torch.autograd.set_detect_anomaly(True)

    # 分散初期化
    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    dataset = torch.load(args.input_file, weights_only=False)

    # DataLoaderの作成
    train_sampler = DistributedSampler(dataset["train"],
                                       num_replicas=dist.get_world_size(),
                                       rank=local_rank,
                                       shuffle=True)
    train_loader = DataLoader(dataset["train"],
                              batch_size=args.batch_size,
                              sampler=train_sampler,
                              collate_fn=pair_collate_fn,
                              num_workers=0)
    val_loader   = DataLoader(dataset["valid"], batch_size=args.batch_size, shuffle=False,
                             collate_fn=pair_collate_fn, num_workers=0)
    test_loader  = DataLoader(dataset["test"], batch_size=args.batch_size, shuffle=False,
                             collate_fn=pair_collate_fn, num_workers=0)

    # Trainer 実行
    trainer = Trainer(args, train_loader, val_loader, test_loader)
    if not args.inference_only:
        trainer.train()
        trainer.test()
    else:
        # model_bestを読み込む
        model_path = os.path.join(args.output_dir, "model_best.pt")
        trainer.model.module.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=False))
        trainer.test()
    
    # 分散処理の終了
    dist.destroy_process_group()

if __name__ == "__main__":
    args = parse_args()
    main(args)