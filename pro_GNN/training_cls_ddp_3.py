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
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)

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



def make_head(in_dim: int, hidden: int, out_dim: int = 2) -> nn.Sequential:
    """2-layer MLP ヘッドを生成"""
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden // 2),
        nn.ReLU(),
        nn.Linear(hidden // 2, out_dim)
    )

class GraphDiffClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, embedding_dim: int,
                 loss_type: str, hidden_dim: int = 64, out_dim: int = 2, merge_method: str = "diff"):
        super().__init__()
        assert loss_type in {"pair", "pair_bi", "pair_bi_sym"}
        self.loss_type = loss_type
        self.encoder   = encoder
        self.merge_method = merge_method

        self.mlp = make_head(embedding_dim, hidden_dim, out_dim)

    def forward(self, data_a, data_b):
        fa = self.encoder(data_a)
        fb = self.encoder(data_b)

        # デフォルトは None にしておき、存在するヘッドだけ計算
        if self.merge_method == "diff":
            diff_feature = torch.abs(fa - fb)
            pred = self.mlp(diff_feature)
        elif self.merge_method == "product":
            product_feature = fa * fb
            pred = self.mlp(product_feature)
        else:
            raise ValueError(f"Unknown merge_method: {self.merge_method}")

        return pred
class GraphDiffClassifierCat(nn.Module):
    def __init__(self, encoder: nn.Module, embedding_dim: int,
                 loss_type: str, hidden_dim: int = 64, out_dim: int = 2):
        super().__init__()
        assert loss_type in {"pair", "pair_bi", "pair_bi_sym"}
        self.loss_type = loss_type
        self.encoder   = encoder

        # loss_type ごとに必要なヘッドだけ作る
        # concatenationするので入力次元は2倍になる
        self.mlp = make_head(embedding_dim * 2, hidden_dim, out_dim)

    def forward(self, data_a, data_b):
        fa = self.encoder(data_a)
        fb = self.encoder(data_b)

        # 差分ではなく特徴量をconcatenate
        concat_feature = torch.cat([fa, fb], dim=1)
        pred = self.mlp(concat_feature)

        return pred
class Trainer:
    def __init__(self, args, train_loader=None, val_loader=None, test_loader=None):
        self.args = args
        self.local_rank = int(os.environ["LOCAL_RANK"])

        # loss_type／weights
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

        # スケジューラタイプを先に設定（ログ出力で使用するため）
        self.scheduler_type = getattr(args, "scheduler", "reduce_on_plateau")

        # ハイパーパラメータをログ出力
        if self.local_rank == 0:
            self.log_message("=" * 50)
            self.log_message("Hyperparameters:")
            self.log_message(f"  Batch size: {args.batch_size}")
            self.log_message(f"  Learning rate: {args.lr}")
            self.log_message(f"  Weight decay: {getattr(args, 'weight_decay', 0.0)}")
            self.log_message(f"  Epochs: {args.epochs}")
            self.log_message(f"  Patience (early stopping): {args.patience}")
            self.log_message(f"  Random seed: {args.seed}")
            self.log_message(f"  Model type: {args.model_type}")
            self.log_message(f"  Loss type: {self.loss_type}")
            self.log_message(f"  Node input dim: {args.node_in}")
            self.log_message(f"  Edge input dim: {args.edge_in}")
            self.log_message(f"  Hidden dim: {getattr(args, 'hidden_dim', 64)}")
            self.log_message(f"  Embedding dim: {args.embedding_dim}")
            self.log_message(f"  Number of layers: {args.num_layers}")
            self.log_message(f"  Dropout: {args.dropout}")
            self.log_message(f"  Scheduler: {self.scheduler_type}")
            if self.scheduler_type == "reduce_on_plateau":
                self.log_message(f"    LR patience: {getattr(args, 'lr_patience', 5)}")
            elif self.scheduler_type == "cosine_annealing":
                self.log_message(f"    T_max: {args.t_max if args.t_max is not None else args.epochs}")
                self.log_message(f"    Min LR: {getattr(args, 'min_lr', 1e-6)}")
            elif self.scheduler_type == "step":
                self.log_message(f"    Step size: {getattr(args, 'step_size', 10)}")
                self.log_message(f"    Gamma: {getattr(args, 'gamma', 0.5)}")
            self.log_message("=" * 50)

        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.test_loader  = test_loader
        self.model_type = args.model_type

        # モデルとオプティマイザの初期化
        encoder = load_encoder(args).to(self.device)
        if self.model_type in ["diff", "product"]:
            self.model = GraphDiffClassifier(
                encoder,
                embedding_dim=args.embedding_dim,
                loss_type=self.loss_type,
                hidden_dim=getattr(args, "hidden_dim", 64),
                out_dim=getattr(args, "out_dim", 2),
                merge_method=getattr(args, "model_type", "diff"),
            ).to(self.device)
            self.model = DDP(self.model, device_ids=[self.local_rank])
        elif self.model_type == "cat":
            self.model = GraphDiffClassifierCat(
                encoder,
                embedding_dim=args.embedding_dim,
                loss_type=self.loss_type,
                hidden_dim=getattr(args, "hidden_dim", 64),
                out_dim=getattr(args, "out_dim", 2),
            ).to(self.device)
            self.model = DDP(self.model, device_ids=[self.local_rank])
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=getattr(args, "weight_decay", 0.0)
        )

        # 学習率スケジューラの設定（scheduler_typeは既に設定済み）
        if self.scheduler_type == "reduce_on_plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='max',
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

        # 早期停止などの記録用
        self.best_auc     = 0.0
        self.best_model_state  = None
        self.early_stop_counter = 0
        self.train_losses = []
        self.val_losses   = []
        self.val_aucs = []
        self.learning_rates = []  # 学習率の履歴を記録

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
        # バイナリラベルを直接使用
        y_class = y.long().squeeze(-1)

        if self.loss_type == "pair":
            # pair では ab ヘッドだけ
            return F.cross_entropy(pred_ab, y_class)
        elif self.loss_type == "pair_bi":
            # pair では ab ヘッドだけ
            return (F.cross_entropy(pred_ab, y_class) + F.cross_entropy(pred_ba, y_class)) / 2.0
        elif self.loss_type == "pair_bi_sym":
            return (F.cross_entropy(pred_ab, y_class) + F.cross_entropy(pred_ba, y_class)) / 2.0 + F.mse_loss(pred_ab, pred_ba)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

    def save_checkpoint(self, epoch):
        """トレーニング状態をチェックポイントとして保存"""
        if self.local_rank == 0:  # マスターランクのみ保存
            checkpoint = {
                'epoch': epoch + 1,  # 次回開始するエポック
                'model_state_dict': self.model.module.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_auc': self.best_auc,
                'best_model_state': self.best_model_state,
                'early_stop_counter': self.early_stop_counter,
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'val_aucs': self.val_aucs,
                'learning_rates': self.learning_rates
            }

            # スケジューラの状態も保存
            if self.scheduler is not None:
                checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

            torch.save(checkpoint, os.path.join(self.log_dir, "checkpoint.pt"))
            self.log_message(f"Checkpoint saved at epoch {epoch}")

    def load_checkpoint(self, checkpoint_path):
        """チェックポイントから学習状態を復元"""
        self.log_message(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # モデルの状態を復元
        self.model.module.load_state_dict(checkpoint['model_state_dict'])

        # オプティマイザの状態を復元
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # スケジューラの状態を復元
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # トレーニング状態を復元
        self.start_epoch = checkpoint['epoch']
        self.best_auc = checkpoint['best_auc']
        self.best_model_state = checkpoint['best_model_state']
        self.early_stop_counter = checkpoint['early_stop_counter']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.val_aucs = checkpoint['val_aucs']
        self.learning_rates = checkpoint['learning_rates']

        self.log_message(f"Resuming from epoch {self.start_epoch} with best AUC: {self.best_auc:.4f}")

    def train(self):
        for epoch in range(self.start_epoch, self.args.epochs):
            self.model.train()
            total_loss = 0.0
            self.train_loader.sampler.set_epoch(epoch)  # エポックごとにシャッフルパターンを変更

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
                val_auc = val_metrics["auc_roc"]
                val_loss = val_metrics["avg_loss"]

                self.train_losses.append(train_loss)
                self.val_losses.append(val_loss)
                self.val_aucs.append(val_auc)
                # 現在時刻を取得して表示
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.log_message(f"[{current_time}] Epoch {epoch:03d}: TrainLoss={train_loss:.4f}  "
                      f"ValLoss={val_loss:.4f}  ValAUC={val_auc:.4f}")

                # 学習率スケジューラの更新
                if self.scheduler is not None:
                    if self.scheduler_type == "reduce_on_plateau":
                        self.scheduler.step(val_auc)  # AUCに基づいて学習率を調整
                    else:
                        self.scheduler.step()  # エポックごとに学習率を調整

                # early stopping
                if val_auc > self.best_auc:
                    self.best_auc    = val_auc
                    self.best_model_state = copy.deepcopy(self.model.module.state_dict())  # DDP使用時はmoduleを取得
                    self.early_stop_counter = 0
                    # save model
                    self.log_message(f"Best model {epoch}")
                    self.save_model(f"model_best.pt")
                elif val_auc == self.best_auc:
                    # AUCが同じ場合は損失で判断
                    if val_loss < self.val_losses[self.val_aucs.index(self.best_auc)]:
                        self.best_model_state = copy.deepcopy(self.model.module.state_dict())  # DDP使用時はmoduleを取得
                        self.early_stop_counter = 0
                        # save model
                        self.log_message(f"Best model {epoch}")
                        self.save_model(f"model_best.pt")
                    else:
                        self.early_stop_counter += 1
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
            # ベストモデルで保存＆プロット
            self.model.module.load_state_dict(self.best_model_state)  # DDP使用時はmoduleを取得
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
        y_prob_list_ab, y_prob_list_ba = [], []  # 確率値も保存

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

                # バイナリラベルを直接使用
                y_true_batch = y.long().squeeze(-1).cpu()

                # 予測クラスの取得
                y_pred_batch_ab = torch.argmax(pred_ab, dim=1).cpu()
                y_pred_batch_ba = None if pred_ba is None else torch.argmax(pred_ba, dim=1).cpu()

                # 確率値も保存
                y_prob_batch_ab = F.softmax(pred_ab, dim=1).cpu()
                y_prob_batch_ba = None if pred_ba is None else F.softmax(pred_ba, dim=1).cpu()

                y_true_list.append(y_true_batch)
                y_pred_list_ab.append(y_pred_batch_ab)
                if pred_ba is not None:
                    y_pred_list_ba.append(y_pred_batch_ba)
                    y_prob_list_ba.append(y_prob_batch_ba)
                y_prob_list_ab.append(y_prob_batch_ab)

        # 全バッチを結合
        y_true = torch.cat(y_true_list, dim=0)
        y_pred_ab = torch.cat(y_pred_list_ab, dim=0)
        y_prob_ab = torch.cat(y_prob_list_ab, dim=0)

        if len(y_pred_list_ba) > 0:
            y_pred_ba = torch.cat(y_pred_list_ba, dim=0)
            y_prob_ba = torch.cat(y_prob_list_ba, dim=0)
        else:
            y_pred_ba = None
            y_prob_ba = None

        # 分類指標 abのみ
        auc_roc = roc_auc_score(y_true.numpy(), y_prob_ab.numpy()[:, 1])
        accuracy = accuracy_score(y_true.numpy(), y_pred_ab.numpy())
        precision = precision_score(y_true.numpy(), y_pred_ab.numpy(), zero_division=0)
        recall = recall_score(y_true.numpy(), y_pred_ab.numpy(), zero_division=0)
        f1 = f1_score(y_true.numpy(), y_pred_ab.numpy(), zero_division=0)
        avg_loss = loss_sum / n_graphs

        if return_preds:
            # 予測値も返す場合
            return {
                "auc_roc": auc_roc,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "avg_loss": avg_loss,
                "confusion_matrix": confusion_matrix(y_true.numpy(), y_pred_ab.numpy()),
                "y_true": y_true,
                "y_pred_ab": y_pred_ab,
                "y_pred_ba": y_pred_ba,
                "y_prob_ab": y_prob_ab,
                "y_prob_ba": y_prob_ba
            }

        # 評価指標のみ返す場合
        return {
            "auc_roc": auc_roc,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "avg_loss": avg_loss,
            "confusion_matrix": confusion_matrix(y_true.numpy(), y_pred_ab.numpy())
        }

    def test(self):
        # マスターランクのみテスト実行
        if self.local_rank != 0:
            return None

        # ベストモデルを使用
        self.model.module.load_state_dict(self.best_model_state)  # DDP使用時はmoduleを取得
        self.model.eval()

        # evaluate から分類指標を取得
        results = self.evaluate(self.test_loader, return_preds=True)

        # テストデータの予測値を保存
        self.save_predictions(results, "test_predictions.npz")

        # 指標をまとめてプリント
        self.log_message(f"Test AUC-ROC={results['auc_roc']:.4f}  Loss={results['avg_loss']:.4f}  "
              f"Precision={results['precision']:.4f}  Recall={results['recall']:.4f}  "
              f"F1={results['f1']:.4f}")
        self.log_message(f"Confusion Matrix:\n{results['confusion_matrix']}")
        self.log_message(f"loss_type={self.loss_type}")

        # 混同行列のプロット
        self.plot_confusion_matrix(results['confusion_matrix'])

        # 必要に応じて分類指標も返却
        return results['auc_roc'], results['avg_loss'], results['precision'], results['recall'], results['f1']

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
            y_pred_ba=results["y_pred_ba"].numpy() if results["y_pred_ba"] is not None else np.array([]),
            y_prob_ab=results["y_prob_ab"].numpy(),
            y_prob_ba=results["y_prob_ba"].numpy() if results["y_prob_ba"] is not None else np.array([])
        )
        self.log_message(f"Predictions saved → {save_path}")

    def save_plots(self):
        plt.figure(figsize=(8,5))
        plt.plot(self.train_losses, label="Train Loss")
        plt.plot(self.val_losses, label="Val Loss")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "loss_curve.png")); plt.close()

        plt.figure(figsize=(8,5))
        plt.plot(self.val_aucs, label="Val AUC-ROC", color="green")
        plt.xlabel("Epoch"); plt.ylabel("AUC-ROC"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "auc_curve.png")); plt.close()

        # 学習率の推移をプロット
        plt.figure(figsize=(8,5))
        plt.plot(self.learning_rates, label="Learning Rate", color="purple")
        plt.xlabel("Epoch"); plt.ylabel("Learning Rate"); plt.legend(); plt.tight_layout()
        plt.yscale('log')  # 対数スケールで表示
        plt.savefig(os.path.join(self.log_dir, "lr_curve.png")); plt.close()

    def plot_confusion_matrix(self, conf_matrix):
        plt.figure(figsize=(6,6))
        plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()

        classes = ["Class 0", "Class 1"]
        tick_marks = range(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)

        # 数値を表示
        thresh = conf_matrix.max() / 2.
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                plt.text(j, i, format(conf_matrix[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if conf_matrix[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(os.path.join(self.log_dir, "confusion_matrix.png"))
        plt.close()

    def __del__(self):
        """デストラクタでログファイルを閉じる"""
        if hasattr(self, 'log_file') and self.local_rank == 0:
            self.log_file.close()

def parse_args():
    # argparseを使用してコマンドライン引数を処理
    import argparse

    parser = argparse.ArgumentParser(description='Graph Difference Classifier Training')

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
    parser.add_argument('--patience', type=int, default=30, help='Patience for early stopping')

    # Learning rate scheduler
    parser.add_argument('--scheduler', type=str, default='cosine_annealing',
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
    parser.add_argument('--loss_type', type=str, default='pair_bi', help='Loss type: pair, pair_bi')
    parser.add_argument('--model_type', type=str, default='diff', help='Model type: diff, cat, product', choices=['diff', 'cat', 'product'])
    parser.add_argument('--inference-only', action='store_true', help='Inference only')

    return parser.parse_args()
def main(args):
    # Set random seed for reproducibility
    random_seed(args.seed)

    # Enable anomaly detection for debugging
    torch.autograd.set_detect_anomaly(True)

    # ==== 1) 分散初期化 ====
    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ["LOCAL_RANK"])            # torchrun が自動セット
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
