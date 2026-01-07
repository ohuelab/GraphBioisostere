import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import sys
import argparse
import copy
import json
import random
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix, mean_squared_error, r2_score
)
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from pathlib import Path

# pro_GNNディレクトリをパスに追加
script_dir = Path(__file__).parent
pro_gnn_dir = script_dir.parent
sys.path.insert(0, str(pro_gnn_dir))

from utils.loader import load_encoder
from train_utils import pair_collate_fn, random_seed

def make_head(in_dim: int, hidden: int, out_dim: int = 1) -> nn.Sequential:
    """2-layer MLP ヘッドを生成"""
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden // 2),
        nn.ReLU(),
        nn.Linear(hidden // 2, out_dim)
    )

class GraphDiffRegressor(nn.Module):
    def __init__(self, encoder: nn.Module, embedding_dim: int, hidden_dim: int = 64, out_dim: int = 1, merge_method: str = "diff"):
        super().__init__()
        self.encoder   = encoder
        self.merge_method = merge_method

        # catの場合は入力次元を2倍にする
        input_dim = embedding_dim * 2 if merge_method == "cat" else embedding_dim
        self.mlp = make_head(input_dim, hidden_dim, out_dim)

    def forward(self, data_a, data_b):
        fa = self.encoder(data_a)
        fb = self.encoder(data_b)

        # デフォルトは None にしておき、存在するヘッドだけ計算
        if self.merge_method == "diff":
            diff_feature = fa - fb
            pred = self.mlp(diff_feature)
        elif self.merge_method == "product":
            product_feature = fa * fb
            pred = self.mlp(product_feature)
        elif self.merge_method == "cat":
            cat_feature = torch.cat([fa, fb], dim=-1)
            pred = self.mlp(cat_feature)
        else:
            raise ValueError(f"Unknown merge_method: {self.merge_method}")

        return pred


class FinetuneTrainer:
    def __init__(self, args, train_loader=None, val_loader=None, test_loader=None, pretrain_model=None):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        # ログファイル
        self.log_file = open(os.path.join(args.output_dir, "log.txt"), "w")

        # ログディレクトリ作成
        self.log_dir = args.output_dir
        os.makedirs(self.log_dir, exist_ok=True)

        # データローダー
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # 訓練データからデータセットを取得
        self.train_dataset = train_loader.dataset if train_loader else None

        # モデル
        encoder = load_encoder(args).to(self.device)

        self.model = GraphDiffRegressor(
            encoder,
            embedding_dim=args.embedding_dim,
            hidden_dim=getattr(args, "hidden_dim", 64),
            out_dim=getattr(args, "out_dim", 1),
            merge_method=getattr(args, "model_type", "diff"),
        ).to(self.device)

        self.model_tmp = GraphDiffRegressor(
            encoder,
            embedding_dim=args.embedding_dim,
            hidden_dim=getattr(args, "hidden_dim", 64),
            out_dim=2,
            merge_method=getattr(args, "model_type", "diff"),
        )

        # 事前学習済みモデルがある場合は読み込む
        self.pretrain_model = pretrain_model
        if self.pretrain_model is not None:
            self.log_message(f"Loading pretrained model from {pretrain_model}")
            self.model_tmp.load_state_dict(torch.load(pretrain_model, map_location="cpu"))
            self.model.encoder.load_state_dict(self.model_tmp.encoder.state_dict())


        # エンコーダは固定する
        if self.args.freeze_encoder:
            for param in encoder.parameters():
                param.requires_grad = False
            # headはリセット?

        # オプティマイザ
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=getattr(args, "weight_decay", 0.0)
        )
        # スケジューラは使用しない
        self.scheduler = None

        # 最良モデルの保存用
        self.best_val_mse = float('inf')
        self.best_epoch = 0
        self.best_model_state = None

        # 記録用
        self.train_losses = []
        self.val_losses = []
        self.val_mses = []
        self.val_r2s = []
        self.learning_rates = []

        # save args
        with open(os.path.join(args.output_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=4)

    def log_message(self, message):
        """ログメッセージを出力"""
        print(message)
        if self.log_file:
            self.log_file.write(message + "\n")
            self.log_file.flush()

    def train(self):
        """学習を実行"""
        if len(self.train_dataset) < 100:
            self.log_message("データセットが100件未満のためMonte Carlo Cross-Validationを実行")
            return self.monte_carlo_cross_validation()
        else:
            self.log_message("通常の学習を実行")
            return self.train_with_early_stopping()

    def train_with_early_stopping(self):
        """通常の学習（Early Stopping付き）"""
        patience = getattr(self.args, "patience", 10)
        counter = 0

        for epoch in range(self.args.epochs):
            train_loss = self.train_one_epoch(self.train_loader)
            val_metrics = self.evaluate(self.val_loader)
            val_loss = val_metrics['loss']
            val_mse = val_metrics['mse']
            val_r2 = val_metrics['r2']

            # 記録
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_mses.append(val_mse)
            self.val_r2s.append(val_r2)
            self.learning_rates.append(self.optimizer.param_groups[0]['lr'])

            self.log_message(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val MSE={val_mse:.4f}, Val R²={val_r2:.4f}")

            # 最良モデルの保存 (MSEが最小のモデル)
            if val_mse < self.best_val_mse:
                self.best_val_mse = val_mse
                self.best_epoch = epoch
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                self.save_model("model_best.pt")
                counter = 0
            else:
                counter += 1

            # Early stopping
            if counter >= patience:
                self.log_message(f"Early stopping at epoch {epoch}")
                break

            # チェックポイントを保存
            if epoch % 10 == 0:
                self.save_checkpoint(epoch)

        # ベストモデルで保存＆プロット
        self.model.load_state_dict(self.best_model_state)
        self.save_model("model_best.pt")
        self.save_plots()

        # 検証データセットの予測値を保存
        val_results = self.evaluate(self.val_loader, return_preds=True)
        self.save_predictions(val_results, "valid_predictions.npz")

        self.log_message(f"Best model at epoch {self.best_epoch} with MSE {self.best_val_mse:.4f}")
        return self.best_val_mse

    def monte_carlo_cross_validation(self):
        """Monte Carlo Cross-Validation（5-Fold）"""
        fold_mses = []
        best_epoch_sum = 0

        # 5-Fold CV
        for fold in range(5):
            self.log_message(f"Fold {fold+1}/5")

            # モデルの初期化
            self.reset_model()

            # データセットの分割
            train_size = int(0.8 * len(self.train_dataset))
            val_size = len(self.train_dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                self.train_dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(fold)  # 再現性のため
            )

            # データローダーの作成
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.args.batch_size,
                shuffle=True,
                collate_fn=pair_collate_fn
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.args.batch_size,
                shuffle=False,
                collate_fn=pair_collate_fn
            )

            # 各エポックでの検証
            fold_best_mse = float('inf')
            fold_best_epoch = 0
            fold_train_losses = []
            fold_val_losses = []
            fold_val_mses = []
            fold_val_r2s = []

            for epoch in range(self.args.epochs):
                train_loss = self.train_one_epoch(train_loader)
                val_metrics = self.evaluate(val_loader)
                val_loss = val_metrics['loss']
                val_mse = val_metrics['mse']
                val_r2 = val_metrics['r2']

                # 記録
                fold_train_losses.append(train_loss)
                fold_val_losses.append(val_loss)
                fold_val_mses.append(val_mse)
                fold_val_r2s.append(val_r2)

                self.log_message(f"Fold {fold+1}, Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val MSE={val_mse:.4f}, Val R²={val_r2:.4f}")

                if val_mse < fold_best_mse:
                    fold_best_mse = val_mse
                    fold_best_epoch = epoch

            fold_mses.append(fold_best_mse)
            best_epoch_sum += fold_best_epoch
            self.log_message(f"Fold {fold+1} best MSE: {fold_best_mse:.4f} at epoch {fold_best_epoch}")

            # 各フォールドの学習曲線をプロット
            # self.plot_learning_curves(fold_train_losses, fold_val_losses, fold_val_mses, fold_val_r2s, f"fold_{fold+1}")

        # 平均MSEと最良エポックの計算
        avg_mse = sum(fold_mses) / len(fold_mses)
        avg_best_epoch = best_epoch_sum // 5

        self.log_message(f"Cross-validation complete. Average MSE: {avg_mse:.4f}")
        self.log_message(f"Average best epoch: {avg_best_epoch}")

        # 最終モデルの学習（全データで学習）
        self.reset_model()
        self.log_message(f"Training final model for {avg_best_epoch+1} epochs on all data")

        for epoch in range(avg_best_epoch + 1):
            train_loss = self.train_one_epoch(self.train_loader)
            self.train_losses.append(train_loss)
            self.learning_rates.append(self.optimizer.param_groups[0]['lr'])
            self.log_message(f"Final model, Epoch {epoch}: Train Loss={train_loss:.4f}")

        # 最終モデルを保存
        self.best_model_state = copy.deepcopy(self.model.state_dict())
        self.save_model("model_best.pt")

        return avg_mse
    def reset_model(self):
        """モデルを初期化"""
        encoder = load_encoder(self.args).to(self.device)
        # エンコーダは固定する

        self.model = GraphDiffRegressor(
            encoder,
            embedding_dim=self.args.embedding_dim,
            hidden_dim=getattr(self.args, "hidden_dim", 64),
            out_dim=getattr(self.args, "out_dim", 1),
            merge_method=getattr(self.args, "model_type", "diff"),
        ).to(self.device)
        # 事前学習済みモデルがある場合は読み込む
        if self.pretrain_model is not None:
            # self.model.load_state_dict(torch.load(self.pretrain_model, map_location=self.device))
            self.model_tmp.load_state_dict(torch.load(self.pretrain_model, map_location="cpu"))
            self.model.encoder.load_state_dict(self.model_tmp.encoder.state_dict())

        if self.args.freeze_encoder:
            for param in self.model.encoder.parameters():
                param.requires_grad = False

        # オプティマイザを初期化
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=getattr(self.args, "weight_decay", 0.0)
        )

    def train_one_epoch(self, train_loader):
        """1エポック分の学習"""
        self.model.train()
        total_loss = 0

        for batch in train_loader:
            g1, g2, y = (x.to(self.device) for x in batch)

            self.optimizer.zero_grad()

            # 順伝播
            pred = self.model(g1, g2)

            # 損失計算 - 入力と出力のサイズを合わせる
            # pred.view(-1)ではなく、pred.squeeze()を使用してサイズを合わせる
            # または、yのサイズをpredに合わせる
            if pred.dim() > 1 and pred.size(1) == 1:
                pred = pred.squeeze(1)  # [batch_size, 1] -> [batch_size]
            elif y.dim() == 1 and pred.dim() > 1:
                y = y.view(-1, pred.size(1))  # [batch_size] -> [batch_size, out_dim]

            loss = nn.MSELoss()(pred, y.squeeze().float())
            # 逆伝播
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def evaluate(self, data_loader, return_preds=False):
        """評価"""
        self.model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0

        with torch.no_grad():
            for batch in data_loader:
                g1, g2, y = (x.to(self.device) for x in batch)

                # 順伝播
                pred = self.model(g1, g2)

                # 損失計算
                loss = nn.MSELoss()(pred.view(-1), y.squeeze().float())

                total_loss += loss.item()
                all_preds.append(pred.view(-1).cpu())
                all_labels.append(y.cpu())

        # 予測値とラベルの結合
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        # 評価指標の計算
        mse = mean_squared_error(all_labels, all_preds)
        r2 = r2_score(all_labels, all_preds)

        metrics = {
            'loss': total_loss / len(data_loader),
            'mse': mse,
            'rmse': np.sqrt(mse),
            'r2': r2
        }

        if return_preds:
            metrics['predictions'] = all_preds
            metrics['labels'] = all_labels

        return metrics

    def test(self):
        """テストデータでの評価"""
        self.log_message("Testing model...")

        # 最良モデルの読み込み
        model_path = os.path.join(self.args.output_dir, "model_best.pt")
        self.model.load_state_dict(torch.load(model_path))

        # テスト
        test_metrics = self.evaluate(self.test_loader, return_preds=True)

        self.log_message(f"Test results: Loss={test_metrics['loss']:.4f}, MSE={test_metrics['mse']:.4f}, "
                         f"RMSE={test_metrics['rmse']:.4f}, R²={test_metrics['r2']:.4f}")

        # 予測値と実測値の散布図プロット
        self.plot_prediction_scatter(test_metrics['predictions'], test_metrics['labels'])

        # テスト予測の保存
        self.save_predictions(test_metrics, "test_predictions.npz")

        return test_metrics

    def save_model(self, filename):
        """モデルの保存"""
        model_path = os.path.join(self.args.output_dir, filename)
        torch.save(self.model.state_dict(), model_path)
        self.log_message(f"Model saved to {model_path}")

    def save_checkpoint(self, epoch):
        """チェックポイントの保存"""
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_mse': self.best_val_mse,
            'best_epoch': self.best_epoch,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_mses': self.val_mses,
            'val_r2s': self.val_r2s,
            'learning_rates': self.learning_rates
        }
        checkpoint_path = os.path.join(self.args.output_dir, "checkpoint.pt")
        torch.save(checkpoint, checkpoint_path)
        self.log_message(f"Checkpoint saved at epoch {epoch}")

    def load_checkpoint(self, checkpoint_path):
        """チェックポイントの読み込み"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_mse = checkpoint['best_val_mse']
        self.best_epoch = checkpoint['best_epoch']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.val_mses = checkpoint['val_mses']
        self.val_r2s = checkpoint['val_r2s']
        self.learning_rates = checkpoint['learning_rates']
        start_epoch = checkpoint['epoch']
        self.log_message(f"Checkpoint loaded, resuming from epoch {start_epoch}")
        return start_epoch

    def save_plots(self):
        """学習曲線のプロット"""
        self.plot_learning_curves(self.train_losses, self.val_losses, self.val_mses, self.val_r2s)
        self.plot_lr_curve()

    def plot_learning_curves(self, train_losses, val_losses, val_mses, val_r2s=None, prefix=""):
        """学習曲線のプロット"""
        plt.figure(figsize=(15, 5))

        # 損失のプロット
        plt.subplot(1, 3, 1)
        plt.plot(train_losses, label='Train Loss')
        if val_losses:
            plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')

        # MSEのプロット
        plt.subplot(1, 3, 2)
        if val_mses:
            plt.plot(val_mses, label='Validation MSE', color='orange')
            plt.axhline(y=min(val_mses), color='r', linestyle='--',
                        label=f'Best MSE: {min(val_mses):.4f}')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.legend()
        plt.title('Validation MSE')

        # R²のプロット
        plt.subplot(1, 3, 3)
        if val_r2s:
            plt.plot(val_r2s, label='Validation R²', color='green')
            plt.axhline(y=max(val_r2s), color='r', linestyle='--',
                        label=f'Best R²: {max(val_r2s):.4f}')
        plt.xlabel('Epoch')
        plt.ylabel('R²')
        plt.legend()
        plt.title('Validation R²')

        plt.tight_layout()
        filename = f"{prefix}_learning_curves.png" if prefix else "learning_curves.png"
        plt.savefig(os.path.join(self.args.output_dir, filename))
        plt.close()

    def plot_lr_curve(self):
        """学習率の変化をプロット"""
        plt.figure(figsize=(10, 4))
        plt.plot(self.learning_rates, label='Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.legend()
        plt.title('Learning Rate Schedule')
        plt.yscale('log')  # 対数スケールで表示
        plt.tight_layout()
        plt.savefig(os.path.join(self.args.output_dir, "lr_curve.png"))
        plt.close()

    def plot_prediction_scatter(self, predictions, labels):
        """予測値と実測値の散布図"""
        plt.figure(figsize=(8, 8))
        plt.scatter(labels, predictions, alpha=0.5)

        # 理想線（y=x）を追加
        min_val = min(np.min(labels), np.min(predictions))
        max_val = max(np.max(labels), np.max(predictions))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')

        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.title('Prediction Scatter Plot')

        # R²値を表示
        r2 = r2_score(labels, predictions)
        mse = mean_squared_error(labels, predictions)
        rmse = np.sqrt(mse)
        plt.text(0.05, 0.95, f'R² = {r2:.4f}\nMSE = {mse:.4f}\nRMSE = {rmse:.4f}',
                 transform=plt.gca().transAxes, verticalalignment='top')

        plt.tight_layout()
        plt.savefig(os.path.join(self.args.output_dir, "prediction_scatter.png"))
        plt.close()

    def save_predictions(self, results, filename):
        """予測結果の保存"""
        if 'predictions' in results and 'labels' in results:
            np.savez(
                os.path.join(self.args.output_dir, filename),
                predictions=results['predictions'],
                labels=results['labels']
            )
            self.log_message(f"Predictions saved to {filename}")

    def __del__(self):
        """デストラクタでログファイルを閉じる"""
        if hasattr(self, 'log_file'):
            self.log_file.close()


def main(args):
    output_file = os.path.join(args.output_dir, "test_predictions.npz")
    if os.path.exists(output_file):
        print(f"✅ {output_file} already exists. Skipping this experiment.")
        sys.exit(0)  # 正常終了

    random_seed(args.seed)
    # データセットの読み込み
    dataset = torch.load(args.input_file, weights_only=False)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    # training_sizeが指定されている場合、データセットをサブサンプリング
    if args.training_size is not None:
        training_size = args.training_size #int(len(train_dataset) * args.training_size)
        rng = random.Random(args.seed)
        train_dataset_copy = train_dataset.copy() if hasattr(train_dataset, 'copy') else train_dataset[:]
        rng.shuffle(train_dataset_copy)
        train_dataset = train_dataset_copy[:training_size]

    # データローダーの作成
    if len(train_dataset) < 100:
        # Monte Carlo CVの場合、val_loaderは不要
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=pair_collate_fn
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=pair_collate_fn
        )
        val_loader = None
    else:
        # 通常の学習の場合、train_datasetを訓練用と検証用に分割
        val_split = int(len(train_dataset) * 0.8)
        val_dataset = train_dataset[val_split:]
        train_dataset = train_dataset[:val_split]

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=pair_collate_fn
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=pair_collate_fn
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=pair_collate_fn
        )

    # 事前学習済みモデルのパス
    pretrain_model_path = args.pretrain_model if hasattr(args, "pretrain_model") else None

    # ファインチューニングトレーナーの初期化
    finetune_trainer = FinetuneTrainer(
        args,
        train_loader,
        val_loader,
        test_loader,
        pretrain_model=pretrain_model_path
    )

    # チェックポイントから再開
    checkpoint_path = os.path.join(args.output_dir, "checkpoint.pt")
    start_epoch = 0
    if os.path.exists(checkpoint_path) and not args.inference_only:
        start_epoch = finetune_trainer.load_checkpoint(checkpoint_path)

    # 学習
    if not args.inference_only:
        best_mse = finetune_trainer.train()

    # テスト
    test_metrics = finetune_trainer.test()

    # ログファイルを閉じる
    finetune_trainer.log_file.close()

    return test_metrics

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--pretrain_model", type=str, required=False)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--inference_only", action="store_true")
    parser.add_argument('--node_in', type=int, default=30, help='Input dimension for node features')
    parser.add_argument('--edge_in', type=int, default=11, help='Input dimension for edge features')
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--out_dim", type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=3, help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--model_type", type=str, default="diff")
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--training_size", type=int, default=None)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    main(args)
