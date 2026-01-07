#!/usr/bin/env python3
"""
ファインチューニング結果の分析と可視化スクリプト

Author: Generated for transfer learning experiments
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from itertools import product
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 日本語フォントの設定は行わない（英語のみ使用）
plt.rcParams['font.family'] = 'DejaVu Sans'


def root_mean_squared_error(y_true, y_pred):
    """RMSEを計算"""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def load_results(results_dir: Path, targets: list, data_modes: list, 
                 model_modes: list, folds: list = range(5)) -> pd.DataFrame:
    """
    全実験結果を読み込む
    
    Args:
        results_dir: 結果ディレクトリ
        targets: ターゲットリスト
        data_modes: データモードリスト（All, N100）
        model_modes: モデルモードリスト（full, ft）
        folds: フォールドリスト
        
    Returns:
        結果DataFrame
    """
    results = []
    missing = 0
    
    for target, data_mode, model_mode, fold in product(targets, data_modes, model_modes, folds):
        pred_file = results_dir / data_mode / target / f"cv{fold}" / model_mode / "test_predictions.npz"
        
        if not pred_file.exists():
            missing += 1
            continue
        
        data = np.load(pred_file)
        labels = data["labels"].reshape(-1)
        predictions = data["predictions"].reshape(-1)
        
        results.append({
            "target": target,
            "data_mode": data_mode,
            "model_mode": model_mode,
            "fold": fold,
            "r2_score": r2_score(labels, predictions),
            "rmse": root_mean_squared_error(labels, predictions),
            "mae": mean_absolute_error(labels, predictions),
            "mse": mean_squared_error(labels, predictions),
            "n_samples": len(labels)
        })
    
    if missing > 0:
        print(f"Warning: {missing} result files not found")
    
    return pd.DataFrame(results)


def plot_comparison(df: pd.DataFrame, output_dir: Path, figure_format: str = "pdf"):
    """
    R²とRMSEの比較プロットを作成（論文Figure用）
    """
    # モデル名のマッピング
    model_names = {
        "ft": "Pretrained",
        "full": "Fully Trained"
    }
    df = df.copy()
    df["Training"] = df["model_mode"].map(model_names)
    
    # ターゲット名の調整
    df["target"] = df["target"].replace("thrombin", "Thrombin")
    
    # データモードのラベル
    data2label = {"All": "All", "N50": "$N=50$", "N100": "$N=100$"}
    
    font_size = 15
    
    # 2x2のプロット（R²とRMSE × AllとN100）
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey='row')
    
    data_modes = ["All", "N100"]
    
    # R²のプロット（1行目）
    for i, data_mode in enumerate(data_modes):
        subset = df[df["data_mode"] == data_mode]
        sns.barplot(
            data=subset, 
            x="target", 
            hue="Training", 
            y="r2_score", 
            palette="Set2", 
            ax=axes[0, i],
            errorbar="sd"
        )
        axes[0, i].set_title(data2label[data_mode], fontsize=font_size)
        axes[0, i].set_xlabel("Target", fontsize=font_size)
        if i == 0:
            axes[0, i].set_ylabel("$R^2$", fontsize=font_size)
        else:
            axes[0, i].set_ylabel("", fontsize=font_size)
        axes[0, i].tick_params(axis='both', labelsize=font_size-2)
        axes[0, i].get_legend().remove()
    
    # RMSEのプロット（2行目）
    for i, data_mode in enumerate(data_modes):
        subset = df[df["data_mode"] == data_mode]
        sns.barplot(
            data=subset, 
            x="target", 
            hue="Training", 
            y="rmse", 
            palette="Set2", 
            ax=axes[1, i],
            errorbar="sd"
        )
        axes[1, i].set_xlabel("Target", fontsize=font_size)
        if i == 0:
            axes[1, i].set_ylabel("RMSE", fontsize=font_size)
        else:
            axes[1, i].set_ylabel("", fontsize=font_size)
        axes[1, i].tick_params(axis='both', labelsize=font_size-2)
        axes[1, i].get_legend().remove()
    
    # 凡例
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.05), 
               ncol=len(labels), fontsize=font_size)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    output_file = output_dir / f"transfer_learning_comparison.{figure_format}"
    plt.savefig(output_file, bbox_inches="tight", dpi=300)
    plt.close()
    
    print(f"Saved: {output_file}")
    
    return output_file


def plot_r2_only(df: pd.DataFrame, output_dir: Path, figure_format: str = "pdf"):
    """
    R²のみの比較プロット（横並び）
    """
    model_names = {
        "ft": "Pretrained",
        "full": "Fully Trained"
    }
    df = df.copy()
    df["Training"] = df["model_mode"].map(model_names)
    df["target"] = df["target"].replace("thrombin", "Thrombin")
    
    data2label = {"All": "All", "N50": "$N=50$", "N100": "$N=100$"}
    font_size = 15
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    
    data_modes = ["All", "N100"]
    
    for i, data_mode in enumerate(data_modes):
        subset = df[df["data_mode"] == data_mode]
        sns.barplot(
            data=subset, 
            x="target", 
            hue="Training", 
            y="r2_score", 
            palette="Set2", 
            ax=axes[i],
            errorbar="sd"
        )
        axes[i].set_title(data2label[data_mode], fontsize=font_size)
        axes[i].set_xlabel("Target", fontsize=font_size)
        if i == 0:
            axes[i].set_ylabel("$R^2$", fontsize=font_size)
        else:
            axes[i].set_ylabel("", fontsize=font_size)
        axes[i].tick_params(axis='both', labelsize=font_size)
        axes[i].get_legend().remove()
    
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.05), 
               ncol=len(labels), fontsize=font_size)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    
    output_file = output_dir / f"transfer_learning_r2.{figure_format}"
    plt.savefig(output_file, bbox_inches="tight", dpi=300)
    plt.close()
    
    print(f"Saved: {output_file}")


def plot_prediction_scatter(results_dir: Path, output_dir: Path, targets: list, 
                           data_modes: list, model_modes: list, figure_format: str = "png"):
    """
    予測値と正解値の散布図を作成
    """
    model_names = {
        "ft": "Pretrained",
        "full": "Fully Trained"
    }
    
    for data_mode in data_modes:
        fig, axes = plt.subplots(2, len(targets), figsize=(4*len(targets), 8), 
                                sharex=True, sharey=True)
        
        if len(targets) == 1:
            axes = axes.reshape(-1, 1)
        
        for j, target in enumerate(targets):
            for i, model_mode in enumerate(model_modes):
                # 全foldの予測を集める
                all_preds = []
                all_labels = []
                
                for fold in range(5):
                    pred_file = results_dir / data_mode / target / f"cv{fold}" / model_mode / "test_predictions.npz"
                    if pred_file.exists():
                        data = np.load(pred_file)
                        all_preds.extend(data["predictions"].flatten())
                        all_labels.extend(data["labels"].flatten())
                
                if len(all_preds) > 0:
                    all_preds = np.array(all_preds)
                    all_labels = np.array(all_labels)
                    
                    # 散布図
                    axes[i, j].scatter(all_labels, all_preds, alpha=0.3, s=1)
                    
                    # 理想線（y=x）
                    min_val = min(all_labels.min(), all_preds.min())
                    max_val = max(all_labels.max(), all_preds.max())
                    axes[i, j].plot([min_val, max_val], [min_val, max_val], 
                                   'r--', linewidth=1, label='Perfect')
                    
                    # R²を計算して表示
                    r2 = r2_score(all_labels, all_preds)
                    rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
                    
                    axes[i, j].text(0.05, 0.95, f'$R^2$ = {r2:.3f}\nRMSE = {rmse:.3f}',
                                   transform=axes[i, j].transAxes,
                                   verticalalignment='top',
                                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                    # タイトルとラベル
                    if i == 0:
                        axes[i, j].set_title(target, fontsize=12, fontweight='bold')
                    if j == 0:
                        axes[i, j].set_ylabel(f'{model_names[model_mode]}\nPredicted', fontsize=10)
                    if i == len(model_modes) - 1:
                        axes[i, j].set_xlabel('True Value', fontsize=10)
                    
                    axes[i, j].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = output_dir / f"prediction_scatter_{data_mode}.{figure_format}"
        plt.savefig(output_file, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"Saved: {output_file}")


def plot_distribution_comparison(results_dir: Path, output_dir: Path, targets: list,
                                 data_modes: list, model_modes: list, figure_format: str = "png"):
    """
    予測値の分布と正解値の分布を比較
    """
    model_names = {
        "ft": "Pretrained",
        "full": "Fully Trained"
    }
    
    for data_mode in data_modes:
        fig, axes = plt.subplots(len(targets), 2, figsize=(12, 4*len(targets)))
        
        if len(targets) == 1:
            axes = axes.reshape(1, -1)
        
        for i, target in enumerate(targets):
            for j, model_mode in enumerate(model_modes):
                # 全foldの予測を集める
                all_preds = []
                all_labels = []
                
                for fold in range(5):
                    pred_file = results_dir / data_mode / target / f"cv{fold}" / model_mode / "test_predictions.npz"
                    if pred_file.exists():
                        data = np.load(pred_file)
                        all_preds.extend(data["predictions"].flatten())
                        all_labels.extend(data["labels"].flatten())
                
                if len(all_preds) > 0:
                    all_preds = np.array(all_preds)
                    all_labels = np.array(all_labels)
                    
                    # ヒストグラム
                    axes[i, j].hist(all_labels, bins=50, alpha=0.5, label='True', density=True)
                    axes[i, j].hist(all_preds, bins=50, alpha=0.5, label='Predicted', density=True)
                    
                    # 統計情報
                    axes[i, j].axvline(all_labels.mean(), color='blue', linestyle='--', 
                                      linewidth=2, label=f'True μ={all_labels.mean():.2f}')
                    axes[i, j].axvline(all_preds.mean(), color='orange', linestyle='--', 
                                      linewidth=2, label=f'Pred μ={all_preds.mean():.2f}')
                    
                    axes[i, j].set_title(f'{target} - {model_names[model_mode]} ({data_mode})', 
                                        fontsize=12, fontweight='bold')
                    axes[i, j].set_xlabel('Value', fontsize=10)
                    axes[i, j].set_ylabel('Density', fontsize=10)
                    axes[i, j].legend(fontsize=8)
                    axes[i, j].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = output_dir / f"distribution_comparison_{data_mode}.{figure_format}"
        plt.savefig(output_file, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"Saved: {output_file}")


def create_summary_table(df: pd.DataFrame, output_dir: Path):
    """
    サマリーテーブルを作成
    """
    # 各条件での平均と標準偏差
    summary = df.groupby(["target", "data_mode", "model_mode"]).agg({
        "r2_score": ["mean", "std"],
        "rmse": ["mean", "std"],
        "mae": ["mean", "std"],
        "n_samples": "mean"
    }).round(4)
    
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()
    
    # 保存
    output_file = output_dir / "transfer_learning_summary.csv"
    summary.to_csv(output_file, index=False)
    print(f"Saved: {output_file}")
    
    # Pretrained vs Fully Trainedの比較
    comparison = df.pivot_table(
        values="r2_score",
        index=["target", "data_mode", "fold"],
        columns="model_mode"
    ).reset_index()
    
    if "ft" in comparison.columns and "full" in comparison.columns:
        comparison["improvement"] = comparison["ft"] - comparison["full"]
        
        improvement_summary = comparison.groupby(["target", "data_mode"])["improvement"].agg(["mean", "std"])
        improvement_file = output_dir / "transfer_learning_improvement.csv"
        improvement_summary.to_csv(improvement_file)
        print(f"Saved: {improvement_file}")
    
    return summary


def main():
    # デフォルトパスの設定
    script_dir = Path(__file__).parent
    default_results_dir = script_dir / "results"
    default_output_dir = script_dir / "figures"
    
    parser = argparse.ArgumentParser(description="Analyze finetuning results")
    parser.add_argument("--results_dir", type=str,
                        default=str(default_results_dir),
                        help="Results directory")
    parser.add_argument("--output_dir", type=str,
                        default=str(default_output_dir),
                        help="Output directory for figures")
    parser.add_argument("--targets", type=str, nargs="+",
                        default=["BACE", "JNK1", "P38", "thrombin", "PTP1B", "CDK2"],
                        help="Target names")
    parser.add_argument("--data_modes", type=str, nargs="+",
                        default=["All", "N100"],
                        help="Data modes")
    parser.add_argument("--model_modes", type=str, nargs="+",
                        default=["full", "ft"],
                        help="Model modes")
    parser.add_argument("--format", type=str, default="png",
                        help="Figure format (pdf, png, svg)")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*50)
    print("Finetuning Results Analysis")
    print("="*50)
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Targets: {args.targets}")
    print(f"Data modes: {args.data_modes}")
    print(f"Model modes: {args.model_modes}")
    print("="*50)
    
    # 結果を読み込み
    df = load_results(
        results_dir,
        args.targets,
        args.data_modes,
        args.model_modes
    )
    
    if len(df) == 0:
        print("No results found!")
        return
    
    print(f"\nLoaded {len(df)} results")
    print(f"Targets: {df['target'].unique().tolist()}")
    print(f"Data modes: {df['data_mode'].unique().tolist()}")
    print(f"Model modes: {df['model_mode'].unique().tolist()}")
    
    # 結果DataFrameを保存
    df.to_csv(output_dir / "all_results.csv", index=False)
    
    # サマリーテーブル作成
    summary = create_summary_table(df, output_dir)
    
    # プロット作成
    plot_comparison(df, output_dir, args.format)
    plot_r2_only(df, output_dir, args.format)
    
    # 散布図と分布の比較を作成
    plot_prediction_scatter(results_dir, output_dir, args.targets, 
                           args.data_modes, args.model_modes, args.format)
    plot_distribution_comparison(results_dir, output_dir, args.targets,
                                args.data_modes, args.model_modes, args.format)
    
    print("\n" + "="*50)
    print("Analysis completed!")
    print("="*50)


if __name__ == "__main__":
    main()
