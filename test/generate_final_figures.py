#!/usr/bin/env python3
"""
統合予測結果分析・図生成スクリプト

予測結果データから最終的な図を出力します。

Author: Generated from notebooks integration
Date: 2025-10-26
"""

import os
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from pathlib import Path


class BioisosterePredictionAnalyzer:
    """バイオアイソスター予測結果の分析クラス"""
    
    def __init__(self, base_dir=None, results_dir=None, figures_dir=None, 
                 lgbm_predictions_dir=None, gnn_predictions_dirs=None, figure_format='png',
                 include_lgbm=False, comparison_mode='merge', cv_pkl_file=None):
        """
        Args:
            base_dir (str): ベースディレクトリのパス（指定しない場合は自動検出）
            results_dir (str): 結果CSVファイルの出力ディレクトリ（指定しない場合はデフォルト）
            figures_dir (str): 図の出力ディレクトリ（指定しない場合はデフォルト）
            lgbm_predictions_dir (str): LGBM予測結果のディレクトリ（指定しない場合はデフォルト）
            gnn_predictions_dirs (list): GNN予測結果のディレクトリリスト（指定しない場合はデフォルト）
            figure_format (str): 図の出力形式（'png', 'pdf', 'svg'など）
            include_lgbm (bool): LGBMモデルの結果を含めるかどうか（デフォルト: False）
            comparison_mode (str): 複数GNN結果の統合モード
                - 'merge': 同じモデルの結果を統合（デフォルト、既存の動作）
                - 'compare': 異なる条件（frag vs 通常など）を比較
            cv_pkl_file (str): Cross-validationデータのpklファイルパス（指定しない場合は自動検出）
        """
        # CVファイルのパスを保存
        self.cv_pkl_file = Path(cv_pkl_file) if cv_pkl_file else None
        # 現在のスクリプトの場所からベースディレクトリを自動検出
        if base_dir is None:
            current_script = Path(__file__).resolve()
            # スクリプトは /path/to/bioiso/test/ にあるので、1つ上の階層がbioisoディレクトリ
            self.bioiso_dir = current_script.parent.parent
            self.base_dir = self.bioiso_dir.parent
        else:
            self.base_dir = Path(base_dir)
            self.bioiso_dir = self.base_dir / "bioiso"
        self.notebooks_dir = self.bioiso_dir / "notebooks"
        
        # 予測結果のディレクトリを先に設定（出力ディレクトリ名の自動生成に必要）
        if gnn_predictions_dirs:
            # 複数のディレクトリをリストとして保持
            self.gnn_predictions_dirs = [Path(d) for d in gnn_predictions_dirs]
        else:
            # デフォルトはpro_GNN/resultsディレクトリ
            self.gnn_predictions_dirs = [self.bioiso_dir / "pro_GNN" / "results"]
        
        # GNN予測結果ディレクトリ名から出力フォルダの接尾辞を自動生成
        def extract_suffix_from_path(path):
            """パスから接尾辞（_xxx部分）を抽出"""
            dir_name = Path(path).name
            if '_' in dir_name:
                # results_blank -> _blank
                suffix = '_' + dir_name.split('_', 1)[1]
                return suffix
            return ''
        
        # 複数ディレクトリの場合は最初のものから接尾辞を取得
        gnn_suffix = extract_suffix_from_path(self.gnn_predictions_dirs[0])
        # 接尾辞を保存（CVファイル選択で使用）
        self.gnn_suffix = gnn_suffix
        
        # 結果と図の出力ディレクトリを設定
        if results_dir:
            self.results_dir = Path(results_dir)
        else:
            # デフォルト: results + GNN予測ディレクトリの接尾辞
            # 複数ディレクトリの場合は "combined" を追加
            if len(self.gnn_predictions_dirs) > 1:
                default_name = f"results_combined{gnn_suffix}" if gnn_suffix else "results_combined"
            else:
                default_name = f"results{gnn_suffix}" if gnn_suffix else "results"
            self.results_dir = Path(__file__).parent / default_name
            
        if figures_dir:
            self.figures_dir = Path(figures_dir)
        else:
            # デフォルト: figures + GNN予測ディレクトリの接尾辞
            # 複数ディレクトリの場合は "combined" を追加
            if len(self.gnn_predictions_dirs) > 1:
                default_name = f"figures_combined{gnn_suffix}" if gnn_suffix else "figures_combined"
            else:
                default_name = f"figures{gnn_suffix}" if gnn_suffix else "figures"
            self.figures_dir = Path(__file__).parent / default_name
            
        # LGBM予測結果のディレクトリを設定
        if lgbm_predictions_dir:
            self.lgbm_predictions_dir = Path(lgbm_predictions_dir)
        else:
            self.lgbm_predictions_dir = self.bioiso_dir / "gbdt"
        
        # 出力ディレクトリを作成
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        # 図の出力形式
        self.figure_format = figure_format
        
        # LGBMを含めるかどうか
        self.include_lgbm = include_lgbm
        
        # 比較モード
        self.comparison_mode = comparison_mode
        
        # モデル設定（2グラフ入力と3グラフ入力の両方に対応）
        self.models = ['lgbm-reg-abs-aug', 'lgbm-cls-abs-aug',
                       'pair-cat', 'pair-diff', 'pair-product',
                       'pair-concat', 'pair-hierarchical',
                       'pair-separate_common_concat', 'pair-separate_common_hierarchical']
        
        self.model2label = {
            'lgbm-reg-abs-aug': 'LGBM-Regressor',
            'lgbm-cls-abs-aug': 'LGBM-Classifier', 
            # 2グラフ入力モデル（通常のペア）
            'pair-cat': 'GraphBioisostere (concat)',
            'pair-diff': 'GraphBioisostere (diff)',
            'pair-product': 'GraphBioisostere (product)',
            # 3グラフ入力モデル - Shared Encoder（triple: 共通部分構造 + 2フラグメント）
            'pair-concat': 'GraphTriple (concat)',
            'pair-hierarchical': 'GraphTriple (hierarchical)',
            # 3グラフ入力モデル - Separate Encoder（共通構造用とフラグメント用で別エンコーダー）
            'pair-separate_common_concat': 'GraphTriple-Separate (concat)',
            'pair-separate_common_hierarchical': 'GraphTriple-Separate (hierarchical)'
        }
        
        # 図のスタイル設定
        self.fontsize = 20
        self.order = ['LGBM-Regressor', 'LGBM-Classifier', 
                      'GraphBioisostere (concat)', 'GraphBioisostere (product)', 'GraphBioisostere (diff)']
        
        # データを初期化時に読み込み
        self.cvs = None
        self.merge_df = None
        self.lgbm_records = None  # LGBMの学習データ（joblibファイル）
        self.lgbm_infos_fold = {}
        self.gnn_infos_fold = {}
        
    def _extract_condition_identifier(self, path_or_name):
        """パスまたは名前から条件識別子を抽出
        
        Args:
            path_or_name: ファイルパス、ディレクトリパス、またはディレクトリ名
            
        Returns:
            str: 条件識別子（例: 'frag', 'consistentsmiles', 'blank'）
        """
        if isinstance(path_or_name, Path):
            dir_name = path_or_name.name
        else:
            dir_name = str(path_or_name)
        
        # results_consistentsmiles_41_frag -> frag
        # results_consistentsmiles -> consistentsmiles
        # results_blank -> blank
        
        # 最後のアンダースコア以降を取得
        parts = dir_name.split('_')
        
        # 数字だけの部分（エポック番号など）はスキップ
        identifier_parts = []
        for part in reversed(parts):
            if part == 'results':
                break
            # 数字のみの部分はスキップ
            if part.isdigit():
                continue
            identifier_parts.insert(0, part)
        
        if identifier_parts:
            return '_'.join(identifier_parts)
        return 'default'
    
    def _map_model_name(self, model_name, condition_identifier=None):
        """モデル名を表示用ラベルにマッピング
        
        Args:
            model_name: 元のモデル名（例: 'pair-cat'）
            condition_identifier: 条件識別子（comparison_mode='compare'の場合のみ使用）
            
        Returns:
            str: 表示用モデル名
        """
        # 基本的なマッピング
        base_label = self.model2label.get(model_name, model_name)
        
        # 比較モードの場合は条件識別子を付加
        if self.comparison_mode == 'compare' and condition_identifier:
            return f"{base_label} [{condition_identifier}]"
        
        return base_label
        
    def load_data(self):
        """必要なデータファイルを読み込み"""
        print("データを読み込み中...")
        
        # Cross-validation データの読み込み
        if self.cv_pkl_file:
            # pklファイルが直接指定されている場合
            cv_path = self.cv_pkl_file
            if not cv_path.is_absolute():
                # 相対パスの場合はbase_dirからの相対パスとして解釈
                cv_path = self.base_dir / cv_path
            
            if not cv_path.exists():
                raise FileNotFoundError(f"指定されたCross-validation データファイルが見つかりません: {cv_path}")
            
            print(f"指定されたCross-validation データを読み込み: {cv_path}")
        else:
            # pklファイルが指定されていない場合は自動検出（接尾辞に基づいてファイルを選択）
            if self.gnn_suffix:
                cv_filename = f"tid_5cv{self.gnn_suffix}.pkl"
            else:
                cv_filename = "tid_5cv.pkl"
            
            cv_path = self.bioiso_dir / "splitting" / cv_filename
            if not cv_path.exists():
                raise FileNotFoundError(f"Cross-validation データファイルが見つかりません: {cv_path}")
            
            print(f"Cross-validation データを読み込み: {cv_filename}")
        
        with open(cv_path, "rb") as f:
            self.cvs = pickle.load(f)
        print(f"  → {len(self.cvs)} フォールドのデータを読み込みました")
            
        # マージデータの読み込み（GNN評価用）
        merge_path = self.base_dir / "MMP_dataset" / "dataset_consistentsmiles_with_properties.csv"
        if not merge_path.exists():
            raise FileNotFoundError(f"マージデータファイルが見つかりません: {merge_path}")
        self.merge_df = pd.read_csv(merge_path)
        
        # SMI-KEYの作成
        self.merge_df['SMI-KEY'] = self.merge_df['REF-SMILES'] + '/' + self.merge_df['PRB-SMILES']
        
        # LGBMの評価用に学習時と同じjoblibファイルを読み込む
        if self.include_lgbm:
            self._load_lgbm_data()
        
        # データの前処理
        self._preprocess_data()
        
        print("データの読み込みが完了しました")
    
    def _load_lgbm_data(self):
        """LGBM評価用にjoblibファイルを読み込む"""
        import joblib
        
        # LGBMの予測ディレクトリから使用されたデータファイルを推測
        # 例: results_consistentsmiles-2048 -> dataset_consistentsmiles-2048.joblib
        lgbm_dir_name = self.lgbm_predictions_dir.name
        
        # ディレクトリ名から "results_" を除去してデータファイル名を推測
        if lgbm_dir_name.startswith("results_"):
            data_suffix = lgbm_dir_name.replace("results_", "")
            joblib_filename = f"dataset_{data_suffix}.joblib"
        else:
            # デフォルト
            joblib_filename = "dataset_consistentsmiles-2048.joblib"
        
        joblib_path = self.base_dir / "MMP_dataset" / joblib_filename
        
        if joblib_path.exists():
            print(f"LGBM評価用データを読み込み: {joblib_filename}")
            self.lgbm_records = joblib.load(joblib_path)
            print(f"  レコード数: {len(self.lgbm_records)}")
        else:
            print(f"Warning: LGBMデータファイルが見つかりません: {joblib_path}")
            print(f"  LGBMの評価はCSVデータで行われます（精度が低くなる可能性があります）")
            self.lgbm_records = None
        
    def _preprocess_data(self):
        """データの前処理"""
        # 目的変数が一致しているもののみを抽出
        unique_keys_x = self.merge_df.groupby("SMI-KEY")["label_bin"].unique()
        unique_keys = unique_keys_x.index[unique_keys_x.apply(len)==1]
        avail_indices = self.merge_df.index[self.merge_df["SMI-KEY"].isin(unique_keys)].values

        unique_keys_len_d = unique_keys_x.apply(len).to_dict()
        self.merge_df["SMI-NUM"] = self.merge_df["SMI-KEY"].apply(unique_keys_len_d.get)
        self.merge_df["label_bin_v2"] = self.merge_df.apply(lambda x: x["label_bin"] if x["SMI-NUM"]==1 else False, axis=1)
        self.merge_df_uniq = self.merge_df.drop_duplicates(subset=["SMI-KEY"])
        self.merge_df_single = self.merge_df.loc[avail_indices]
        self.merge_df_single_uni = self.merge_df_single.drop_duplicates(subset=["SMI-KEY"])

        # インデックス設定
        all_unique_indices = self.merge_df_uniq.index
        single_label_unique_indices = self.merge_df_single_uni.index

        # 各フォールドで利用可能なテストインデックスを計算
        self.available_test_indices_all_unique = {}
        self.available_test_indices_single_label = {}
        self.available_test_indices_bad = {}
        
        all_unique_indices_set = set(all_unique_indices)
        single_label_unique_indices_set = set(single_label_unique_indices)
        all_indices_set = set(self.merge_df.index.values)
        
        for fold in range(len(self.cvs)):
            train, valid, test = self.cvs[fold]
            test_indices = [sample["index"] for sample in test]
            
            self.available_test_indices_all_unique[fold] = [idx in all_unique_indices_set for idx in test_indices]
            self.available_test_indices_single_label[fold] = [idx in single_label_unique_indices_set for idx in test_indices]
            self.available_test_indices_bad[fold] = [idx in all_indices_set for idx in test_indices]


        
    def get_group_scores(self, pred, pred_binary, y, groups, min_samples=50):
        """グループ別スコアの計算"""
        unique_groups = np.unique(groups)
        scores = {}
        for group_id in unique_groups:
            group_mask = groups == group_id
            if np.sum(group_mask) > min_samples and y[group_mask].std() > 0.01:
                group_y = y[group_mask]
                group_pred = pred[group_mask]
                group_pred_binary = pred_binary[group_mask]
                
                group_auc = roc_auc_score(group_y, group_pred) if group_y.std() > 0 else np.nan
                
                scores[group_id] = {
                    "group_count": np.sum(group_mask),
                    "positive_count": np.sum(group_y == 1),
                    "negative_count": np.sum(group_y == 0),
                    "group_accuracy": accuracy_score(group_y.astype(int), group_pred_binary),
                    "group_precision": precision_score(group_y.astype(int), group_pred_binary, zero_division=0),
                    "group_recall": recall_score(group_y.astype(int), group_pred_binary, zero_division=0),
                    "group_f1": f1_score(group_y.astype(int), group_pred_binary, zero_division=0),
                    "group_auc": group_auc
                }
        return scores

    def calculate_results(self, y_test, pred, pred_binary, groups):
        """評価指標の計算"""
        global_roc_auc = roc_auc_score(y_test, pred)
        
        # Binary classification metrics
        y_binary = y_test.astype(int)
        pred_binary = pred_binary.astype(int)
        
        accuracy = accuracy_score(y_binary, pred_binary)
        precision = precision_score(y_binary, pred_binary, zero_division=0)
        recall = recall_score(y_binary, pred_binary, zero_division=0)
        f1 = f1_score(y_binary, pred_binary, zero_division=0)
        mcc = matthews_corrcoef(y_binary, pred_binary)

        group_scores = self.get_group_scores(pred, pred_binary, y_test, groups)
        mean_group_scores = {}
        if group_scores:
            for metric in group_scores[list(group_scores.keys())[0]].keys():
                values = [scores[metric] for group_id, scores in group_scores.items() if not np.isnan(scores[metric])]
                mean_group_scores["mean_" + metric] = float(np.mean(values)) if values else np.nan

        results = {
            "global_roc_auc": global_roc_auc,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mcc": mcc,
            **mean_group_scores
        }
        results = {k: float(v) for k, v in results.items()}
        return results, group_scores



    def generate_figures(self, results_data):
        """図を生成して保存"""
        print("図を生成中...")
        
        # 基本的な図のスタイル設定
        plt.style.use('default')
        sns.set_palette("Set2")
        
        # フォント設定（文字化け防止）
        plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        
        # Single labelデータセットの図を生成
        if 'single' in results_data:
            if self.comparison_mode == 'compare':
                # 比較モード: 異なる条件間の比較図を生成
                self._generate_comparison_figures(results_data['single'])
            elif self.include_lgbm:
                # 統合モード + LGBM
                self._generate_combined_figures(results_data['single'])
            else:
                # 統合モード（GNNのみ）
                self._generate_gnn_figures(results_data['single'])
            
            # Tanimoto類似度別の予測結果を生成（単一GNNファイルの場合のみ）
            if self._should_generate_tanimoto_analysis():
                print("\nTanimoto類似度別の分析を実行します...")
                self._generate_tanimoto_analysis_figures(results_data['single'])
            else:
                print("\nTanimoto類似度別の分析をスキップします（複数GNN統合モードまたはCSV直接指定モード）")
            
        print("図の生成が完了しました")
        
    def _generate_comparison_figures(self, data):
        """条件間比較用の図を生成"""
        results_df = data['results']
        group_scores_df = data['group_scores']
        
        # ユニークな条件とモデルを取得
        conditions = sorted(results_df['condition'].unique())
        base_models = ['GraphBioisostere (concat)', 'GraphBioisostere (product)', 'GraphBioisostere (diff)']
        
        print(f"\n比較対象の条件: {conditions}")
        print(f"比較対象のモデル: {base_models}")
        
        # 条件ごとにモデル名のリストを作成
        comparison_order = []
        for model in base_models:
            for condition in conditions:
                model_name = f"{model} [{condition}]"
                if model_name in results_df['Model'].values:
                    comparison_order.append(model_name)
        
        # Global ROC AUC boxplot（条件間比較）
        # 統計情報を計算
        stats_df = results_df.groupby('Model')['global_roc_auc'].agg(['mean', 'std'])
        
        fig, ax = plt.subplots(figsize=(12, max(6, len(comparison_order) * 0.4)))
        sns.boxplot(data=results_df, y="Model", x="global_roc_auc",
                   hue="Model", palette="Set2", ax=ax, order=comparison_order, legend=False)
        plt.vlines(0.5, -0.5, len(comparison_order)-0.5, linestyle="--", color="black")
        plt.ylim(-0.5, len(comparison_order)-0.5)
        plt.xlim(0.40, 0.87)
        
        # 平均値±標準偏差を各箱ひげ図の右に表示
        for i, model in enumerate(comparison_order):
            if model in stats_df.index:
                mean_val = stats_df.loc[model, 'mean']
                std_val = stats_df.loc[model, 'std']
                ax.text(0.80, i, f'{mean_val:.4f}±{std_val:.4f}',
                       va='center', fontsize=self.fontsize-4, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.8))
        
        ax.set_ylabel("Model", fontsize=self.fontsize)
        ax.set_xlabel("Global ROC AUC", fontsize=self.fontsize)
        ax.tick_params(labelsize=self.fontsize-2)
        plt.tight_layout()
        plt.savefig(self.figures_dir / f"comparison-global-rocauc.{self.figure_format}", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Target ROC AUC violinplot（条件間比較）
        if not group_scores_df.empty:
            fig, ax = plt.subplots(figsize=(10, max(6, len(comparison_order) * 0.4)))
            sns.violinplot(data=group_scores_df, y="Model", x="group_auc",
                          hue="Model", ax=ax, order=comparison_order, cut=0, inner="quartile", 
                          palette="Set2", legend=False)
            plt.vlines(0.5, -0.5, len(comparison_order)-0.5, linestyle="--", color="black")
            plt.ylim(-0.5, len(comparison_order)-0.5)
            ax.set_ylabel("Model", fontsize=self.fontsize)
            ax.set_xlabel("Target ROC-AUC", fontsize=self.fontsize)
            ax.tick_params(labelsize=self.fontsize-2)
            plt.tight_layout()
            plt.savefig(self.figures_dir / f"comparison-group-rocauc.{self.figure_format}", dpi=300, bbox_inches='tight')
            plt.show()
        
        # モデル・条件別の統計情報を表示
        self._display_comparison_statistics(results_df)
        
        # 各モデルについて条件間の直接比較図を生成
        for base_model in base_models:
            self._generate_per_model_comparison(results_df, group_scores_df, base_model, conditions)
    
    def _generate_per_model_comparison(self, results_df, group_scores_df, base_model, conditions):
        """特定のモデルについて条件間の直接比較図を生成"""
        # このモデルの全条件のデータを抽出
        model_data = results_df[results_df['Model'].str.startswith(base_model)]
        
        if model_data.empty:
            return
        
        # モデル名を簡略化（ファイル名用）
        model_short = base_model.replace('GraphBioisostere (', '').replace(')', '').strip()
        
        # 条件別のモデル名リスト
        condition_models = [f"{base_model} [{cond}]" for cond in conditions if f"{base_model} [{cond}]" in model_data['Model'].values]
        
        if len(condition_models) < 2:
            return  # 比較対象が1つ以下の場合はスキップ
        
        # Global ROC AUC 比較
        # 統計情報を計算
        stats_df = model_data.groupby('Model')['global_roc_auc'].agg(['mean', 'std'])
        
        fig, ax = plt.subplots(figsize=(10, 3))
        sns.boxplot(data=model_data, y="Model", x="global_roc_auc",
                   hue="Model", palette="Set2", ax=ax, order=condition_models, legend=False)
        plt.vlines(0.5, -0.5, len(condition_models)-0.5, linestyle="--", color="black")
        plt.ylim(-0.5, len(condition_models)-0.5)
        plt.xlim(0.40, 0.87)
        
        # 平均値±標準偏差を各箱ひげ図の右に表示
        for i, model in enumerate(condition_models):
            if model in stats_df.index:
                mean_val = stats_df.loc[model, 'mean']
                std_val = stats_df.loc[model, 'std']
                ax.text(0.80, i, f'{mean_val:.4f}±{std_val:.4f}',
                       va='center', fontsize=self.fontsize-4, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.8))
        
        ax.set_ylabel("", fontsize=self.fontsize)
        ax.set_xlabel("Global ROC AUC", fontsize=self.fontsize)
        ax.set_title(f"{base_model}", fontsize=self.fontsize)
        ax.tick_params(labelsize=self.fontsize-2)
        plt.tight_layout()
        plt.savefig(self.figures_dir / f"comparison-{model_short}-global-rocauc.{self.figure_format}", dpi=300, bbox_inches='tight')
        plt.show()
        
    def _display_comparison_statistics(self, results_df):
        """条件間比較の統計情報を表示"""
        print("\n=== 条件間比較統計 ===")
        
        # 条件別・モデル別の統計
        stats = results_df.groupby(['condition', 'Model'])['global_roc_auc'].agg(['mean', 'std', 'count']).round(4)
        print(stats)
        
        # 各条件での最良モデル
        print("\n=== 各条件での最良モデル ===")
        for condition in results_df['condition'].unique():
            cond_data = results_df[results_df['condition'] == condition]
            best_model = cond_data.groupby('Model')['global_roc_auc'].mean().idxmax()
            best_auc = cond_data.groupby('Model')['global_roc_auc'].mean().max()
            print(f"{condition}: {best_model} (AUC: {best_auc:.4f})")
    
    def _generate_gnn_figures(self, data):
        """GNN結果専用の図を生成"""
        results_df = data['results']
        group_scores_df = data['group_scores']
        
        # GNN用のモデル順序（データに実際に存在するモデルを使用）
        # 2グラフ版: GraphBioisostere (concat/product/diff)
        # 3グラフ版 - Shared: GraphTriple (concat/hierarchical)
        # 3グラフ版 - Separate: GraphTriple-Separate (concat/hierarchical)
        available_models = results_df['Model'].unique().tolist()
        
        # 優先順位に基づいてソート
        model_priority = {
            # 2グラフモデル
            'GraphBioisostere (concat)': 1,
            'GraphBioisostere (product)': 2,
            'GraphBioisostere (diff)': 3,
            # 3グラフモデル - Shared Encoder
            'GraphTriple (concat)': 11,
            'GraphTriple (hierarchical)': 12,
            # 3グラフモデル - Separate Encoder
            'GraphTriple-Separate (concat)': 21,
            'GraphTriple-Separate (hierarchical)': 22
        }
        gnn_order = sorted(available_models, key=lambda x: model_priority.get(x, 999))
        
        print(f"利用可能なGNNモデル: {gnn_order}")
        
        # Global ROC AUC boxplot（モデル数に応じて図のサイズを調整）
        # 統計情報を計算
        stats_df = results_df.groupby('Model')['global_roc_auc'].agg(['mean', 'std'])
        
        fig_height = max(4, len(gnn_order) * 0.5)
        fig, ax = plt.subplots(figsize=(10, fig_height))
        sns.boxplot(data=results_df, y="Model", x="global_roc_auc",
                   hue="Model", palette="Set2", ax=ax, order=gnn_order, legend=False)
        plt.vlines(0.5, -0.5, len(gnn_order)-0.5, linestyle="--", color="black")
        plt.ylim(-0.5, len(gnn_order)-0.5)
        plt.xlim(0.40, 0.87)
        
        # 平均値±標準偏差を各箱ひげ図の右に表示
        for i, model in enumerate(gnn_order):
            if model in stats_df.index:
                mean_val = stats_df.loc[model, 'mean']
                std_val = stats_df.loc[model, 'std']
                ax.text(0.80, i, f'{mean_val:.4f}±{std_val:.4f}',
                       va='center', fontsize=self.fontsize-2, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.8))
        
        ax.set_ylabel("Model", fontsize=self.fontsize)
        ax.set_xlabel("Global ROC AUC", fontsize=self.fontsize)
        ax.tick_params(labelsize=self.fontsize)
        plt.tight_layout()
        plt.savefig(self.figures_dir / f"gnn-global-rocauc.{self.figure_format}", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Target ROC AUC violinplot（モデル数に応じて図のサイズを調整）
        if not group_scores_df.empty:
            fig, ax = plt.subplots(figsize=(8, fig_height))
            sns.violinplot(data=group_scores_df, y="Model", x="group_auc",
                          hue="Model", ax=ax, order=gnn_order, cut=0, inner="quartile", 
                          palette="Set2", legend=False)
            plt.vlines(0.5, -0.5, len(gnn_order)-0.5, linestyle="--", color="black")
            plt.ylim(-0.5, len(gnn_order)-0.5)
            ax.set_ylabel("Model", fontsize=self.fontsize)
            ax.set_xlabel("Target ROC-AUC", fontsize=self.fontsize)
            ax.tick_params(labelsize=self.fontsize)
            plt.tight_layout()
            plt.savefig(self.figures_dir / f"gnn-group-rocauc.{self.figure_format}", dpi=300, bbox_inches='tight')
            plt.show()
            
        # モデル比較の統計情報を表示
        self._display_model_statistics(results_df)
        
    def _generate_combined_figures(self, data):
        """GNN + LGBM統合結果の図を生成"""
        results_df = data['results']
        group_scores_df = data['group_scores']
        
        # 全モデルの順序（LGBMを含む）
        full_order = self.order
        
        # Global ROC AUC boxplot
        # 統計情報を計算
        stats_df = results_df.groupby('Model')['global_roc_auc'].agg(['mean', 'std'])
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(data=results_df, y="Model", x="global_roc_auc",
                   hue="Model", palette="Set2", ax=ax, order=full_order, legend=False)
        plt.vlines(0.5, -0.5, len(full_order)-0.5, linestyle="--", color="black")
        plt.ylim(-0.5, len(full_order)-0.5)
        plt.xlim(0.40, 0.87)
        
        # 平均値±標準偏差を各箱ひげ図の右に表示
        for i, model in enumerate(full_order):
            if model in stats_df.index:
                mean_val = stats_df.loc[model, 'mean']
                std_val = stats_df.loc[model, 'std']
                ax.text(0.80, i, f'{mean_val:.4f}±{std_val:.4f}',
                       va='center', fontsize=self.fontsize-2, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.8))
        
        ax.set_ylabel("Model", fontsize=self.fontsize)
        ax.set_xlabel("Global ROC AUC", fontsize=self.fontsize)
        ax.tick_params(labelsize=self.fontsize)
        plt.tight_layout()
        plt.savefig(self.figures_dir / f"all-models-global-rocauc.{self.figure_format}", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Target ROC AUC violinplot
        if not group_scores_df.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.violinplot(data=group_scores_df, y="Model", x="group_auc",
                          hue="Model", ax=ax, order=full_order, cut=0, inner="quartile", 
                          palette="Set2", legend=False)
            plt.vlines(0.5, -0.5, len(full_order)-0.5, linestyle="--", color="black")
            plt.ylim(-0.5, len(full_order)-0.5)
            ax.set_ylabel("Model", fontsize=self.fontsize)
            ax.set_xlabel("Target ROC-AUC", fontsize=self.fontsize)
            ax.tick_params(labelsize=self.fontsize)
            plt.tight_layout()
            plt.savefig(self.figures_dir / f"all-models-group-rocauc.{self.figure_format}", dpi=300, bbox_inches='tight')
            plt.show()
            
        # モデル比較の統計情報を表示
        self._display_model_statistics(results_df)
        
    def _display_model_statistics(self, results_df):
        """モデルの統計情報を表示"""
        print("\n=== モデル性能統計 ===")
        stats = results_df.groupby('Model')['global_roc_auc'].agg(['mean', 'std', 'count']).round(4)
        print(stats)
        
        # 最良モデル
        best_model = results_df.groupby('Model')['global_roc_auc'].mean().idxmax()
        best_auc = results_df.groupby('Model')['global_roc_auc'].mean().max()
        print(f"\n最良モデル: {best_model} (AUC: {best_auc:.4f})")
    
    def _should_generate_tanimoto_analysis(self):
        """Tanimoto類似度別の分析を実行すべきかどうかを判定"""
        # 複数のGNN CSVファイルが直接指定されている場合はスキップ
        if hasattr(self, 'gnn_csv_files') and self.gnn_csv_files and len(self.gnn_csv_files) > 1:
            return False
        
        # 複数のGNN予測ディレクトリが指定されている場合はスキップ
        if len(self.gnn_predictions_dirs) > 1:
            return False
        
        # LGBMとの統合の場合もスキップ（予測ファイルへのアクセスが必要なため）
        # ただし、単一GNN + LGBMの場合は実行可能
        # ここでは単一GNNディレクトリの場合のみ実行とする
        if hasattr(self, 'gnn_csv_files') and self.gnn_csv_files:
            # CSV直接指定モードの場合はスキップ
            return False
        
        # 単一GNN予測ディレクトリの場合のみ実行
        return True
        
    def _generate_tanimoto_analysis_figures(self, data):
        """Tanimoto類似度別の予測結果を折れ線グラフで可視化（5-Fold CV平均）"""
        print("\nTanimoto類似度別の分析を実行中...")
        
        results_df = data['results']
        threshold = 0.3
        
        # 予測結果をFoldごとに収集
        fold_predictions = {}
        
        for fold in range(len(self.cvs)):
            train, valid, test = self.cvs[fold]
            test_indices = [sample["index"] for sample in test]
            
            # single label datasetのテストインデックスを取得
            available_mask = np.array(self.available_test_indices_single_label[fold])
            available_test_indices = np.array(test_indices)[available_mask]
            
            # フィルタ後のテストデータ
            merge_te = self.merge_df.loc[test_indices]
            merge_te_filtered = merge_te.iloc[available_mask]
            
            # 真のラベル
            y_true = merge_te_filtered["delta_value"].abs().values < threshold
            
            fold_predictions[fold] = {
                'indices': available_test_indices,
                'y_true_base': y_true,  # 基準となる真のラベル
                'y_pred': {},
                'y_prob': {}
            }
        
        # 各モデルの予測結果を収集
        print(f"  モデル数: {len(results_df['Model'].unique())}")
        
        # GNN予測ディレクトリ（単一の場合のみ対応）
        gnn_pred_dir = self.gnn_predictions_dirs[0]
        
        # 利用可能なモデルの確認
        available_models = {}
        two_graph_models = ["pair-cat", "pair-diff", "pair-product"]
        triple_shared_models = ["pair-concat", "pair-hierarchical"]
        triple_separate_models = ["pair-separate_common_concat", "pair-separate_common_hierarchical"]
        all_models = two_graph_models + triple_shared_models + triple_separate_models
        
        for model in all_models:
            if self.model2label.get(model) in results_df['Model'].unique():
                available_models[model] = self.model2label[model]
        
        for original_model, model_name in available_models.items():
            print(f"  処理中: {model_name}")
            collected_folds = 0
            
            for fold in range(len(self.cvs)):
                pred_path = gnn_pred_dir / f"cv{fold}" / original_model / "test_predictions.npz"
                
                if not pred_path.exists():
                    continue
                
                # 予測結果を読み込み
                pred_data = np.load(pred_path)
                
                # 2グラフ版と3グラフ版のキーの違いに対応
                if "y_prob_ab" in pred_data:
                    # 2グラフ版
                    pred_full = pred_data["y_prob_ab"][:,1]  # クラス1の確率
                elif "y_prob" in pred_data:
                    # 3グラフ版(triple)
                    pred_full = pred_data["y_prob"][:,1]  # クラス1の確率
                else:
                    print(f"    Warning: Fold {fold}, Model {model_name} - unknown prediction format")
                    print(f"      Available keys: {list(pred_data.keys())}")
                    continue
                
                # フィルタ適用
                available_mask = np.array(self.available_test_indices_single_label[fold])
                pred_filtered = pred_full[available_mask]
                
                # 予測値を保存
                fold_predictions[fold]['y_pred'][model_name] = (pred_filtered > 0.5).astype(int)
                fold_predictions[fold]['y_prob'][model_name] = pred_filtered
                collected_folds += 1
            
            print(f"    {model_name}: {collected_folds} フォールド分のデータを収集")
        
        # Tanimoto類似度でビン分け
        tanimoto_bins = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        tanimoto_labels = ['0.0-0.3', '0.3-0.4', '0.4-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0']
        
        # 各モデル・各Foldでの各Tanimotoビン別のROC AUCを計算
        tanimoto_results = []
        
        print(f"\n  Tanimotoビン別のROC AUC計算中...")
        for model_name in available_models.values():
            model_bin_count = 0
            for fold in range(len(self.cvs)):
                if model_name not in fold_predictions[fold]['y_prob']:
                    continue
                
                indices = fold_predictions[fold]['indices']
                y_true = fold_predictions[fold]['y_true_base']
                y_prob = fold_predictions[fold]['y_prob'][model_name]
                
                # merge_dfからTanimoto値を取得
                tanimoto_values = self.merge_df.loc[indices, 'Tanimoto'].values
                
                # Tanimotoビンごとに分類
                tanimoto_binned = pd.cut(tanimoto_values, bins=tanimoto_bins, labels=tanimoto_labels, include_lowest=True)
                
                for bin_label in tanimoto_labels:
                    bin_mask = (tanimoto_binned == bin_label)
                    
                    if np.sum(bin_mask) < 10:  # サンプル数が少ない場合はスキップ
                        continue
                    
                    bin_y_true = y_true[bin_mask]
                    bin_y_prob = y_prob[bin_mask]
                    
                    # ラベルが両方存在するか確認
                    if len(np.unique(bin_y_true)) < 2:
                        continue
                    
                    try:
                        bin_auc = roc_auc_score(bin_y_true, bin_y_prob)
                        bin_accuracy = accuracy_score(bin_y_true, (bin_y_prob > 0.5).astype(int))
                        
                        tanimoto_results.append({
                            'Model': model_name,
                            'Fold': fold,
                            'Tanimoto_Bin': bin_label,
                            'Tanimoto_Center': (tanimoto_bins[tanimoto_labels.index(bin_label)] + 
                                              tanimoto_bins[tanimoto_labels.index(bin_label) + 1]) / 2,
                            'ROC_AUC': bin_auc,
                            'Accuracy': bin_accuracy,
                            'Sample_Size': np.sum(bin_mask)
                        })
                        model_bin_count += 1
                    except Exception as e:
                        print(f"    Warning: {model_name}, Fold {fold}, Bin {bin_label} でエラー: {e}")
                        continue
            
            print(f"    {model_name}: {model_bin_count} データポイント（Fold x Tanimotoビン）を収集")
        
        if not tanimoto_results:
            print("警告: Tanimoto類似度別の分析に十分なデータがありません")
            print("  デバッグ情報:")
            print(f"    - モデル数: {len(results_df['Model'].unique())}")
            print(f"    - フォールド数: {len(self.cvs)}")
            print(f"    - merge_dfにTanimotoカラムが存在: {'Tanimoto' in self.merge_df.columns}")
            if 'Tanimoto' in self.merge_df.columns:
                print(f"    - Tanimoto値の統計: {self.merge_df['Tanimoto'].describe()}")
            return
        
        tanimoto_df = pd.DataFrame(tanimoto_results)
        
        # CSVに保存
        tanimoto_csv_path = self.results_dir / "tanimoto_analysis.csv"
        tanimoto_df.to_csv(tanimoto_csv_path, index=False)
        print(f"Tanimoto分析結果を保存: {tanimoto_csv_path}")
        
        # 折れ線グラフの作成
        self._plot_tanimoto_line_graphs(tanimoto_df)
        
    def _plot_tanimoto_line_graphs(self, tanimoto_df):
        """Tanimoto類似度別の性能を折れ線グラフで可視化（5-Fold CV平均）"""
        
        # モデルごと・Tanimotoビンごとに5フォールド全体での平均値と標準偏差を計算
        summary_df = tanimoto_df.groupby(['Model', 'Tanimoto_Center']).agg({
            'ROC_AUC': ['mean', 'std', 'count'],
            'Accuracy': ['mean', 'std', 'count'],
            'Sample_Size': 'sum'
        }).reset_index()
        
        summary_df.columns = ['Model', 'Tanimoto_Center', 'ROC_AUC_mean', 'ROC_AUC_std', 'ROC_AUC_count',
                             'Accuracy_mean', 'Accuracy_std', 'Accuracy_count', 'Sample_Size']
        
        # 利用可能なモデルを取得（結果がある順にソート）
        available_models = summary_df.groupby('Model')['ROC_AUC_mean'].mean().sort_values(ascending=False).index.tolist()
        
        print(f"\nTanimoto分析対象モデル数: {len(available_models)}")
        print(f"各モデルについて5フォールドの平均と標準偏差を計算")
        
        # 詳細な統計を表示（上位3モデルのみ）
        for i, model in enumerate(available_models[:3]):
            print(f"\n{model}:")
            model_summary = summary_df[summary_df['Model'] == model][
                ['Tanimoto_Center', 'ROC_AUC_mean', 'ROC_AUC_std', 'ROC_AUC_count', 'Sample_Size']
            ].sort_values('Tanimoto_Center')
            for _, row in model_summary.iterrows():
                print(f"  Tanimoto {row['Tanimoto_Center']:.2f}: "
                      f"AUC={row['ROC_AUC_mean']:.4f}±{row['ROC_AUC_std']:.4f} "
                      f"(n={int(row['ROC_AUC_count'])} folds, {int(row['Sample_Size'])} samples)")
        
        # ROC AUCの折れ線グラフ（エラーバー付き）
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for model_name in available_models:
            model_data = summary_df[summary_df['Model'] == model_name].sort_values('Tanimoto_Center')
            
            # 折れ線グラフとエラーバー
            ax.errorbar(model_data['Tanimoto_Center'], model_data['ROC_AUC_mean'],
                       yerr=model_data['ROC_AUC_std'],
                       marker='o', label=model_name, linewidth=2.5, markersize=8,
                       capsize=5, capthick=2, alpha=0.85)
        
        ax.axhline(y=0.5, color='black', linestyle='--', linewidth=1.5, alpha=0.5, label='Random (0.5)')
        ax.set_xlabel('Tanimoto Similarity', fontsize=self.fontsize + 2, fontweight='bold')
        ax.set_ylabel('ROC AUC (5-Fold CV Mean ± SD)', fontsize=self.fontsize + 2, fontweight='bold')
        ax.set_title('Model Performance vs Tanimoto Similarity (5-Fold Cross-Validation)', 
                    fontsize=self.fontsize + 4, fontweight='bold', pad=20)
        ax.tick_params(labelsize=self.fontsize)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=self.fontsize - 2,
                 frameon=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim([0.35, 0.90])
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / f"tanimoto_vs_rocauc_5fold.{self.figure_format}", 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Accuracyの折れ線グラフ（エラーバー付き）
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for model_name in available_models:
            model_data = summary_df[summary_df['Model'] == model_name].sort_values('Tanimoto_Center')
            
            # 折れ線グラフとエラーバー
            ax.errorbar(model_data['Tanimoto_Center'], model_data['Accuracy_mean'],
                       yerr=model_data['Accuracy_std'],
                       marker='s', label=model_name, linewidth=2.5, markersize=8,
                       capsize=5, capthick=2, alpha=0.85)
        
        ax.axhline(y=0.5, color='black', linestyle='--', linewidth=1.5, alpha=0.5, label='Random (0.5)')
        ax.set_xlabel('Tanimoto Similarity', fontsize=self.fontsize + 2, fontweight='bold')
        ax.set_ylabel('Accuracy (5-Fold CV Mean ± SD)', fontsize=self.fontsize + 2, fontweight='bold')
        ax.set_title('Model Accuracy vs Tanimoto Similarity (5-Fold Cross-Validation)', 
                    fontsize=self.fontsize + 4, fontweight='bold', pad=20)
        ax.tick_params(labelsize=self.fontsize)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=self.fontsize - 2,
                 frameon=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim([0.35, 0.90])
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / f"tanimoto_vs_accuracy_5fold.{self.figure_format}", 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # サンプルサイズの分布（1つのモデルのみを使用して重複カウントを避ける）
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 最初のモデルのデータのみを使用
        first_model = available_models[0]
        sample_dist = tanimoto_df[tanimoto_df['Model'] == first_model].groupby('Tanimoto_Bin')['Sample_Size'].sum().sort_index()
        
        ax.bar(range(len(sample_dist)), sample_dist.values, color='steelblue', alpha=0.7)
        ax.set_xlabel('Tanimoto Similarity Range', fontsize=self.fontsize)
        ax.set_ylabel('Sample Count (per Fold)', fontsize=self.fontsize)
        ax.set_title('Sample Distribution by Tanimoto Similarity', fontsize=self.fontsize + 2)
        ax.set_xticks(range(len(sample_dist)))
        ax.set_xticklabels(sample_dist.index, rotation=45, ha='right')
        ax.tick_params(labelsize=self.fontsize - 2)
        ax.grid(True, alpha=0.3, axis='y')
        
        # サンプル数を棒グラフの上に表示
        for i, v in enumerate(sample_dist.values):
            ax.text(i, v, str(int(v)), ha='center', va='bottom', fontsize=self.fontsize - 4)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / f"tanimoto_sample_distribution.{self.figure_format}", 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\nTanimoto類似度別の図を保存しました:")
        print(f"  - {self.figures_dir}/tanimoto_vs_rocauc_5fold.{self.figure_format}")
        print(f"  - {self.figures_dir}/tanimoto_vs_accuracy_5fold.{self.figure_format}")
        print(f"  - {self.figures_dir}/tanimoto_sample_distribution.{self.figure_format}")
        


    def load_existing_results(self):
        """既存のCSVファイルから結果を読み込み"""
        print("既存の結果ファイルを読み込み中...")
        
        results_data = {}
        datasets = ['single']  # mainで使用するデータセットのみ
        
        for dataset_name in datasets:
            try:
                # CSVファイルのパス（LGBMを含むかどうかで異なる）
                if self.include_lgbm:
                    results_path = self.results_dir / f"all_results_df_{dataset_name}.csv"
                    group_scores_path = self.results_dir / f"all_group_scores_df_{dataset_name}.csv"
                else:
                    results_path = self.results_dir / f"gnn_results_df_{dataset_name}.csv"
                    group_scores_path = self.results_dir / f"gnn_group_scores_df_{dataset_name}.csv"
                
                # ファイルの存在チェック
                if not results_path.exists():
                    print(f"Warning: {dataset_name} データセットの結果ファイルが見つかりません")
                    continue
                
                # データを読み込み
                results_df = pd.read_csv(results_path)
                
                # group_scoresファイルは存在しない場合もあるので、オプショナル
                group_scores_df = pd.DataFrame()
                
                if group_scores_path.exists():
                    group_scores_df = pd.read_csv(group_scores_path)
                
                # マージ結果の作成
                if not results_df.empty:
                    results_df_merge = results_df.copy()
                    group_scores_df_merge = group_scores_df.copy() if not group_scores_df.empty else pd.DataFrame()
                    
                    results_df_merge["Model"] = results_df_merge["model"].map(self.model2label)
                    if not group_scores_df_merge.empty:
                        group_scores_df_merge["Model"] = group_scores_df_merge["model"].map(self.model2label)
                    
                    results_data[dataset_name] = {
                        'results': results_df_merge,
                        'group_scores': group_scores_df_merge
                    }
                    
                    print(f"  {dataset_name} データセット: 読み込み完了")
                
            except Exception as e:
                print(f"Warning: {dataset_name} データセットの読み込みに失敗しました: {e}")
                continue
        
        if not results_data:
            raise ValueError("読み込み可能な結果ファイルが見つかりませんでした")
            
        print("既存結果の読み込みが完了しました")
        return results_data

    def run_full_analysis(self):
        """完全な分析を実行（予測結果から直接計算）"""
        print("=== バイオアイソスター予測結果分析を開始 ===")
        
        # CSVファイルが直接指定されている場合
        if hasattr(self, 'gnn_csv_files') and self.gnn_csv_files:
            print("\nGNN結果CSVファイルが直接指定されています。")
            gnn_results = self.load_gnn_from_csv_files()
            
            if self.include_lgbm:
                print("Warning: CSV直接指定モードではLGBMの統合はサポートされていません。GNN結果のみ使用します。")
                self.include_lgbm = False
            
            results_data = gnn_results
        # 複数のGNNディレクトリが指定されている場合は既存CSVから読み込みのみ
        elif len(self.gnn_predictions_dirs) > 1:
            print("\n複数のGNNディレクトリが指定されています。既存のCSVファイルから読み込みます。")
            print("（データファイルの読み込みはスキップします）")
            gnn_results = self.load_multiple_gnn_results()
            
            if self.include_lgbm:
                print("Warning: 複数GNNモードではLGBMの統合はサポートされていません。GNN結果のみ使用します。")
                self.include_lgbm = False
            
            results_data = gnn_results
        else:
            # 単一ディレクトリの場合は予測ファイルから直接計算
            # データの読み込み
            self.load_data()
            
            gnn_results = self.calculate_gnn_results_from_predictions()
            
            if self.include_lgbm:
                # LGBM予測結果の読み込みと結果計算
                lgbm_results = self.calculate_lgbm_results_from_predictions()
                
                # LGBM結果が空でない場合のみ統合
                if not lgbm_results['single']['results'].empty:
                    # GNNとLGBMの結果を統合
                    results_data = {
                        'single': {
                            'results': pd.concat([gnn_results['single']['results'], 
                                                lgbm_results['single']['results']], 
                                               ignore_index=True),
                            'group_scores': pd.concat([gnn_results['single']['group_scores'], 
                                                     lgbm_results['single']['group_scores']], 
                                                    ignore_index=True) if not lgbm_results['single']['group_scores'].empty else gnn_results['single']['group_scores']
                        }
                    }
                else:
                    print("Warning: LGBM結果が見つからなかったため、GNN結果のみ使用します。")
                    results_data = gnn_results
                    # include_lgbmフラグをFalseに設定して、GNN専用の図を生成
                    self.include_lgbm = False
            else:
                # GNNのみ
                results_data = gnn_results
        
        # 結果をCSVとして保存
        self.save_results_to_csv(results_data)
        
        # 図の生成
        self.generate_figures(results_data)
        
        print("=== 分析が完了しました ===")
        print(f"結果は {self.results_dir} に保存されました")
        print(f"図は {self.figures_dir} に保存されました")
        
    def load_gnn_from_csv_files(self):
        """指定されたCSVファイルから直接GNN結果を読み込んで統合"""
        print("\nCSVファイルから直接読み込み中...")
        print(f"比較モード: {self.comparison_mode}")
        
        all_results = []
        all_group_scores = []
        
        for csv_file in self.gnn_csv_files:
            print(f"\n=== Loading: {csv_file} ===")
            
            if not csv_file.exists():
                print(f"  Warning: ファイルが見つかりません: {csv_file}")
                continue
            
            # results CSV読み込み
            results_df = pd.read_csv(csv_file)
            
            # ファイル名から識別子を抽出（例: results_consistentsmiles -> consistentsmiles）
            file_stem = csv_file.stem  # gnn_results_df_single
            parent_name = csv_file.parent.name  # results or results_consistentsmiles
            
            # 親ディレクトリから条件識別子を抽出
            if parent_name != "results":
                identifier = self._extract_condition_identifier(parent_name)
            else:
                # results_xxx の形式の親の親から取得
                identifier = self._extract_condition_identifier(csv_file.parent.parent.name)
            
            print(f"  条件識別子: {identifier}")
            
            # ディレクトリ情報を追加（記録用）
            results_df['pred_dir'] = identifier
            results_df['condition'] = identifier
            
            # モデル名のマッピング
            if self.comparison_mode == 'compare':
                # 比較モード: モデル名に条件識別子を付加
                results_df['Model'] = results_df['model'].apply(
                    lambda x: self._map_model_name(x, identifier)
                )
            else:
                # 統合モード: 条件識別子なしでマッピング
                results_df['Model'] = results_df['model'].apply(
                    lambda x: self._map_model_name(x, None)
                )
            
            all_results.append(results_df)
            
            # group_scores も同じディレクトリから探す
            group_scores_file = csv_file.parent / "gnn_group_scores_df_single.csv"
            if group_scores_file.exists():
                group_scores_df = pd.read_csv(group_scores_file)
                group_scores_df['pred_dir'] = identifier
                group_scores_df['condition'] = identifier
                
                # モデル名のマッピング
                if self.comparison_mode == 'compare':
                    group_scores_df['Model'] = group_scores_df['model'].apply(
                        lambda x: self._map_model_name(x, identifier)
                    )
                else:
                    group_scores_df['Model'] = group_scores_df['model'].apply(
                        lambda x: self._map_model_name(x, None)
                    )
                
                all_group_scores.append(group_scores_df)
        
        if not all_results:
            raise FileNotFoundError("指定されたCSVファイルから結果を読み込めませんでした。")
        
        # 統合
        combined_results = pd.concat(all_results, ignore_index=True)
        combined_group_scores = pd.concat(all_group_scores, ignore_index=True) if all_group_scores else pd.DataFrame()
        
        if not combined_group_scores.empty:
            # tidがindexにある場合のみreset_index
            if combined_group_scores.index.name == "tid" or (hasattr(combined_group_scores.index, 'names') and "tid" in combined_group_scores.index.names):
                combined_group_scores = combined_group_scores.reset_index()
            # tidがカラムとして既に存在しない場合のみindex名を設定してreset
            elif "tid" not in combined_group_scores.columns:
                combined_group_scores.index.name = "tid"
                combined_group_scores = combined_group_scores.reset_index()
        
        print(f"\n統合完了: {len(combined_results)} rows")
        if self.comparison_mode == 'compare':
            print(f"比較対象の条件数: {combined_results['condition'].nunique()}")
            print(f"条件別サンプル数:")
            print(combined_results.groupby('condition').size())
        
        return {
            'single': {
                'results': combined_results,
                'group_scores': combined_group_scores
            }
        }
    
    def load_multiple_gnn_results(self):
        """複数のGNN結果ディレクトリから既存CSVを読み込んで統合"""
        print("\n複数のGNN結果CSVファイルを読み込み中...")
        print(f"比較モード: {self.comparison_mode}")
        
        all_results = []
        all_group_scores = []
        
        for gnn_pred_dir in self.gnn_predictions_dirs:
            dir_name = gnn_pred_dir.name
            print(f"\n=== Loading results from: {dir_name} ===")
            
            # 条件識別子を抽出
            identifier = self._extract_condition_identifier(gnn_pred_dir)
            print(f"  条件識別子: {identifier}")
            
            # 各ディレクトリのresultsサブディレクトリ（またはtestディレクトリ）からCSVを探す
            # 想定パターン:
            # 1. gnn_pred_dir/../test/results/gnn_results_df_single.csv
            # 2. gnn_pred_dir/test/results/gnn_results_df_single.csv
            
            candidate_result_dirs = [
                gnn_pred_dir.parent / "test" / "results",  # pro_GNN/results/../test/results
                gnn_pred_dir / "test" / "results",  # pro_GNN/results/test/results
                gnn_pred_dir.parent / "results",  # pro_GNN/results/../results (= pro_GNN/results)
                gnn_pred_dir,  # 直接指定されたディレクトリ
            ]
            
            csv_found = False
            for result_dir in candidate_result_dirs:
                results_csv = result_dir / "gnn_results_df_single.csv"
                group_scores_csv = result_dir / "gnn_group_scores_df_single.csv"
                
                if results_csv.exists():
                    print(f"  Found CSV: {results_csv}")
                    results_df = pd.read_csv(results_csv)
                    
                    # ディレクトリ情報を追加（記録用）
                    results_df['pred_dir'] = dir_name
                    results_df['condition'] = identifier
                    
                    # モデル名のマッピング
                    if self.comparison_mode == 'compare':
                        # 比較モード: モデル名に条件識別子を付加
                        results_df['Model'] = results_df['model'].apply(
                            lambda x: self._map_model_name(x, identifier)
                        )
                    else:
                        # 統合モード: 条件識別子なしでマッピング
                        results_df['Model'] = results_df['model'].apply(
                            lambda x: self._map_model_name(x, None)
                        )
                    
                    all_results.append(results_df)
                    
                    # group_scores も読み込み
                    if group_scores_csv.exists():
                        group_scores_df = pd.read_csv(group_scores_csv)
                        group_scores_df['pred_dir'] = dir_name
                        group_scores_df['condition'] = identifier
                        
                        # モデル名のマッピング
                        if self.comparison_mode == 'compare':
                            group_scores_df['Model'] = group_scores_df['model'].apply(
                                lambda x: self._map_model_name(x, identifier)
                            )
                        else:
                            group_scores_df['Model'] = group_scores_df['model'].apply(
                                lambda x: self._map_model_name(x, None)
                            )
                        
                        all_group_scores.append(group_scores_df)
                    
                    csv_found = True
                    break
            
            if not csv_found:
                print(f"  Warning: {dir_name} の結果CSVが見つかりませんでした")
                print(f"    探索したパス:")
                for result_dir in candidate_result_dirs:
                    print(f"      - {result_dir / 'gnn_results_df_single.csv'}")
        
        if not all_results:
            raise FileNotFoundError("複数のGNNディレクトリが指定されましたが、結果CSVが見つかりませんでした。")
        
        # 統合
        combined_results = pd.concat(all_results, ignore_index=True)
        combined_group_scores = pd.concat(all_group_scores, ignore_index=True) if all_group_scores else pd.DataFrame()
        
        if not combined_group_scores.empty:
            # tidがindexにある場合のみreset_index
            if combined_group_scores.index.name == "tid" or (hasattr(combined_group_scores.index, 'names') and "tid" in combined_group_scores.index.names):
                combined_group_scores = combined_group_scores.reset_index()
            # tidがカラムとして既に存在しない場合のみindex名を設定してreset
            elif "tid" not in combined_group_scores.columns:
                combined_group_scores.index.name = "tid"
                combined_group_scores = combined_group_scores.reset_index()
        
        print(f"\n統合完了: {len(combined_results)} rows")
        if self.comparison_mode == 'compare':
            print(f"比較対象の条件数: {combined_results['condition'].nunique()}")
            print(f"条件別サンプル数:")
            print(combined_results.groupby('condition').size())
        
        return {
            'single': {
                'results': combined_results,
                'group_scores': combined_group_scores
            }
        }
    
    def calculate_gnn_results_from_predictions(self):
        """GNN予測結果から直接結果を計算（単一ディレクトリ専用）"""
        print("GNN予測結果から結果を計算中...")
        
        results_fold = []
        group_scores_fold = []
        threshold = 0.3
        
        # 単一ディレクトリのみ処理
        gnn_pred_dir = self.gnn_predictions_dirs[0]
        
        for fold in range(5):
            print(f"Processing fold {fold}")
            tr, va, te = self.cvs[fold]
            te_index = [a["index"] for a in te]
            
            # テストデータの取得
            merge_te = self.merge_df.loc[te_index]
            print(f"  Test set size: {len(merge_te)}")
            
            # 単一ラベルのデータのみ使用（results.ipynbと同じ条件）
            avail_te = self.available_test_indices_single_label[fold]
            merge_te_filtered = merge_te.iloc[avail_te]
            print(f"  Filtered test set size: {len(merge_te_filtered)}")
            print(f"  Available indices (single label): {sum(avail_te)}/{len(avail_te)}")
            
            # 各GNNモデルについて処理（2グラフ入力と3グラフ入力の両方に対応）
            # 2グラフモデル（通常のペア）
            two_graph_models = ["pair-cat", "pair-diff", "pair-product"]
            # 3グラフモデル - Shared Encoder（triple: 共通部分構造 + 2フラグメント）
            triple_shared_models = ["pair-concat", "pair-hierarchical"]
            # 3グラフモデル - Separate Encoder（共通構造用とフラグメント用で別エンコーダー）
            triple_separate_models = ["pair-separate_common_concat", "pair-separate_common_hierarchical"]
            
            # 全モデルリスト
            all_models = two_graph_models + triple_shared_models + triple_separate_models
            
            for model in all_models:
                pred_path = gnn_pred_dir / f"cv{fold}" / model / "test_predictions.npz"
                if pred_path.exists():
                    pred_data = np.load(pred_path)
                    
                    # 2グラフ版と3グラフ版の両方に対応
                    # 2グラフ版: y_prob_ab, y_prob_ba, y_pred_ab, y_pred_ba
                    # 3グラフ版(triple): y_prob, y_pred, y_true
                    if "y_prob_ab" in pred_data:
                        # 2グラフ版
                        pred = pred_data["y_prob_ab"][:,1]  # クラス1の確率
                    elif "y_prob" in pred_data:
                        # 3グラフ版(triple)
                        pred = pred_data["y_prob"][:,1]  # クラス1の確率
                    else:
                        print(f"Warning: Fold {fold}, Model {model} - unknown prediction format")
                        print(f"  Available keys: {list(pred_data.keys())}")
                        continue
                    
                    print(f"  Model {model}: prediction size = {len(pred)}")
                    
                    # 予測データのサイズが利用可能なテストデータと一致するか確認
                    if len(pred) != len(merge_te):
                        print(f"Warning: Fold {fold}, Model {model} - pred size ({len(pred)}) != test set size ({len(merge_te)})")
                        print(f"  Skipping this fold for model {model}")
                        continue
                    
                    # 単一ラベルのデータのみフィルタリング
                    avail_indices_array = np.array(avail_te)
                    pred_filtered = pred[avail_indices_array]
                    pred_binary = (pred_filtered > 0.5).astype(int)
                    
                    # 真のラベル
                    y = merge_te_filtered["delta_value"].abs().values < threshold
                    groups = merge_te_filtered["TID"].values
                    
                    if len(np.unique(y)) > 1:  # 両方のクラスが存在する場合のみ
                        results, group_scores = self.calculate_results(y, pred_filtered, pred_binary, groups)
                        
                        # モデル情報を追加
                        results.update({
                            "fold": fold,
                            "model": model
                        })
                        results_fold.append(results)
                        
                        # グループスコア
                        if group_scores:
                            gdf = pd.DataFrame(group_scores).T
                            gdf["fold"] = fold
                            gdf["model"] = model
                            group_scores_fold.append(gdf)
                else:
                    print(f"Warning: 予測ファイルが見つかりません: {pred_path}")
        
        # DataFrameに変換
        results_df = pd.DataFrame(results_fold)
        group_scores_df = pd.concat(group_scores_fold, axis=0) if group_scores_fold else pd.DataFrame()
        
        if not group_scores_df.empty:
            group_scores_df.index.name = "tid"
            group_scores_df = group_scores_df.reset_index()
        
        # モデル名をラベルにマッピング
        results_df["Model"] = results_df["model"].map(self.model2label)
        if not group_scores_df.empty:
            group_scores_df["Model"] = group_scores_df["model"].map(self.model2label)
        
        return {
            'single': {
                'results': results_df,
                'group_scores': group_scores_df
            }
        }
    
    def calculate_lgbm_results_from_predictions(self):
        """LGBM予測結果から直接結果を計算（学習時と同じjoblibデータを使用）"""
        print("LGBM予測結果から結果を計算中...")
        
        results_fold = []
        group_scores_fold = []
        threshold = 0.3
        
        # LGBMディレクトリ: ../../gbdt/results_consistentsmiles-2048 のような指定を想定
        lgbm_base = self.lgbm_predictions_dir
        
        # joblibデータが利用可能か確認
        use_joblib_data = self.lgbm_records is not None
        
        if use_joblib_data:
            print("✓ 学習時と同じjoblibデータを使用してLGBMを評価します")
        else:
            print("⚠ Warning: joblibデータが利用できません。CSVデータで評価します（精度が低い可能性があります）")
        
        for fold in range(5):
            print(f"Processing fold {fold}")
            tr, va, te = self.cvs[fold]
            te_index = [a["index"] for a in te]
            
            # joblibデータが利用可能か確認
            use_joblib_data = self.lgbm_records is not None
            
            if use_joblib_data:
                # joblibデータを使用（学習時と同じ）
                te_data = [self.lgbm_records[i] for i in te_index]
                
                # 真のラベルを取得
                y = np.array([abs(r['delta_value']) < threshold for r in te_data]).astype(int)
                
                # TIDを取得（group評価用）
                groups = np.array([r.get('TID', r.get('tid', 0)) for r in te_data])
                
                print(f"  Test set size: {len(te_data)}")
                print(f"  Class distribution: Class 0={np.sum(y==0)}, Class 1={np.sum(y==1)}")
            else:
                # CSVデータを使用（フォールバック）
                merge_te = self.merge_df.loc[te_index]
                y = (merge_te["delta_value"].abs().values < threshold).astype(int)
                groups = merge_te["TID"].values
                print(f"  Test set size (CSV): {len(y)}")
            
            # モデルについて処理
            for model in ["lgbm-reg-abs-aug", "lgbm-cls-abs-aug"]:
                pred_path = lgbm_base / model / f"fold{fold}_Test_predictions.npy"
                
                if pred_path.exists():
                    pred_raw = np.load(pred_path)
                    print(f"  Model {model}: prediction size = {len(pred_raw)}")
                    
                    # 予測データのサイズ確認
                    if len(pred_raw) != len(y):
                        print(f"Warning: Fold {fold}, Model {model} - pred size ({len(pred_raw)}) != test set size ({len(y)})")
                        print(f"  Skipping this fold for model {model}")
                        continue
                    
                    # 予測値とバイナリ予測の計算
                    if "cls" in model:
                        # 分類モデル: pred_rawは確率値（クラス1の確率）
                        pred = pred_raw
                        pred_binary = (pred > 0.5).astype(int)
                    else:
                        # 回帰モデル: pred_rawは絶対値の予測、小さい方が陽性（bioisoster）
                        pred = -pred_raw  # ROC-AUC計算用に符号反転
                        pred_binary = (pred_raw < threshold).astype(int)
                    
                    if len(np.unique(y)) > 1:  # 両方のクラスが存在する場合のみ
                        results, group_scores = self.calculate_results(y, pred, pred_binary, groups)
                        
                        # モデル情報を追加
                        results.update({
                            "fold": fold,
                            "model": model
                        })
                        results_fold.append(results)
                        
                        # グループスコア
                        if group_scores:
                            gdf = pd.DataFrame(group_scores).T
                            gdf["fold"] = fold
                            gdf["model"] = model
                            group_scores_fold.append(gdf)
                else:
                    print(f"Warning: 予測ファイルが見つかりません: {pred_path}")
        
        # 結果が空の場合の処理
        if not results_fold:
            print("Warning: LGBM予測結果が見つかりませんでした。空の結果を返します。")
            return {
                'single': {
                    'results': pd.DataFrame(),
                    'group_scores': pd.DataFrame()
                }
            }
        
        # DataFrameに変換
        results_df = pd.DataFrame(results_fold)
        group_scores_df = pd.concat(group_scores_fold, axis=0) if group_scores_fold else pd.DataFrame()
        
        if not group_scores_df.empty:
            group_scores_df.index.name = "tid"
            group_scores_df = group_scores_df.reset_index()
        
        # モデル名をラベルにマッピング
        results_df["Model"] = results_df["model"].map(self.model2label)
        if not group_scores_df.empty:
            group_scores_df["Model"] = group_scores_df["model"].map(self.model2label)
        
        return {
            'single': {
                'results': results_df,
                'group_scores': group_scores_df
            }
        }
    
    def save_results_to_csv(self, results_data):
        """結果をCSVファイルとして保存"""
        print("結果をCSVファイルとして保存中...")
        
        for dataset_name, data in results_data.items():
            results_df = data['results']
            group_scores_df = data['group_scores']
            
            # CSVとして保存（LGBMを含むかどうかでファイル名を変える）
            if self.include_lgbm:
                results_df.to_csv(self.results_dir / f"all_results_df_{dataset_name}.csv", index=False)
                if not group_scores_df.empty:
                    group_scores_df.to_csv(self.results_dir / f"all_group_scores_df_{dataset_name}.csv", index=False)
            else:
                results_df.to_csv(self.results_dir / f"gnn_results_df_{dataset_name}.csv", index=False)
                if not group_scores_df.empty:
                    group_scores_df.to_csv(self.results_dir / f"gnn_group_scores_df_{dataset_name}.csv", index=False)
            
            print(f"  {dataset_name} データセット: CSVファイル保存完了")


def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(
        description='バイオアイソスター予測結果分析・図生成スクリプト',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # デフォルト設定で実行（GNNのみ、出力: test/results/, test/figures/）
  python generate_final_figures.py
  
  # LGBMモデルも含めて実行
  python generate_final_figures.py --include-lgbm
  
  # GNN予測ディレクトリを指定（出力: test/results_consistentsmiles/, test/figures_consistentsmiles/）
  python generate_final_figures.py --gnn-predictions-dir ../pro_GNN/results/results_consistentsmiles
  
  # 複数のGNN予測ディレクトリを指定して統合（同じモデルをマージ）
  python generate_final_figures.py \\
    --gnn-predictions-dir ../pro_GNN/results/results_consistentsmiles ../pro_GNN/results/results_blank \\
    --comparison-mode merge
  
  # 複数のGNN予測ディレクトリを指定して比較（frag vs 通常など）
  python generate_final_figures.py \\
    --gnn-predictions-dir ../pro_GNN/results/results_consistentsmiles_41_frag ../pro_GNN/results/results_consistentsmiles \\
    --comparison-mode compare
  
  # 複数のGNN結果CSVファイルを直接指定して統合（推奨）
  python generate_final_figures.py \\
    --gnn-csv-files test_run1/results/gnn_results_df_single.csv test_run2/results/gnn_results_df_single.csv
  
  # LGBMと複数GNNの両方を含めて実行
  python generate_final_figures.py \\
    --include-lgbm \\
    --lgbm-predictions-dir ../gbdt/results_consistentsmiles-2048 \\
    --gnn-predictions-dir ../pro_GNN/results/results_consistentsmiles ../pro_GNN/results/results_blank \\
    --results-dir ./results \\
    --figures-dir ./figures
  
  # 既存のCSVから図のみ生成（LGBMを含める場合）
  python generate_final_figures.py --skip-calculation --include-lgbm
        """
    )
    
    parser.add_argument(
        '--base-dir', 
        help='ベースディレクトリのパス（指定しない場合は自動検出）'
    )
    
    parser.add_argument(
        '--results-dir',
        help='結果CSVファイルの出力ディレクトリ（指定しない場合は自動生成: results_xxx）'
    )
    
    parser.add_argument(
        '--figures-dir',
        help='図の出力ディレクトリ（指定しない場合は自動生成: figures_xxx）'
    )
    
    parser.add_argument(
        '--lgbm-predictions-dir',
        help='LGBM予測結果のディレクトリ（指定しない場合は gbdt）。'
             '例: gbdt/results_consistentsmiles-2048 (fpsize=2048のモデル結果を直接指定)'
    )
    
    parser.add_argument(
        '--gnn-predictions-dir',
        nargs='+',  # 複数のディレクトリを受け取る
        help='GNN予測結果のディレクトリ（指定しない場合は pro_GNN/results）。'
             '複数指定可能。このディレクトリ名の接尾辞が出力ディレクトリ名に自動適用されます。'
             '例: pro_GNN/results/results_consistentsmiles pro_GNN/results/results_blank'
    )
    
    parser.add_argument(
        '--gnn-csv-files',
        nargs='+',
        help='複数のGNN結果CSVファイルを直接指定（複数GNN統合用）。'
             '例: test/results/gnn_results_df_single.csv test2/results/gnn_results_df_single.csv'
    )
    
    parser.add_argument(
        '--skip-calculation', 
        action='store_true',
        help='結果計算をスキップして、既存のCSVファイルから図のみ生成'
    )
    
    parser.add_argument(
        '--figure-format',
        default='png',
        choices=['png', 'pdf', 'svg', 'eps'],
        help='図の出力形式（デフォルト: png、文字化け防止のためpngを推奨）'
    )
    
    parser.add_argument(
        '--include-lgbm',
        action='store_true',
        help='LGBMモデルの結果も一緒に図を作成する（デフォルト: GNNのみ）'
    )
    
    parser.add_argument(
        '--comparison-mode',
        default='merge',
        choices=['merge', 'compare'],
        help='複数GNN結果の統合方法。'
             'merge: 同じモデルの結果を統合（デフォルト）。'
             'compare: 異なる条件（frag vs 通常など）を比較。'
    )
    
    args = parser.parse_args()
    
    # CSVファイル直接指定とディレクトリ指定の両方がある場合はエラー
    if args.gnn_csv_files and args.gnn_predictions_dir and len(args.gnn_predictions_dir) > 1:
        parser.error("--gnn-csv-files と複数の --gnn-predictions-dir は同時に指定できません")
    
    # アナライザーを初期化
    analyzer = BioisosterePredictionAnalyzer(
        base_dir=args.base_dir,
        results_dir=args.results_dir,
        figures_dir=args.figures_dir,
        lgbm_predictions_dir=args.lgbm_predictions_dir,
        gnn_predictions_dirs=args.gnn_predictions_dir,  # リストとして渡す
        figure_format=args.figure_format,
        include_lgbm=args.include_lgbm,
        comparison_mode=args.comparison_mode
    )
    
    # CSV直接指定の場合はアナライザーに設定
    if args.gnn_csv_files:
        analyzer.gnn_csv_files = [Path(f) for f in args.gnn_csv_files]
    else:
        analyzer.gnn_csv_files = None
    
    # 引数の表示（実際のパスを表示）
    print("=== 設定 ===")
    print(f"ベースディレクトリ: {analyzer.base_dir}")
    print(f"結果出力ディレクトリ: {analyzer.results_dir}")
    print(f"図出力ディレクトリ: {analyzer.figures_dir}")
    print(f"LGBM予測結果ディレクトリ: {analyzer.lgbm_predictions_dir}")
    print(f"GNN予測結果ディレクトリ ({len(analyzer.gnn_predictions_dirs)}個):")
    for i, dir_path in enumerate(analyzer.gnn_predictions_dirs, 1):
        print(f"  {i}. {dir_path}")
    print(f"計算スキップ: {args.skip_calculation}")
    print(f"図の出力形式: {args.figure_format}")
    print(f"LGBMを含める: {args.include_lgbm}")
    print(f"比較モード: {args.comparison_mode}")
    print()
    
    if args.skip_calculation:
        # 図のみ生成（既存のCSVファイルから読み込み）
        print("=== 既存結果から図を生成 ===")
        try:
            # 既存のCSVファイルから結果を読み込み
            results_data = analyzer.load_existing_results()
            analyzer.generate_figures(results_data)
        except Exception as e:
            print(f"エラー: 既存結果の読み込みに失敗しました: {e}")
            print("--skip-calculationオプションを使う場合、結果CSVファイルが存在している必要があります。")
    else:
        # 予測結果から直接分析を実行（results.ipynbの流れに従う）
        analyzer.run_full_analysis()


if __name__ == "__main__":
    main()