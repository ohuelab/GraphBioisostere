#!/usr/bin/env python3
"""
Attention-Structure Analysis for Molecular Pairs

このスクリプトは分子ペアの全データに対して、アテンションが最も強い原子が
共通部分か置換部分かの統計解析を行います。

実行方法not-cv:
    
python analyze_attention_structure.py \
    --model_path ../results/results_consistentsmiles/cv1/pair-cat/model_best.pt \
    --csv_path ../../../unimols/dataset_consistentsmiles.csv \
    --output_dir attention_structure_analysis_new \
    --max_pairs 1000000 \
    --batch_size 100


実行方法cv:
python analyze_attention_structure.py \
    --use_cv \
    --split_file ../../splitting/tid_5cv_consistentsmiles.pkl \
    --model_dir ../results/results_consistentsmiles \
    --model_pattern 'cv{fold}/pair-cat/model_best.pt' \
    --csv_path ../../../MMP_dataset/dataset_consistentsmiles.csv \
    --output_dir attention_structure_analysis_cv \
    --cv_folds 5 \
    --batch_size  64


python analyze_attention_structure.py \
    --use_cv \
    --split_file ../../splitting/tid_5cv_consistentsmiles_molecule.pkl \
    --model_dir ../results/results_consistentsmiles_molecule \
    --model_pattern 'cv{fold}/pair-cat/model_best.pt' \
    --csv_path ../../../MMP_dataset/dataset_consistentsmiles.csv \
    --output_dir attention_structure_analysis_cv_molecule \
    --cv_folds 5 \
    --batch_size  64




=== Attention-Structure Analysis ===
データ: ../../../MMP_dataset/dataset.csv
出力先: attention_structure_analysis_cv
使用デバイス: cuda

=== 交差検証モード ===
分割ファイル: ../../splitting/tid_5cv_consistentsmiles.pkl
モデルディレクトリ: ../results/results_consistentsmiles
モデルパターン: cv{fold}/pair-cat/model_best.pt
  Fold 0: ../results/results_consistentsmiles/cv0/pair-cat/model_best.pt
  Fold 1: ../results/results_consistentsmiles/cv1/pair-cat/model_best.pt
  Fold 2: ../results/results_consistentsmiles/cv2/pair-cat/model_best.pt
  Fold 3: ../results/results_consistentsmiles/cv3/pair-cat/model_best.pt
  Fold 4: ../results/results_consistentsmiles/cv4/pair-cat/model_best.pt
分割情報を読み込み中: ../../splitting/tid_5cv_consistentsmiles.pkl
リスト形式の分割情報を検出しました。辞書形式に変換します。
  Fold 0: train=384455 records, test=87168 records
  Fold 1: train=342719 records, test=172314 records
  Fold 2: train=433772 records, test=128904 records
  Fold 3: train=388386 records, test=81261 records
  Fold 4: train=382479 records, test=174290 records
交差検証フォールド数: 5
データセットを読み込み中...
データセット全体: 779081 行

=== Fold 0 の処理開始 ===
Testデータサイズ: 87168
モデル読み込み中: ../results/results_consistentsmiles/cv0/pair-cat/model_best.pt
Detected number of GNN layers: 2
Detected MLP input dimension: 128
Detected output dimension: 2
Using GraphDiffRegressorCat (concatenation model)
Model loaded successfully (strict=False)
有効なペア数: 87168
"""



import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import argparse
import json
from torch_geometric.data import Data, Batch
from rdkit import Chem
from rdkit import RDLogger
import re
from datetime import datetime

# RDKitのエラーログを抑制（不正なSMILESのパースエラーを表示しない）
RDLogger.DisableLog('rdApp.*')

# 親ディレクトリをPythonパスに追加
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# 外部ファイルのインポート（visualize_attention.pyと共通）
from encoder.gnn_encoder import GNNEncoder
from utils.loader import smiles_to_data
from config import args as default_args

# training_reg_ddp.py からモデル定義をインポート
def make_head(in_dim: int, hidden: int, out_dim: int = 1):
    """回帰用の2-layer MLP ヘッドを生成"""
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden // 2),
        nn.ReLU(),
        nn.Linear(hidden // 2, out_dim)
    )

class GraphDiffRegressor(torch.nn.Module):
    def __init__(self, encoder, embedding_dim: int, loss_type: str, 
                 hidden_dim: int = 64, out_dim: int = 1, merge_method: str = "diff"):
        super().__init__()
        self.loss_type = loss_type
        self.encoder = encoder
        self.merge_method = merge_method
        self.mlp = make_head(embedding_dim, hidden_dim, out_dim)
    
    def forward(self, data_a, data_b):
        fa = self.encoder(data_a)
        fb = self.encoder(data_b)
        
        if self.merge_method == "diff":
            diff_feature = fa - fb
            pred = self.mlp(diff_feature)
        elif self.merge_method == "abs_diff":
            diff_feature = torch.abs(fa - fb)
            pred = self.mlp(diff_feature)
        elif self.merge_method == "product":
            product_feature = fa * fb
            pred = self.mlp(product_feature)
        else:
            raise ValueError(f"Unknown merge_method: {self.merge_method}")
        
        return pred

class GraphDiffRegressorCat(torch.nn.Module):
    def __init__(self, encoder, embedding_dim: int, loss_type: str,
                 hidden_dim: int = 64, out_dim: int = 1):
        super().__init__()
        self.loss_type = loss_type
        self.encoder = encoder
        self.mlp = make_head(embedding_dim * 2, hidden_dim, out_dim)
    
    def forward(self, data_a, data_b):
        fa = self.encoder(data_a)
        fb = self.encoder(data_b)
        concat_feature = torch.cat([fa, fb], dim=1)
        pred = self.mlp(concat_feature)
        return pred

# --- モデルをロードする関数 ---
def load_model(model_path, args):
    """学習済みモデルをロードする"""
    # まず state_dict を読み込んでモデルタイプとレイヤー数を判定
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    
    # チェックポイントから実際のnum_layersを検出
    max_layer_idx = 0
    for key in state_dict.keys():
        if 'encoder.convs.' in key:
            # encoder.convs.0, encoder.convs.1 などからインデックスを抽出
            try:
                layer_idx = int(key.split('encoder.convs.')[1].split('.')[0])
                max_layer_idx = max(max_layer_idx, layer_idx)
            except:
                pass
    
    detected_num_layers = max_layer_idx + 1
    print(f"Detected number of GNN layers: {detected_num_layers}")
    
    # エンコーダーを検出されたレイヤー数で初期化
    encoder = GNNEncoder(
        node_in=args.node_in, edge_in=args.edge_in, hidden_dim=args.hidden_dim,
        out_dim=args.embedding_dim, num_layers=detected_num_layers, dropout=args.dropout
    )
    
    # mlp.0.weight の形状からモデルタイプを判定
    mlp_weight_key = None
    if 'mlp.0.weight' in state_dict:
        mlp_weight_key = 'mlp.0.weight'
    elif 'mlp_ab.0.weight' in state_dict:
        mlp_weight_key = 'mlp_ab.0.weight'
    
    if mlp_weight_key:
        mlp_input_dim = state_dict[mlp_weight_key].shape[1]
        print(f"Detected MLP input dimension: {mlp_input_dim}")
        
        # 最終層の重みから出力次元を検出
        final_layer_key = None
        for key in state_dict.keys():
            if 'mlp' in key and 'weight' in key:
                final_layer_key = key
        
        if final_layer_key:
            out_dim = state_dict[final_layer_key].shape[0]
            print(f"Detected output dimension: {out_dim}")
        else:
            out_dim = getattr(args, "out_dim", 1)
            print(f"Could not detect output dimension, using default: {out_dim}")
        
        if mlp_input_dim == args.embedding_dim * 2:
            print("Using GraphDiffRegressorCat (concatenation model)")
            model = GraphDiffRegressorCat(
                encoder, embedding_dim=args.embedding_dim, loss_type=args.loss_type,
                hidden_dim=getattr(args, "hidden_dim", 64), out_dim=out_dim
            )
        else:
            print("Using GraphDiffRegressor (difference model)")
            model = GraphDiffRegressor(
                encoder, embedding_dim=args.embedding_dim, loss_type=args.loss_type,
                hidden_dim=getattr(args, "hidden_dim", 64), out_dim=out_dim
            )
    else:
        print("Could not determine model type, using GraphDiffRegressor")
        out_dim = getattr(args, "out_dim", 1)
        model = GraphDiffRegressor(
            encoder, embedding_dim=args.embedding_dim, loss_type=args.loss_type,
            hidden_dim=getattr(args, "hidden_dim", 64), out_dim=out_dim
        )
    
    # 不足しているBatchNormキーを適切に処理
    model_state_dict = model.state_dict()
    
    # 不足しているキーを特定
    missing_keys = set(model_state_dict.keys()) - set(state_dict.keys())
    unexpected_keys = set(state_dict.keys()) - set(model_state_dict.keys())
    
    if missing_keys:
        print(f"Warning: Missing keys in state_dict: {missing_keys}")
        # 不足しているキーをモデルの初期値で補完
        for key in missing_keys:
            if key in model_state_dict:
                state_dict[key] = model_state_dict[key]
                print(f"  Initialized missing key: {key}")
    
    if unexpected_keys:
        print(f"Warning: Unexpected keys in state_dict: {unexpected_keys}")
        # 不要なキーを削除（リストを作ってからイテレーション）
        for key in list(unexpected_keys):
            if key in state_dict:
                del state_dict[key]
                print(f"  Removed unexpected key: {key}")
    
    # strict=Falseでロードして、キーの不一致を許容
    try:
        model.load_state_dict(state_dict, strict=False)
        print("Model loaded successfully (strict=False)")
    except Exception as e:
        print(f"Error loading model even with strict=False: {e}")
        # さらに柔軟な読み込み
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print("Model loaded with shape-compatible parameters only")
    
    model.eval()
    return model

# --- 分子の共通部分と置換部分を特定する関数 ---
def is_valid_fragment_smiles(smiles_str):
    """
    フラグメントSMILESが有効かどうかを事前チェック
    
    Args:
        smiles_str: チェックするSMILES文字列
    
    Returns:
        bool: 有効ならTrue、不正ならFalse
    """
    if not smiles_str or len(smiles_str) < 2:
        return False
    
    # 特殊値をチェック
    if smiles_str in ['nan', 'None', '', '*']:
        return False
    
    # 明らかに不正なパターンをチェック
    invalid_patterns = [
        r'\(\)',          # 空のカッコ C()=O など
        r'^\*[A-Z]',      # 先頭にアタッチメントポイント *C など
        r'[A-Z]\(\)[A-Z]', # 原子間の空カッコ C()C など
        r'\(\)=',         # 空カッコと結合 ()= など
        r'=\(\)',         # 結合と空カッコ =() など
    ]
    
    for pattern in invalid_patterns:
        if re.search(pattern, smiles_str):
            return False
    
    return True

def identify_common_substituted_atoms(smiles, common_frag, ref_frag, prb_frag, is_ref_molecule=True):
    """
    SMILESとフラグメント情報から共通部分と置換部分の原子を特定する
    
    Args:
        smiles: 分子のSMILES
        common_frag: 共通フラグメント
        ref_frag: REF分子固有フラグメント
        prb_frag: PRB分子固有フラグメント
        is_ref_molecule: REF分子かどうか
    
    Returns:
        tuple: (common_atom_indices, substituted_atom_indices)
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [], []
        
        num_atoms = mol.GetNumAtoms()
        
        # フラグメント情報を直接利用して共通部分と置換部分を特定
        common_atoms = []
        substituted_atoms = []
        
        # 共通フラグメントが存在する場合の処理
        if common_frag and pd.notna(common_frag) and str(common_frag).strip():
            try:
                # アタッチメントポイント記号を除去
                cleaned_common = re.sub(r'\[\d+\*\]', '', str(common_frag))
                
                # ドット区切りで分割して各フラグメントを処理
                for frag_part in cleaned_common.split('.'):
                    frag_part = frag_part.strip()
                    
                    # 長さチェック
                    if not frag_part or len(frag_part) <= 1:
                        continue
                    
                    # 不正なフラグメントSMILESを事前にフィルタリング
                    if not is_valid_fragment_smiles(frag_part):
                        continue
                    
                    try:
                        frag_mol = Chem.MolFromSmiles(frag_part)
                        if frag_mol and frag_mol.GetNumAtoms() > 0:
                            matches = mol.GetSubstructMatches(frag_mol)
                            for match in matches:
                                common_atoms.extend(match)
                    except:
                        # RDKitのパースエラーは静かにスキップ
                        continue
            except:
                pass
        
        # 置換部分を直接フラグメント情報から取得
        # REF分子ならREF-FRAG、PRB分子ならPRB-FRAGを使用
        substituted_frag = ref_frag if is_ref_molecule else prb_frag
        
        if substituted_frag and pd.notna(substituted_frag) and str(substituted_frag).strip():
            try:
                # アタッチメントポイント記号を除去
                cleaned_substituted = re.sub(r'\[\d+\*\]', '', str(substituted_frag))
                
                # ドット区切りで分割して各フラグメントを処理
                for frag_part in cleaned_substituted.split('.'):
                    frag_part = frag_part.strip()
                    
                    # 長さチェック
                    if not frag_part or len(frag_part) <= 1:
                        continue
                    
                    # 不正なフラグメントSMILESを事前にフィルタリング
                    if not is_valid_fragment_smiles(frag_part):
                        continue
                    
                    try:
                        frag_mol = Chem.MolFromSmiles(frag_part)
                        if frag_mol and frag_mol.GetNumAtoms() > 0:
                            matches = mol.GetSubstructMatches(frag_mol)
                            for match in matches:
                                substituted_atoms.extend(match)
                    except:
                        # RDKitのパースエラーは静かにスキップ
                        continue
            except:
                pass
        
        # 重複を除去してsetに変換
        common_atoms_set = set(common_atoms)
        substituted_atoms_set = set(substituted_atoms)
        
        # もし置換部分が検出できなかった場合、全原子から共通部分を除外したものを置換部分とする
        if not substituted_atoms_set:
            all_atoms = set(range(num_atoms))
            substituted_atoms_set = all_atoms - common_atoms_set
        
        return list(common_atoms_set), list(substituted_atoms_set)
        
    except Exception as e:
        # エラーの場合は全原子を置換部分として扱う（不明として処理）
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                return [], list(range(mol.GetNumAtoms()))
        except:
            pass
        return [], []

def analyze_attention_vs_structure(smiles, attention_weights, detailed_info, is_ref_molecule=True):
    """
    アテンション重みと分子構造の関係を解析する
    
    Args:
        smiles: 分子のSMILES
        attention_weights: アテンション重み
        detailed_info: 詳細情報（フラグメント情報等）
        is_ref_molecule: REF分子かどうか
    
    Returns:
        dict: 解析結果
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        num_atoms = mol.GetNumAtoms()
        
        # アテンション重みの処理
        if attention_weights is None:
            return None
            
        w = attention_weights
        if hasattr(w, "detach"):
            w = w.detach()
        try:
            w = w.cpu().numpy()
        except:
            w = np.asarray(w)
        w = np.squeeze(w)
        
        # 原子数に合わせる
        if w.ndim == 0:
            w = np.full((num_atoms,), float(w))
        if w.shape[0] < num_atoms:
            w = np.pad(w, (0, num_atoms - w.shape[0]), constant_values=0.0)
        elif w.shape[0] > num_atoms:
            w = w[:num_atoms]
            
        w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0).astype(float)
        
        # 共通・置換部分の特定
        common_frag = detailed_info.get('COMMON_FRAG', '')
        ref_frag = detailed_info.get('REF-FRAG', '')
        prb_frag = detailed_info.get('PRB-FRAG', '')
        
        common_atoms, substituted_atoms = identify_common_substituted_atoms(
            smiles, common_frag, ref_frag, prb_frag, is_ref_molecule
        )
        
        # 最大アテンション原子を特定
        max_attention_idx = np.argmax(w)
        max_attention_value = w[max_attention_idx]
        
        # 最大アテンション原子が共通部分か置換部分かを判定
        is_max_common = max_attention_idx in common_atoms
        is_max_substituted = max_attention_idx in substituted_atoms
        
        # 共通部分と置換部分のアテンション統計
        common_attention = w[common_atoms] if common_atoms else np.array([])
        substituted_attention = w[substituted_atoms] if substituted_atoms else np.array([])
        
        result = {
            'num_atoms': num_atoms,
            'max_attention_idx': int(max_attention_idx),
            'max_attention_value': float(max_attention_value),
            'is_max_common': is_max_common,
            'is_max_substituted': is_max_substituted,
            'num_common_atoms': len(common_atoms),
            'num_substituted_atoms': len(substituted_atoms),
            'common_atoms': common_atoms,
            'substituted_atoms': substituted_atoms,
            'common_attention_mean': float(common_attention.mean()) if len(common_attention) > 0 else 0.0,
            'substituted_attention_mean': float(substituted_attention.mean()) if len(substituted_attention) > 0 else 0.0,
            'common_attention_max': float(common_attention.max()) if len(common_attention) > 0 else 0.0,
            'substituted_attention_max': float(substituted_attention.max()) if len(substituted_attention) > 0 else 0.0,
            'common_attention_std': float(common_attention.std()) if len(common_attention) > 0 else 0.0,
            'substituted_attention_std': float(substituted_attention.std()) if len(substituted_attention) > 0 else 0.0,
        }
        
        return result
        
    except Exception as e:
        print(f"構造解析でエラー: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_attention_statistics_report(all_analyses, output_dir):
    """アテンションと構造の関係の統計レポートを作成する"""
    report_path = os.path.join(output_dir, "attention_structure_statistics.txt")
    csv_path = os.path.join(output_dir, "attention_statistics_detailed.csv")
    summary_csv_path = os.path.join(output_dir, "attention_statistics_summary.csv")
    
    try:
        # 統計データを収集
        ref_max_common = 0
        ref_max_substituted = 0
        ref_max_unknown = 0
        prb_max_common = 0
        prb_max_substituted = 0
        prb_max_unknown = 0
        
        # CSV用データ
        csv_data = []
        
        # ラベル別統計
        bioisostere_ref_common = 0
        bioisostere_ref_substituted = 0
        nonbioisostere_ref_common = 0
        nonbioisostere_ref_substituted = 0
        
        bioisostere_prb_common = 0
        bioisostere_prb_substituted = 0
        nonbioisostere_prb_common = 0
        nonbioisostere_prb_substituted = 0
        
        # 予測精度用の統計（8カテゴリ）
        accuracy_stats = {
            'bioisostere_ref_common': {'correct': 0, 'total': 0},
            'bioisostere_ref_substituted': {'correct': 0, 'total': 0},
            'nonbioisostere_ref_common': {'correct': 0, 'total': 0},
            'nonbioisostere_ref_substituted': {'correct': 0, 'total': 0},
            'bioisostere_prb_common': {'correct': 0, 'total': 0},
            'bioisostere_prb_substituted': {'correct': 0, 'total': 0},
            'nonbioisostere_prb_common': {'correct': 0, 'total': 0},
            'nonbioisostere_prb_substituted': {'correct': 0, 'total': 0},
        }
        
        # 標的別統計
        target_stats = {}
        
        for analysis in all_analyses:
            pair_info = analysis['pair_info']
            ref_analysis = analysis['ref_analysis']
            prb_analysis = analysis['prb_analysis']
            is_bioisostere = pair_info.get('true_label', False)
            target_id = pair_info.get('detailed_info', {}).get('TID', 'Unknown')
            
            # 予測情報
            pred_label = pair_info.get('pred_label', None)
            true_label = int(is_bioisostere)
            is_correct = (pred_label == true_label) if pred_label is not None else None
            
            # 標的別統計の初期化
            if target_id not in target_stats:
                target_stats[target_id] = {
                    'total_pairs': 0,
                    'ref_common': 0, 'ref_substituted': 0,
                    'prb_common': 0, 'prb_substituted': 0,
                    'bioisostere_pairs': 0
                }
            
            target_stats[target_id]['total_pairs'] += 1
            if is_bioisostere:
                target_stats[target_id]['bioisostere_pairs'] += 1
            
            # CSV行データ
            csv_row = {
                'pair_index': analysis['pair_index'],
                'is_bioisostere': is_bioisostere,
                'pred_value': pair_info.get('pred_value', None),
                'pred_label': pred_label,
                'is_correct': is_correct,
                'ref_cid': pair_info.get('detailed_info', {}).get('REF-CID', ''),
                'prb_cid': pair_info.get('detailed_info', {}).get('PRB-CID', ''),
                'target_id': target_id,
                'assay_id': pair_info.get('detailed_info', {}).get('AID', ''),
                'delta_value': pair_info.get('detailed_info', {}).get('delta_value', ''),
                'standard_type': pair_info.get('detailed_info', {}).get('STANDARD_TYPE', ''),
            }
            
            # REF分子の統計
            if ref_analysis:
                if ref_analysis['is_max_common']:
                    ref_max_common += 1
                    target_stats[target_id]['ref_common'] += 1
                    if is_bioisostere:
                        bioisostere_ref_common += 1
                        if is_correct is not None:
                            accuracy_stats['bioisostere_ref_common']['total'] += 1
                            if is_correct:
                                accuracy_stats['bioisostere_ref_common']['correct'] += 1
                    else:
                        nonbioisostere_ref_common += 1
                        if is_correct is not None:
                            accuracy_stats['nonbioisostere_ref_common']['total'] += 1
                            if is_correct:
                                accuracy_stats['nonbioisostere_ref_common']['correct'] += 1
                elif ref_analysis['is_max_substituted']:
                    ref_max_substituted += 1
                    target_stats[target_id]['ref_substituted'] += 1
                    if is_bioisostere:
                        bioisostere_ref_substituted += 1
                        if is_correct is not None:
                            accuracy_stats['bioisostere_ref_substituted']['total'] += 1
                            if is_correct:
                                accuracy_stats['bioisostere_ref_substituted']['correct'] += 1
                    else:
                        nonbioisostere_ref_substituted += 1
                        if is_correct is not None:
                            accuracy_stats['nonbioisostere_ref_substituted']['total'] += 1
                            if is_correct:
                                accuracy_stats['nonbioisostere_ref_substituted']['correct'] += 1
                else:
                    ref_max_unknown += 1
                
                csv_row.update({
                    'ref_max_attention_idx': ref_analysis['max_attention_idx'],
                    'ref_max_attention_value': ref_analysis['max_attention_value'],
                    'ref_is_max_common': ref_analysis['is_max_common'],
                    'ref_is_max_substituted': ref_analysis['is_max_substituted'],
                    'ref_num_common_atoms': ref_analysis['num_common_atoms'],
                    'ref_num_substituted_atoms': ref_analysis['num_substituted_atoms'],
                    'ref_common_attention_mean': ref_analysis['common_attention_mean'],
                    'ref_substituted_attention_mean': ref_analysis['substituted_attention_mean'],
                    'ref_common_attention_std': ref_analysis['common_attention_std'],
                    'ref_substituted_attention_std': ref_analysis['substituted_attention_std'],
                })
            
            # PRB分子の統計
            if prb_analysis:
                if prb_analysis['is_max_common']:
                    prb_max_common += 1
                    target_stats[target_id]['prb_common'] += 1
                    if is_bioisostere:
                        bioisostere_prb_common += 1
                        if is_correct is not None:
                            accuracy_stats['bioisostere_prb_common']['total'] += 1
                            if is_correct:
                                accuracy_stats['bioisostere_prb_common']['correct'] += 1
                    else:
                        nonbioisostere_prb_common += 1
                        if is_correct is not None:
                            accuracy_stats['nonbioisostere_prb_common']['total'] += 1
                            if is_correct:
                                accuracy_stats['nonbioisostere_prb_common']['correct'] += 1
                elif prb_analysis['is_max_substituted']:
                    prb_max_substituted += 1
                    target_stats[target_id]['prb_substituted'] += 1
                    if is_bioisostere:
                        bioisostere_prb_substituted += 1
                        if is_correct is not None:
                            accuracy_stats['bioisostere_prb_substituted']['total'] += 1
                            if is_correct:
                                accuracy_stats['bioisostere_prb_substituted']['correct'] += 1
                    else:
                        nonbioisostere_prb_substituted += 1
                        if is_correct is not None:
                            accuracy_stats['nonbioisostere_prb_substituted']['total'] += 1
                            if is_correct:
                                accuracy_stats['nonbioisostere_prb_substituted']['correct'] += 1
                else:
                    prb_max_unknown += 1
                
                csv_row.update({
                    'prb_max_attention_idx': prb_analysis['max_attention_idx'],
                    'prb_max_attention_value': prb_analysis['max_attention_value'],
                    'prb_is_max_common': prb_analysis['is_max_common'],
                    'prb_is_max_substituted': prb_analysis['is_max_substituted'],
                    'prb_num_common_atoms': prb_analysis['num_common_atoms'],
                    'prb_num_substituted_atoms': prb_analysis['num_substituted_atoms'],
                    'prb_common_attention_mean': prb_analysis['common_attention_mean'],
                    'prb_substituted_attention_mean': prb_analysis['substituted_attention_mean'],
                    'prb_common_attention_std': prb_analysis['common_attention_std'],
                    'prb_substituted_attention_std': prb_analysis['substituted_attention_std'],
                })
            
            csv_data.append(csv_row)
        
        # テキストレポート作成
        with open(report_path, "w", encoding='utf-8') as f:
            f.write("=== アテンションと分子構造の関係 統計レポート ===\n\n")
            
            total_ref = ref_max_common + ref_max_substituted + ref_max_unknown
            total_prb = prb_max_common + prb_max_substituted + prb_max_unknown
            
            f.write("【基本統計】\n")
            f.write(f"解析成功ペア数: {len(all_analyses)}\n")
            f.write(f"解析対象標的数: {len(target_stats)}\n\n")
            
            f.write("【全体統計: 最大アテンション原子の位置】\n")
            if total_ref > 0:
                f.write(f"REF分子:\n")
                f.write(f"  共通部分: {ref_max_common} ({ref_max_common/total_ref*100:.1f}%)\n")
                f.write(f"  置換部分: {ref_max_substituted} ({ref_max_substituted/total_ref*100:.1f}%)\n")
                f.write(f"  不明/その他: {ref_max_unknown} ({ref_max_unknown/total_ref*100:.1f}%)\n")
            
            if total_prb > 0:
                f.write(f"PRB分子:\n")
                f.write(f"  共通部分: {prb_max_common} ({prb_max_common/total_prb*100:.1f}%)\n")
                f.write(f"  置換部分: {prb_max_substituted} ({prb_max_substituted/total_prb*100:.1f}%)\n")
                f.write(f"  不明/その他: {prb_max_unknown} ({prb_max_unknown/total_prb*100:.1f}%)\n")
            f.write("\n")
            
            # ラベル別統計
            f.write("【Bioisostereペア vs Non-Bioisostereペアの比較】\n")
            bioisostere_total_ref = bioisostere_ref_common + bioisostere_ref_substituted
            nonbioisostere_total_ref = nonbioisostere_ref_common + nonbioisostere_ref_substituted
            
            f.write("REF分子:\n")
            if bioisostere_total_ref > 0:
                f.write(f"  Bioisostereペア (n={bioisostere_total_ref}):\n")
                f.write(f"    共通部分: {bioisostere_ref_common} ({bioisostere_ref_common/bioisostere_total_ref*100:.1f}%)\n")
                f.write(f"    置換部分: {bioisostere_ref_substituted} ({bioisostere_ref_substituted/bioisostere_total_ref*100:.1f}%)\n")
            if nonbioisostere_total_ref > 0:
                f.write(f"  Non-Bioisostereペア (n={nonbioisostere_total_ref}):\n")
                f.write(f"    共通部分: {nonbioisostere_ref_common} ({nonbioisostere_ref_common/nonbioisostere_total_ref*100:.1f}%)\n")
                f.write(f"    置換部分: {nonbioisostere_ref_substituted} ({nonbioisostere_ref_substituted/nonbioisostere_total_ref*100:.1f}%)\n")
            
            bioisostere_total_prb = bioisostere_prb_common + bioisostere_prb_substituted
            nonbioisostere_total_prb = nonbioisostere_prb_common + nonbioisostere_prb_substituted
            
            f.write("PRB分子:\n")
            if bioisostere_total_prb > 0:
                f.write(f"  Bioisostereペア (n={bioisostere_total_prb}):\n")
                f.write(f"    共通部分: {bioisostere_prb_common} ({bioisostere_prb_common/bioisostere_total_prb*100:.1f}%)\n")
                f.write(f"    置換部分: {bioisostere_prb_substituted} ({bioisostere_prb_substituted/bioisostere_total_prb*100:.1f}%)\n")
            if nonbioisostere_total_prb > 0:
                f.write(f"  Non-Bioisostereペア (n={nonbioisostere_total_prb}):\n")
                f.write(f"    共通部分: {nonbioisostere_prb_common} ({nonbioisostere_prb_common/nonbioisostere_total_prb*100:.1f}%)\n")
                f.write(f"    置換部分: {nonbioisostere_prb_substituted} ({nonbioisostere_prb_substituted/nonbioisostere_total_prb*100:.1f}%)\n")
            f.write("\n")
            
            # 予測精度の統計
            f.write("【予測精度accuracy: アテンション位置 × ラベル別】\n")
            f.write("REF分子:\n")
            for key in ['bioisostere_ref_common', 'bioisostere_ref_substituted', 
                       'nonbioisostere_ref_common', 'nonbioisostere_ref_substituted']:
                stats = accuracy_stats[key]
                if stats['total'] > 0:
                    acc = stats['correct'] / stats['total'] * 100
                    label_type = "Bioisostere" if "bioisostere" in key else "Non-Bioisostere"
                    attn_type = "共通部分" if "common" in key else "置換部分"
                    f.write(f"  {label_type} × {attn_type}: {stats['correct']}/{stats['total']} ({acc:.1f}%)\n")
            
            f.write("PRB分子:\n")
            for key in ['bioisostere_prb_common', 'bioisostere_prb_substituted',
                       'nonbioisostere_prb_common', 'nonbioisostere_prb_substituted']:
                stats = accuracy_stats[key]
                if stats['total'] > 0:
                    acc = stats['correct'] / stats['total'] * 100
                    label_type = "Bioisostere" if "bioisostere" in key else "Non-Bioisostere"
                    attn_type = "共通部分" if "common" in key else "置換部分"
                    f.write(f"  {label_type} × {attn_type}: {stats['correct']}/{stats['total']} ({acc:.1f}%)\n")
            f.write("\n")
            
            # 標的別統計（上位10標的）
            f.write("【標的別統計（上位10標的）】\n")
            sorted_targets = sorted(target_stats.items(), key=lambda x: x[1]['total_pairs'], reverse=True)[:10]
            for target_id, stats in sorted_targets:
                f.write(f"標的 {target_id} (n={stats['total_pairs']}):\n")
                total_target_ref = stats['ref_common'] + stats['ref_substituted']
                total_target_prb = stats['prb_common'] + stats['prb_substituted']
                
                if total_target_ref > 0:
                    f.write(f"  REF: 共通部分 {stats['ref_common']}/{total_target_ref} ({stats['ref_common']/total_target_ref*100:.1f}%)\n")
                if total_target_prb > 0:
                    f.write(f"  PRB: 共通部分 {stats['prb_common']}/{total_target_prb} ({stats['prb_common']/total_target_prb*100:.1f}%)\n")
                f.write(f"  Bioisostereペア率: {stats['bioisostere_pairs']}/{stats['total_pairs']} ({stats['bioisostere_pairs']/stats['total_pairs']*100:.1f}%)\n")
                f.write("\n")
            
            f.write(f"レポート生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # 詳細CSVファイル作成
        df_detailed = pd.DataFrame(csv_data)
        df_detailed.to_csv(csv_path, index=False, encoding='utf-8')
        
        # 予測精度の詳細CSVファイル作成
        accuracy_csv_path = os.path.join(output_dir, "attention_accuracy_analysis.csv")
        accuracy_data = []
        for key, stats in accuracy_stats.items():
            if stats['total'] > 0:
                parts = key.split('_')
                label_type = "bioisostere" if parts[0] == "bioisostere" else "non-bioisostere"
                molecule = parts[1].upper()  # REF or PRB
                attention_pos = "common" if parts[2] == "common" else "substituted"
                accuracy = stats['correct'] / stats['total']
                
                accuracy_data.append({
                    'molecule': molecule,
                    'attention_position': attention_pos,
                    'label_type': label_type,
                    'correct': stats['correct'],
                    'total': stats['total'],
                    'accuracy': accuracy
                })
        
        df_accuracy = pd.DataFrame(accuracy_data)
        df_accuracy.to_csv(accuracy_csv_path, index=False, encoding='utf-8')
        print(f"予測精度解析CSVを作成しました: {accuracy_csv_path}")
        
        # サマリーCSVファイル作成
        summary_data = []
        for target_id, stats in target_stats.items():
            total_ref = stats['ref_common'] + stats['ref_substituted']
            total_prb = stats['prb_common'] + stats['prb_substituted']
            
            summary_row = {
                'target_id': target_id,
                'total_pairs': stats['total_pairs'],
                'bioisostere_pairs': stats['bioisostere_pairs'],
                'bioisostere_ratio': stats['bioisostere_pairs'] / stats['total_pairs'] if stats['total_pairs'] > 0 else 0,
                'ref_common_count': stats['ref_common'],
                'ref_substituted_count': stats['ref_substituted'],
                'ref_common_ratio': stats['ref_common'] / total_ref if total_ref > 0 else 0,
                'prb_common_count': stats['prb_common'],
                'prb_substituted_count': stats['prb_substituted'],
                'prb_common_ratio': stats['prb_common'] / total_prb if total_prb > 0 else 0,
            }
            summary_data.append(summary_row)
        
        df_summary = pd.DataFrame(summary_data)
        df_summary = df_summary.sort_values('total_pairs', ascending=False)
        df_summary.to_csv(summary_csv_path, index=False, encoding='utf-8')
        
        print(f"統計レポートを作成しました: {report_path}")
        print(f"詳細データCSVを作成しました: {csv_path}")
        print(f"サマリーCSVを作成しました: {summary_csv_path}")
        
        # 簡易統計を表示
        print("\n=== 主要な発見 ===")
        if total_ref > 0:
            ref_common_pct = ref_max_common/total_ref*100
            print(f"REF分子: {ref_common_pct:.1f}% のペアで最大アテンションが共通部分に位置")
        if total_prb > 0:
            prb_common_pct = prb_max_common/total_prb*100
            print(f"PRB分子: {prb_common_pct:.1f}% のペアで最大アテンションが共通部分に位置")
        
        if bioisostere_total_ref > 0 and nonbioisostere_total_ref > 0:
            bioisostere_common_pct = bioisostere_ref_common/bioisostere_total_ref*100
            nonbioisostere_common_pct = nonbioisostere_ref_common/nonbioisostere_total_ref*100
            print(f"REF分子: Bioisostereペア {bioisostere_common_pct:.1f}% vs Non-Bioisostereペア {nonbioisostere_common_pct:.1f}% が共通部分に最大アテンション")
        
    except Exception as e:
        print(f"統計レポート作成でエラー: {e}")
        import traceback
        traceback.print_exc()

def load_cv_splits(split_file_path):
    """
    交差検証の分割情報を読み込む
    
    Args:
        split_file_path: 分割情報ファイルのパス (pkl形式)
        data_splitting.pyで生成されたファイルは以下の形式:
        [(train_records, val_records, test_records), ...]
    
    Returns:
        dict: {fold_id: {'train': train_indices, 'test': test_indices}}
    """
    import pickle
    
    print(f"分割情報を読み込み中: {split_file_path}")
    with open(split_file_path, 'rb') as f:
        splits = pickle.load(f)
    
    # data_splitting.pyの形式: [(train, val, test), ...]
    if isinstance(splits, list):
        print(f"リスト形式の分割情報を検出しました。辞書形式に変換します。")
        splits_dict = {}
        for fold_id, fold_data in enumerate(splits):
            # 3要素タプル (train_records, val_records, test_records) の場合
            if isinstance(fold_data, tuple) and len(fold_data) == 3:
                train_records, val_records, test_records = fold_data
                # レコードからインデックスを抽出
                train_indices = [r['index'] for r in train_records]
                test_indices = [r['index'] for r in test_records]
                splits_dict[fold_id] = {
                    'train': train_indices,
                    'test': test_indices
                }
                print(f"  Fold {fold_id}: train={len(train_indices)} records, test={len(test_indices)} records")
            # 2要素タプル (train_indices, test_indices) の場合
            elif isinstance(fold_data, tuple) and len(fold_data) == 2:
                splits_dict[fold_id] = {
                    'train': fold_data[0],
                    'test': fold_data[1]
                }
                print(f"  Fold {fold_id}: train={len(fold_data[0])} indices, test={len(fold_data[1])} indices")
            # 既に辞書形式の場合
            elif isinstance(fold_data, dict):
                splits_dict[fold_id] = fold_data
                print(f"  Fold {fold_id}: train={len(fold_data.get('train', []))}, test={len(fold_data.get('test', []))}")
            else:
                raise ValueError(f"Fold {fold_id} のデータ形式が不正です: {type(fold_data)}, 要素数={len(fold_data) if hasattr(fold_data, '__len__') else 'N/A'}")
        splits = splits_dict
    
    print(f"交差検証フォールド数: {len(splits)}")
    
    return splits

def process_cv_dataset(model_paths, csv_path, split_file, device, output_dir, batch_size=100, target_id=None, target_col='TID'):
    """
    交差検証の各foldのtestデータのみを使って全データを解析する
    
    Args:
        model_paths: dict {fold_id: model_path}
        csv_path: データセットのCSVパス
        split_file: 分割情報ファイルのパス
        device: 使用デバイス
        output_dir: 出力ディレクトリ
        batch_size: バッチサイズ
        target_id: 標的ID（オプション）
        target_col: 標的IDのカラム名
    """
    # 分割情報を読み込み
    cv_splits = load_cv_splits(split_file)
    
    # CSVファイルを読み込み
    print("データセットを読み込み中...")
    df = pd.read_csv(csv_path)
    print(f"データセット全体: {len(df)} 行")
    
    # 標的指定がある場合の処理
    if target_id is not None:
        if target_col not in df.columns:
            raise ValueError(f"標的指定されましたが、カラム '{target_col}' が見つかりません。")
        
        df_filtered = df[df[target_col] == target_id]
        print(f"標的 '{target_id}' でフィルタリング: {len(df_filtered)} 行")
        
        if len(df_filtered) == 0:
            available_targets = df[target_col].dropna().unique()
            raise ValueError(f"標的 '{target_id}' のデータが見つかりません。\n利用可能な標的: {list(available_targets)}")
        
        df = df_filtered
    
    all_analyses = []
    
    # 各foldのtestデータを解析
    for fold_id, fold_data in cv_splits.items():
        test_indices = fold_data['test']
        
        print(f"\n=== Fold {fold_id} の処理開始 ===")
        print(f"Testデータサイズ: {len(test_indices)}")
        
        # 該当foldのモデルをロード
        if isinstance(model_paths, dict):
            if fold_id not in model_paths:
                print(f"警告: Fold {fold_id} のモデルが見つかりません。スキップします。")
                continue
            model_path = model_paths[fold_id]
        else:
            # 単一モデルの場合は全foldで同じモデルを使用
            model_path = model_paths
        
        print(f"モデル読み込み中: {model_path}")
        model = load_model(model_path, default_args).to(device)
        
        # testデータのみを抽出
        df_test = df.iloc[test_indices].copy()
        
        # process_single_datasetの処理を実行
        fold_analyses = process_single_dataset(
            model, df_test, device, batch_size=batch_size
        )
        
        print(f"Fold {fold_id}: {len(fold_analyses)} ペアを解析")
        all_analyses.extend(fold_analyses)
    
    print(f"\n=== 全fold処理完了 ===")
    print(f"総解析ペア数: {len(all_analyses)}")
    
    # 統計レポートを作成
    if output_dir and all_analyses:
        os.makedirs(output_dir, exist_ok=True)
        create_attention_statistics_report(all_analyses, output_dir)
    
    return all_analyses

def process_single_dataset(model, df, device, batch_size=100):
    """
    単一データセットを処理する（process_large_datasetから分離）
    
    Args:
        model: 学習済みモデル
        df: 処理対象のDataFrame
        device: 使用デバイス
        batch_size: バッチサイズ
    
    Returns:
        list: 解析結果のリスト
    """
    # 必要なカラムの存在確認
    required_columns = ['REF-SMILES', 'PRB-SMILES', 'label_bin']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"必要なカラム '{col}' が見つかりません")
    
    # NaNを含む行を除外
    df_clean = df.dropna(subset=required_columns)
    print(f"有効なペア数: {len(df_clean)}")
    
    # 詳細情報を取得するための定義済みカラム
    detail_columns = [
        'REF-CID', 'PRB-CID', 'AID', 'TID', 'CUT_NUM', 'COMMON_FRAG', 
        'REF-FRAG', 'PRB-FRAG', 'SMIRKS', 'REF-standard_value', 
        'PRB-standard_value', 'STANDARD_TYPE', 'delta_value'
    ]
    
    all_analyses = []
    processed_count = 0
    error_count = 0
    
    # プログレスバー用
    total_pairs = len(df_clean)
    
    # バッチ処理
    for batch_start in range(0, len(df_clean), batch_size):
        batch_end = min(batch_start + batch_size, len(df_clean))
        batch_df = df_clean.iloc[batch_start:batch_end]
        
        print(f"バッチ処理中: {batch_start+1}-{batch_end} / {total_pairs} ({(batch_end/total_pairs)*100:.1f}%)")
        
        # バッチ内の各ペアを処理
        for idx, row in batch_df.iterrows():
            try:
                pair_info = {
                    'index': idx,
                    'ref_smiles': row['REF-SMILES'],
                    'prb_smiles': row['PRB-SMILES'],
                    'true_label': row['label_bin']
                }
                
                # 詳細情報を追加
                detailed_info = {}
                for col in detail_columns:
                    if col in df.columns and pd.notna(row[col]):
                        detailed_info[col] = row[col]
                
                pair_info['detailed_info'] = detailed_info
                
                # 分子データの作成
                data1 = smiles_to_data(pair_info['ref_smiles'])
                data2 = smiles_to_data(pair_info['prb_smiles'])
                
                if data1 is None or data2 is None:
                    error_count += 1
                    continue
                
                batch1 = Batch.from_data_list([data1]).to(device)
                batch2 = Batch.from_data_list([data2]).to(device)
                
                with torch.no_grad():
                    # エンコーダーを実行してアテンション重みを取得
                    model.encoder(batch1)
                    attn_weights1 = model.encoder.attention_weights
                    
                    model.encoder(batch2)
                    attn_weights2 = model.encoder.attention_weights
                    
                    # モデルの予測値を取得
                    pred_output = model(batch1, batch2)
                    if pred_output.dim() > 1 and pred_output.shape[1] > 1:
                        # 分類問題の場合（2クラス以上）
                        pred_probs = torch.softmax(pred_output, dim=1)
                        pred_label = torch.argmax(pred_probs, dim=1).item()
                        pred_value = pred_probs[0, 1].item()  # クラス1の確率
                    else:
                        # 回帰問題または単一出力の場合
                        pred_value = pred_output.item()
                        pred_label = 1 if pred_value > 0.5 else 0
                    
                    pair_info['pred_value'] = pred_value
                    pair_info['pred_label'] = pred_label
                
                # 構造解析を実行
                detailed_info_extended = pair_info['detailed_info'].copy()
                detailed_info_extended['REF-SMILES'] = pair_info['ref_smiles']
                detailed_info_extended['PRB-SMILES'] = pair_info['prb_smiles']
                
                ref_analysis = analyze_attention_vs_structure(
                    pair_info['ref_smiles'], attn_weights1, detailed_info_extended, is_ref_molecule=True
                )
                prb_analysis = analyze_attention_vs_structure(
                    pair_info['prb_smiles'], attn_weights2, detailed_info_extended, is_ref_molecule=False
                )
                
                if ref_analysis or prb_analysis:
                    analysis_data = {
                        'pair_index': pair_info['index'],
                        'pair_info': pair_info,
                        'ref_analysis': ref_analysis,
                        'prb_analysis': prb_analysis
                    }
                    all_analyses.append(analysis_data)
                    processed_count += 1
                else:
                    error_count += 1
                
            except Exception as e:
                print(f"ペア {idx} 処理でエラー: {e}")
                error_count += 1
                continue
        
        # 定期的に進捗を保存
        if len(all_analyses) > 0 and len(all_analyses) % (batch_size * 10) == 0:
            print(f"中間結果: {len(all_analyses)} 件の解析が完了")
    
    print(f"\n処理成功: {processed_count}, 処理失敗: {error_count}")
    
    return all_analyses

def process_large_dataset(model, csv_path, device, output_dir, batch_size=100, max_pairs=None, target_id=None, target_col='TID'):
    """大量データを効率的に処理する（単一モデル用）"""
    print(f"大量データ処理を開始します...")
    
    # CSVファイルを読み込み
    print("データセットを読み込み中...")
    df = pd.read_csv(csv_path)
    print(f"データセット全体: {len(df)} 行")
    
    # 標的指定がある場合の処理
    if target_id is not None:
        if target_col not in df.columns:
            raise ValueError(f"標的指定されましたが、カラム '{target_col}' が見つかりません。")
        
        df_filtered = df[df[target_col] == target_id]
        print(f"標的 '{target_id}' でフィルタリング: {len(df_filtered)} 行")
        
        if len(df_filtered) == 0:
            available_targets = df[target_col].dropna().unique()
            raise ValueError(f"標的 '{target_id}' のデータが見つかりません。\n利用可能な標的: {list(available_targets)}")
        
        df = df_filtered
    
    if max_pairs:
        df = df.head(max_pairs)
        print(f"処理対象を {max_pairs} ペアに制限します")
    
    # 必要なカラムの存在確認
    required_columns = ['REF-SMILES', 'PRB-SMILES', 'label_bin']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"必要なカラム '{col}' が見つかりません")
    
    # NaNを含む行を除外
    df_clean = df.dropna(subset=required_columns)
    print(f"有効なペア数: {len(df_clean)}")
    
    # 詳細情報を取得するための定義済みカラム
    detail_columns = [
        'REF-CID', 'PRB-CID', 'AID', 'TID', 'CUT_NUM', 'COMMON_FRAG', 
        'REF-FRAG', 'PRB-FRAG', 'SMIRKS', 'REF-standard_value', 
        'PRB-standard_value', 'STANDARD_TYPE', 'delta_value'
    ]
    
    all_analyses = []
    processed_count = 0
    error_count = 0
    
    # プログレスバー用
    total_pairs = len(df_clean)
    
    # バッチ処理
    for batch_start in range(0, len(df_clean), batch_size):
        batch_end = min(batch_start + batch_size, len(df_clean))
        batch_df = df_clean.iloc[batch_start:batch_end]
        
        print(f"バッチ処理中: {batch_start+1}-{batch_end} / {total_pairs} ({(batch_end/total_pairs)*100:.1f}%)")
        
        # バッチ内の各ペアを処理
        for idx, row in batch_df.iterrows():
            try:
                pair_info = {
                    'index': idx,
                    'ref_smiles': row['REF-SMILES'],
                    'prb_smiles': row['PRB-SMILES'],
                    'true_label': row['label_bin']
                }
                
                # 詳細情報を追加
                detailed_info = {}
                for col in detail_columns:
                    if col in df.columns and pd.notna(row[col]):
                        detailed_info[col] = row[col]
                
                pair_info['detailed_info'] = detailed_info
                
                # 分子データの作成
                data1 = smiles_to_data(pair_info['ref_smiles'])
                data2 = smiles_to_data(pair_info['prb_smiles'])
                
                if data1 is None or data2 is None:
                    error_count += 1
                    continue
                
                batch1 = Batch.from_data_list([data1]).to(device)
                batch2 = Batch.from_data_list([data2]).to(device)
                
                with torch.no_grad():
                    # エンコーダーを実行してアテンション重みを取得
                    model.encoder(batch1)
                    attn_weights1 = model.encoder.attention_weights
                    
                    model.encoder(batch2)
                    attn_weights2 = model.encoder.attention_weights
                    
                    # モデルの予測値を取得
                    pred_output = model(batch1, batch2)
                    if pred_output.dim() > 1 and pred_output.shape[1] > 1:
                        # 分類問題の場合（2クラス以上）
                        pred_probs = torch.softmax(pred_output, dim=1)
                        pred_label = torch.argmax(pred_probs, dim=1).item()
                        pred_value = pred_probs[0, 1].item()  # クラス1の確率
                    else:
                        # 回帰問題または単一出力の場合
                        pred_value = pred_output.item()
                        pred_label = 1 if pred_value > 0.5 else 0
                    
                    pair_info['pred_value'] = pred_value
                    pair_info['pred_label'] = pred_label
                
                # 構造解析を実行
                detailed_info_extended = pair_info['detailed_info'].copy()
                detailed_info_extended['REF-SMILES'] = pair_info['ref_smiles']
                detailed_info_extended['PRB-SMILES'] = pair_info['prb_smiles']
                
                ref_analysis = analyze_attention_vs_structure(
                    pair_info['ref_smiles'], attn_weights1, detailed_info_extended, is_ref_molecule=True
                )
                prb_analysis = analyze_attention_vs_structure(
                    pair_info['prb_smiles'], attn_weights2, detailed_info_extended, is_ref_molecule=False
                )
                
                if ref_analysis or prb_analysis:
                    analysis_data = {
                        'pair_index': pair_info['index'],
                        'pair_info': pair_info,
                        'ref_analysis': ref_analysis,
                        'prb_analysis': prb_analysis
                    }
                    all_analyses.append(analysis_data)
                    processed_count += 1
                else:
                    error_count += 1
                
            except Exception as e:
                print(f"ペア {idx} 処理でエラー: {e}")
                error_count += 1
                continue
        
        # 定期的に進捗を保存
        if len(all_analyses) > 0 and len(all_analyses) % (batch_size * 10) == 0:
            print(f"中間結果: {len(all_analyses)} 件の解析が完了")
    
    print(f"\n=== 大量データ処理完了 ===")
    print(f"総ペア数: {total_pairs}")
    print(f"処理成功: {processed_count}")
    print(f"処理失敗: {error_count}")
    
    # 統計レポートを作成
    if output_dir and all_analyses:
        os.makedirs(output_dir, exist_ok=True)
        create_attention_statistics_report(all_analyses, output_dir)
    
    return all_analyses

def main():
    parser = argparse.ArgumentParser(description='Analyze attention patterns vs molecular structure for all data')
    parser.add_argument('--model_path', type=str, default=None, help='Path to the trained model (single model mode)')
    parser.add_argument('--model_dir', type=str, default=None, help='Directory containing CV models (fold0/pair-cat/model_best.pt, fold1/..., etc.)')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to CSV file containing molecular pair data')
    parser.add_argument('--output_dir', type=str, default='attention_structure_analysis', 
                       help='Directory to save analysis results')
    parser.add_argument('--max_pairs', type=int, default=None, 
                       help='Maximum number of pairs to process (default: all)')
    parser.add_argument('--batch_size', type=int, default=100, 
                       help='Batch size for processing (default: 100)')
    parser.add_argument('--config', type=str, default=None, help='Path to config file (optional)')
    
    # 交差検証関連の引数
    parser.add_argument('--use_cv', action='store_true', 
                       help='Use cross-validation mode (requires --split_file and --model_dir)')
    parser.add_argument('--split_file', type=str, default=None, 
                       help='Path to CV split file (pkl format, e.g., ../../splitting/tid_5cv_consistentsmiles.pkl)')
    parser.add_argument('--cv_folds', type=int, default=5, 
                       help='Number of CV folds (default: 5)')
    parser.add_argument('--model_pattern', type=str, default='cv{fold}/pair-cat/model_best.pt',
                       help='Model path pattern with {fold} placeholder (default: cv{fold}/pair-cat/model_best.pt)')
    
    # 標的指定関連の引数
    parser.add_argument('--target_id', type=str, default=None, help='Target ID to filter pairs (e.g., CHEMBL312)')
    parser.add_argument('--target_col', type=str, default='TID', help='Column name for target ID (default: TID)')
    parser.add_argument('--list_targets', action='store_true', help='List available targets and their pair counts')
    
    args = parser.parse_args()
    
    # 標的一覧表示モード
    if args.list_targets:
        print("利用可能な標的一覧を表示します...")
        df = pd.read_csv(args.csv_path)
        
        if args.target_col not in df.columns:
            print(f"警告: カラム '{args.target_col}' が見つかりません。")
            print(f"利用可能なカラム: {list(df.columns)}")
            return
        
        target_counts = df[args.target_col].value_counts()
        print(f"利用可能な標的 (上位20件):")
        print("=" * 50)
        for i, (target_id, count) in enumerate(target_counts.head(20).items()):
            print(f"{i+1:2d}. {target_id}: {count} ペア")
        
        if len(target_counts) > 20:
            print(f"... 他 {len(target_counts) - 20} 件の標的")
        
        print(f"\n総標的数: {len(target_counts)}")
        print(f"総ペア数: {len(df)}")
        return
    
    print("=== Attention-Structure Analysis ===")
    print(f"データ: {args.csv_path}")
    print(f"出力先: {args.output_dir}")
    
    if args.target_id:
        print(f"指定標的: {args.target_id}")
    
    # 設定ファイルがあれば読み込み
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
            for key, value in config.items():
                setattr(default_args, key, value)
    
    # デバイス設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用デバイス: {device}")
    
    # 交差検証モードの場合
    if args.use_cv:
        if not args.split_file:
            raise ValueError("交差検証モード (--use_cv) では --split_file が必要です")
        if not args.model_dir:
            raise ValueError("交差検証モード (--use_cv) では --model_dir が必要です")
        
        print(f"\n=== 交差検証モード ===")
        print(f"分割ファイル: {args.split_file}")
        print(f"モデルディレクトリ: {args.model_dir}")
        print(f"モデルパターン: {args.model_pattern}")
        
        # 各foldのモデルパスを構築 (fold 0-4)
        model_paths = {}
        for fold in range(args.cv_folds):
            model_path = os.path.join(args.model_dir, args.model_pattern.format(fold=fold))
            if os.path.exists(model_path):
                model_paths[fold] = model_path
                print(f"  Fold {fold}: {model_path}")
            else:
                print(f"  警告: Fold {fold} のモデルが見つかりません: {model_path}")
        
        if not model_paths:
            raise ValueError("有効なモデルが見つかりません")
        
        # 交差検証データ処理を実行
        all_analyses = process_cv_dataset(
            model_paths, args.csv_path, args.split_file, device=device,
            output_dir=args.output_dir, batch_size=args.batch_size,
            target_id=args.target_id, target_col=args.target_col
        )
        
    # 単一モデルモードの場合
    else:
        if not args.model_path:
            raise ValueError("単一モデルモードでは --model_path が必要です")
        
        print(f"モデル: {args.model_path}")
        
        # モデル読み込み
        print("モデルを読み込み中...")
        model = load_model(args.model_path, default_args).to(device)
        print("モデル読み込み完了")
        
        # 大量データ処理を実行
        all_analyses = process_large_dataset(
            model, args.csv_path, device=device, output_dir=args.output_dir,
            batch_size=args.batch_size, max_pairs=args.max_pairs,
            target_id=args.target_id, target_col=args.target_col
        )
    
    print(f"\n解析完了！結果は {args.output_dir} に保存されました。")
    print(f"総解析ペア数: {len(all_analyses)}")

if __name__ == "__main__":
    main()