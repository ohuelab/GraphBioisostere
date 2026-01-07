###実行方法
# python visualize_attention.py --model_path ../results/results_consistentsmiles/cv0/pair-cat/model_best.pt --csv_path ../../../MMP_dataset/dataset_consistentsmiles.csv --n_pairs 10 --output_dir attention_visualization



import os
import sys
import torch
import numpy as np
from torch_geometric.data import Data, Batch
import matplotlib.cm as cm
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
import argparse
import json
import pandas as pd
import random

# 親ディレクトリをPythonパスに追加
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# 外部ファイルのインポート
from encoder.gnn_encoder import GNNEncoder
from model.graph_diff_regressor import GraphDiffRegressor
from utils.loader import smiles_to_data
from config import args as default_args

# --- モデルをロードする関数 ---
def load_model(model_path, args):
    """学習済みモデルをロードする"""
    encoder = GNNEncoder(
        node_in=args.node_in, edge_in=args.edge_in, hidden_dim=args.hidden_dim,
        out_dim=args.embedding_dim, num_layers=args.num_layers, dropout=args.dropout
    )
    model = GraphDiffRegressor(
        encoder, embedding_dim=args.embedding_dim, loss_type=args.loss_type,
        hidden_dim=getattr(args, "hidden_dim", 64), out_dim=getattr(args, "out_dim", 2)
    )

    # state_dictを読み込み
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    
    # mlp -> mlp_ab の変換（既存の処理）
    if 'mlp.0.weight' in state_dict and 'mlp_ab.0.weight' not in state_dict:
        state_dict = {k.replace('mlp.', 'mlp_ab.'): v for k, v in state_dict.items()}
    
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
        # 不要なキーを削除
        for key in unexpected_keys:
            if key not in model_state_dict:
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

# --- CSVファイルを読み込んでランダムサンプリングする関数 ---
def load_and_sample_pairs(csv_path, n_pairs=5, ref_col='REF-SMILES', prb_col='PRB-SMILES', label_col='label_bin', 
                         random_seed=None, target_id=None, target_col='TID'):
    """CSVファイルからSMILESペアを読み込み、ランダムにn個サンプリングする
    
    Args:
        csv_path: CSVファイルのパス
        n_pairs: サンプリング数
        ref_col: REF-SMILESのカラム名
        prb_col: PRB-SMILESのカラム名
        label_col: ラベルのカラム名
        random_seed: ランダムシード
        target_id: 指定する標的ID（Noneの場合は全標的から選択）
        target_col: 標的IDのカラム名
    """
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    # CSVファイルを読み込み
    df = pd.read_csv(csv_path)
    
    # 指定されたカラムが存在するかチェック
    if ref_col not in df.columns:
        raise ValueError(f"カラム '{ref_col}' がCSVファイルに見つかりません。利用可能なカラム: {list(df.columns)}")
    if prb_col not in df.columns:
        raise ValueError(f"カラム '{prb_col}' がCSVファイルに見つかりません。利用可能なカラム: {list(df.columns)}")
    if label_col not in df.columns:
        raise ValueError(f"カラム '{label_col}' がCSVファイルに見つかりません。利用可能なカラム: {list(df.columns)}")
    
    # 標的指定がある場合の処理
    if target_id is not None:
        if target_col not in df.columns:
            raise ValueError(f"標的指定されましたが、カラム '{target_col}' が見つかりません。利用可能なカラム: {list(df.columns)}")
        
        # 指定された標的でフィルタリング
        df_filtered = df[df[target_col] == target_id]
        print(f"標的 '{target_id}' でフィルタリング: {len(df_filtered)} 行")
        
        if len(df_filtered) == 0:
            # 利用可能な標的一覧を表示
            available_targets = df[target_col].dropna().unique()
            raise ValueError(f"標的 '{target_id}' のデータが見つかりません。\n利用可能な標的: {list(available_targets)}")
        
        df = df_filtered
    
    # NaNを含む行を除外
    df_clean = df.dropna(subset=[ref_col, prb_col, label_col])
    
    if target_id is not None:
        print(f"標的 '{target_id}': {len(df)} 行から {len(df_clean)} 行が有効なペアです。")
    else:
        print(f"CSVファイルから {len(df)} 行を読み込み、{len(df_clean)} 行が有効なペアです。")
    
    if len(df_clean) == 0:
        raise ValueError("有効なSMILESペアが見つかりません。")
    
    # 標的指定時のラベル分布を表示
    if target_id is not None:
        label_counts = df_clean[label_col].value_counts()
        print(f"標的 '{target_id}' のラベル分布:")
        for label, count in label_counts.items():
            label_text = "生物学的等価体" if label else "非等価体"
            print(f"  {label_text}: {count} ペア ({count/len(df_clean)*100:.1f}%)")
    
    # ランダムサンプリング
    n_available = len(df_clean)
    n_sample = min(n_pairs, n_available)
    
    if n_sample < n_pairs:
        print(f"警告: 要求された {n_pairs} ペアより少ない {n_sample} ペアのみ利用可能です。")
    
    sampled_df = df_clean.sample(n=n_sample, random_state=random_seed)
    
    # 詳細情報を取得するための定義済みカラム
    detail_columns = [
        'REF-CID', 'PRB-CID', 'AID', 'TID', 'CUT_NUM', 'COMMON_FRAG', 
        'REF-FRAG', 'PRB-FRAG', 'SMIRKS', 'REF-standard_value', 
        'PRB-standard_value', 'STANDARD_TYPE', 'delta_value'
    ]
    
    pairs = []
    for idx, row in sampled_df.iterrows():
        pair_info = {
            'index': idx,
            'ref_smiles': row[ref_col],
            'prb_smiles': row[prb_col],
            'true_label': row[label_col]
        }
        
        # 利用可能な詳細情報を追加
        detailed_info = {}
        for col in detail_columns:
            if col in df.columns and pd.notna(row[col]):
                detailed_info[col] = row[col]
        
        pair_info['detailed_info'] = detailed_info
        pairs.append(pair_info)
    
    return pairs

def list_available_targets(csv_path, target_col='TID', top_n=20):
    """利用可能な標的とそのペア数を表示する"""
    try:
        df = pd.read_csv(csv_path)
        
        if target_col not in df.columns:
            print(f"警告: カラム '{target_col}' が見つかりません。")
            print(f"利用可能なカラム: {list(df.columns)}")
            return
        
        # 標的別のペア数を集計
        target_counts = df[target_col].value_counts()
        
        print(f"利用可能な標的 (上位{top_n}件):")
        print("=" * 50)
        for i, (target_id, count) in enumerate(target_counts.head(top_n).items()):
            print(f"{i+1:2d}. {target_id}: {count} ペア")
        
        if len(target_counts) > top_n:
            print(f"... 他 {len(target_counts) - top_n} 件の標的")
        
        print(f"\n総標的数: {len(target_counts)}")
        print(f"総ペア数: {len(df)}")
        
        return target_counts
    
    except Exception as e:
        print(f"標的一覧取得でエラー: {e}")
        return None

# --- アテンションを可視化する関数 ---
import numpy as np
import matplotlib
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

def visualize_molecule_attention(smiles, attention_weights, output_path=None):
    """SMILES とアテンション重みから分子の重要度ヒートマップ（SVG）を作る"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    num_atoms = mol.GetNumAtoms()

    # --- アテンション重みの取得と正規化 ---
    if attention_weights is None:
        # 何もなければ全原子を同色（薄色）に
        weights = np.zeros(num_atoms, dtype=float)
        original_min, original_max = 0.0, 0.0
        w_original = np.zeros(num_atoms)
        normalized_min, normalized_max = 0.0, 0.0
    else:
        w = attention_weights
        # (N,1) → (N,) などの形状調整
        if hasattr(w, "detach"):
            w = w.detach()
        try:
            w = w.cpu().numpy()
        except Exception:
            w = np.asarray(w)
        w = np.squeeze(w)
        
        # 原子数に合わせて切る/パディング（基本は切る）
        if w.ndim == 0:
            w = np.full((num_atoms,), float(w))
        if w.shape[0] < num_atoms:
            # 足りない分は0で埋める
            w = np.pad(w, (0, num_atoms - w.shape[0]), constant_values=0.0)
        elif w.shape[0] > num_atoms:
            w = w[:num_atoms]

        # 数値の安定化（NaN/inf → 0）
        w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0).astype(float)
        
        # 元の値を保存（トリミング後）
        w_original = w.copy()
        original_min, original_max = float(w.min()), float(w.max())
        
        # デバッグ情報を出力（元の値）
        print(f"Original attention weights stats:")
        print(f"  min={original_min:.6f}, max={original_max:.6f}")
        print(f"  mean={w.mean():.6f}, std={w.std():.6f}")
        print(f"  unique values: {len(np.unique(w))}, shape: {w.shape}")
        print(f"  first 5 values: {w[:5]}")

        # [0,1] に正規化（定数配列は 0 に）
        wmin, wmax = float(w.min()), float(w.max())
        if wmax > wmin:
            weights = (w - wmin) / (wmax - wmin)
        else:
            weights = np.zeros_like(w, dtype=float)
        
        # 正規化後のデバッグ情報を追加
        normalized_min, normalized_max = float(weights.min()), float(weights.max())
        print(f"Normalized attention weights stats:")
        print(f"  min={normalized_min:.6f}, max={normalized_max:.6f}")
        print(f"  mean={weights.mean():.6f}, std={weights.std():.6f}")
        print(f"  first 5 normalized values: {weights[:5]}")
        
        # 正規化が正しく行われているかの検証
        if abs(normalized_min - 0.0) > 1e-6 or abs(normalized_max - 1.0) > 1e-6:
            if wmax > wmin:  # 定数配列でない場合のみ警告
                print(f"WARNING: Normalization may have failed!")
                print(f"  Expected: min=0.0, max=1.0")
                print(f"  Actual: min={normalized_min:.6f}, max={normalized_max:.6f}")

    # --- colormap（新API） ---
    cmap = matplotlib.colormaps.get('viridis')

    # --- RDKit 用の色辞書（Python float の RGB タプル）---
    atoms_to_highlight = list(range(num_atoms))
    atom_colors = {
        i: tuple(float(x) for x in cmap(weights[i])[:3])  # RGBA → RGB、かつ Python float 化
        for i in atoms_to_highlight
    }

    # --- 描画（SVG）分子部分 ---
    mol_width, mol_height = 400, 300
    drawer = rdMolDraw2D.MolDraw2DSVG(mol_width, mol_height)
    drawer.DrawMolecule(
        mol,
        highlightAtoms=atoms_to_highlight,
        highlightAtomColors=atom_colors
    )
    drawer.FinishDrawing()
    mol_svg = drawer.GetDrawingText()

    # --- カラーバーのSVG作成 ---
    colorbar_width, colorbar_height = 160, mol_height  # 幅をさらに広げて情報を表示
    bar_width = 25
    bar_x = 15
    bar_y = 30  # 上部のスペースを増やす
    bar_height = colorbar_height - 60  # 下部のスペースも増やす
    
    # カラーバーのグラデーション作成
    colorbar_svg_parts = []
    colorbar_svg_parts.append(f'<svg width="{colorbar_width}" height="{colorbar_height}" xmlns="http://www.w3.org/2000/svg">')
    
    # 白い背景を追加
    colorbar_svg_parts.append(f'<rect width="{colorbar_width}" height="{colorbar_height}" fill="white"/>')
    
    # グラデーション定義
    colorbar_svg_parts.append('<defs>')
    colorbar_svg_parts.append('<linearGradient id="colorbar_gradient" x1="0%" y1="100%" x2="0%" y2="0%">')
    
    # カラーマップから色を取得してグラデーションを作成
    num_stops = 20
    for i in range(num_stops + 1):
        stop_value = i / num_stops
        color = cmap(stop_value)
        rgb_color = f"rgb({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)})"
        colorbar_svg_parts.append(f'<stop offset="{stop_value*100}%" style="stop-color:{rgb_color};stop-opacity:1" />')
    
    colorbar_svg_parts.append('</linearGradient>')
    colorbar_svg_parts.append('</defs>')
    
    # カラーバーの矩形
    colorbar_svg_parts.append(f'<rect x="{bar_x}" y="{bar_y}" width="{bar_width}" height="{bar_height}" fill="url(#colorbar_gradient)" stroke="#cccccc" stroke-width="1"/>')
    
    # カラーバーの値表示
    font_size = 12
    text_x = bar_x + bar_width + 3
    
    # 値の範囲を確認
    value_range = original_max - original_min
    
    # タイトル（黒文字）
    colorbar_svg_parts.append(f'<text x="{colorbar_width/2}" y="15" font-family="Arial" font-size="{font_size+1}" fill="black" text-anchor="middle" font-weight="bold">Attention Weight</text>')
    
    if value_range < 1e-10:  # ほぼ定数の場合
        # 単一の値として表示
        colorbar_svg_parts.append(f'<text x="{text_x}" y="{bar_y + bar_height/2}" font-family="Arial" font-size="{font_size}" fill="black">All: {original_max:.6f}</text>')
        # バー内にも表示
        inner_text_x = bar_x + bar_width/2
        colorbar_svg_parts.append(f'<text x="{inner_text_x}" y="{bar_y + bar_height/2}" font-family="Arial" font-size="{font_size-1}" fill="white" text-anchor="middle" font-weight="bold">Constant</text>')
        
        # 正規化情報
        colorbar_svg_parts.append(f'<text x="{colorbar_width/2}" y="{colorbar_height - 25}" font-family="Arial" font-size="{font_size-1}" fill="black" text-anchor="middle">Normalized: [0.0, 0.0]</text>')
    else:
        # 通常の場合: 元の値と正規化後の値を両方表示
        num_ticks = 5  # 表示する目盛りを減らす
        for i in range(num_ticks):
            ratio = i / (num_ticks - 1)  # 0.0, 0.25, 0.5, 0.75, 1.0
            y_pos = bar_y + ratio * bar_height
            
            # 元の値を計算
            original_value = original_max - ratio * (original_max - original_min)
            
            # 精度を決定
            if value_range < 0.001:
                precision = 6
            elif value_range < 0.01:
                precision = 5
            elif value_range < 0.1:
                precision = 4
            else:
                precision = 3
            
            # 元の値を表示（右側、黒文字）
            colorbar_svg_parts.append(f'<text x="{text_x}" y="{y_pos + font_size/3}" font-family="Arial" font-size="{font_size-1}" fill="black">{original_value:.{precision}f}</text>')
            
            # max/minラベルを最初と最後の目盛りにのみ追加
            if i == 0:  # 最上部（最大値）
                colorbar_svg_parts.append(f'<text x="{text_x + 45}" y="{y_pos + font_size/3}" font-family="Arial" font-size="{font_size-1}" fill="#666666">max</text>')
            elif i == num_ticks - 1:  # 最下部（最小値）
                colorbar_svg_parts.append(f'<text x="{text_x + 45}" y="{y_pos + font_size/3}" font-family="Arial" font-size="{font_size-1}" fill="#666666">min</text>')
            
            # 目盛り線
            tick_x = bar_x + bar_width - 2
            colorbar_svg_parts.append(f'<line x1="{tick_x}" y1="{y_pos}" x2="{bar_x + bar_width}" y2="{y_pos}" stroke="black" stroke-width="1" opacity="0.7"/>')
        
        # 正規化情報を下部に追加
        colorbar_svg_parts.append(f'<text x="{colorbar_width/2}" y="{colorbar_height - 15}" font-family="Arial" font-size="{font_size-1}" fill="black" text-anchor="middle">Range: {value_range:.6f}</text>')
        # colorbar_svg_parts.append(f'<text x="{colorbar_width/2}" y="{colorbar_height - 15}" font-family="Arial" font-size="{font_size-1}" fill="black" text-anchor="middle">Normalized: [{normalized_min:.2f}, {normalized_max:.2f}]</text>')
    
    colorbar_svg_parts.append(f'<text x="{colorbar_width/2}" y="{colorbar_height - 5}" font-family="Arial" font-size="{font_size-1}" fill="black" text-anchor="middle">(Higher = More Important)</text>')
    
    colorbar_svg_parts.append('</svg>')
    colorbar_svg = '\n'.join(colorbar_svg_parts)

    # --- 分子SVGとカラーバーSVGを結合 ---
    total_width = mol_width + colorbar_width
    total_height = max(mol_height, colorbar_height)
    
    # 結合SVGの作成
    combined_svg_parts = []
    combined_svg_parts.append(f'<svg width="{total_width}" height="{total_height}" xmlns="http://www.w3.org/2000/svg">')
    
    # 分子SVGを埋め込み（viewBoxを使用して位置調整）
    mol_svg_content = mol_svg.split('<svg')[1].split('</svg>')[0]
    combined_svg_parts.append(f'<g transform="translate(0,0)">')
    combined_svg_parts.append(f'<svg{mol_svg_content}</svg>')
    combined_svg_parts.append('</g>')
    
    # カラーバーSVGを埋め込み
    colorbar_svg_content = colorbar_svg.split('<svg')[1].split('</svg>')[0]
    combined_svg_parts.append(f'<g transform="translate({mol_width},0)">')
    combined_svg_parts.append(f'<svg{colorbar_svg_content}</svg>')
    combined_svg_parts.append('</g>')
    
    combined_svg_parts.append('</svg>')
    final_svg = '\n'.join(combined_svg_parts)

    if output_path:
        with open(output_path, "w") as f:
            f.write(final_svg)
    return final_svg

# --- ペア全体の情報を含む統合SVGを作成する関数 ---
def create_pair_summary_svg(pair_dir, smiles1, smiles2, true_label, pair_index):
    """ペア全体の情報を含む統合SVGを作成する"""
    # 既存のSVGファイルを読み込み
    ref_svg_path = os.path.join(pair_dir, "ref_molecule_attention.svg")
    prb_svg_path = os.path.join(pair_dir, "prb_molecule_attention.svg")
    
    if not os.path.exists(ref_svg_path) or not os.path.exists(prb_svg_path):
        print("個別のSVGファイルが見つからないため、統合SVGを作成できません。")
        return
    
    with open(ref_svg_path, 'r') as f:
        ref_svg = f.read()
    with open(prb_svg_path, 'r') as f:
        prb_svg = f.read()
    
    # SVGの寸法を取得
    mol_width, mol_height = 560, 300  # アテンション可視化の幅（分子400 + カラーバー160）
    
    # 統合SVGの設定
    total_width = mol_width * 2 + 100  # 2つの分子 + 間隔
    header_height = 80  # ヘッダー情報用の高さ
    total_height = mol_height + header_height + 50  # 分子 + ヘッダー + 下部マージン
    
    # 統合SVGの作成
    svg_parts = []
    svg_parts.append(f'<svg width="{total_width}" height="{total_height}" xmlns="http://www.w3.org/2000/svg">')
    
    # 白い背景
    svg_parts.append(f'<rect width="{total_width}" height="{total_height}" fill="white"/>')
    
    # ヘッダー情報
    title_y = 25
    info_y = 50
    font_size = 14
    
    # 正解ラベル情報のみ表示（タイトルは削除）
    if true_label is not None:
        label_color = "#2E8B57" if true_label else "#DC143C"
        label_text = "生物学的等価体" if true_label else "非等価体"
        svg_parts.append(f'<text x="{total_width/2}" y="{title_y}" font-family="Arial" font-size="{font_size+1}" fill="{label_color}" text-anchor="middle" font-weight="bold">正解ラベル: {label_text}</text>')
    
    # 分子Aのラベル
    ref_x = mol_width / 2
    label_y = header_height - 10
    svg_parts.append(f'<text x="{ref_x}" y="{label_y}" font-family="Arial" font-size="{font_size+1}" fill="#4169E1" text-anchor="middle" font-weight="bold">分子A</text>')
    
    # 分子Bのラベル
    prb_x = mol_width + 50 + mol_width / 2
    svg_parts.append(f'<text x="{prb_x}" y="{label_y}" font-family="Arial" font-size="{font_size+1}" fill="#FF6347" text-anchor="middle" font-weight="bold">分子B</text>')
    
    # SVGファイルの内容を正しく抽出して埋め込み
    def extract_svg_content(svg_text):
        """SVGテキストから内部コンテンツを抽出"""
        try:
            # 最初の<svg>タグを見つけて開始位置を取得
            start_tag = svg_text.find('<svg')
            if start_tag == -1:
                return ""
            
            # 対応する</svg>タグを見つけて終了位置を取得
            end_tag = svg_text.rfind('</svg>')
            if end_tag == -1:
                return ""
            
            # <svg>タグの終了（>）を見つける
            tag_end = svg_text.find('>', start_tag)
            if tag_end == -1:
                return ""
            
            # <svg>の開始タグの後から</svg>の前まで抽出
            content = svg_text[tag_end + 1:end_tag]
            return content.strip()
        except Exception as e:
            print(f"SVG内容の抽出でエラー: {e}")
            return ""
    
    # REF分子のSVGを埋め込み
    ref_content = extract_svg_content(ref_svg)
    if ref_content:
        svg_parts.append(f'<g transform="translate(0,{header_height})">')
        svg_parts.append(f'<svg width="{mol_width}" height="{mol_height}" xmlns="http://www.w3.org/2000/svg">')
        svg_parts.append(ref_content)
        svg_parts.append('</svg>')
        svg_parts.append('</g>')
    else:
        # フォールバック：プレースホルダーを表示
        svg_parts.append(f'<g transform="translate(0,{header_height})">')
        svg_parts.append(f'<rect x="0" y="0" width="{mol_width}" height="{mol_height}" fill="#f0f0f0" stroke="#cccccc"/>')
        svg_parts.append(f'<text x="{mol_width/2}" y="{mol_height/2}" font-family="Arial" font-size="16" fill="red" text-anchor="middle">REF分子表示エラー</text>')
        svg_parts.append('</g>')
    
    # PRB分子のSVGを埋め込み
    prb_content = extract_svg_content(prb_svg)
    if prb_content:
        svg_parts.append(f'<g transform="translate({mol_width + 50},{header_height})">')
        svg_parts.append(f'<svg width="{mol_width}" height="{mol_height}" xmlns="http://www.w3.org/2000/svg">')
        svg_parts.append(prb_content)
        svg_parts.append('</svg>')
        svg_parts.append('</g>')
    else:
        # フォールバック：プレースホルダーを表示
        svg_parts.append(f'<g transform="translate({mol_width + 50},{header_height})">')
        svg_parts.append(f'<rect x="0" y="0" width="{mol_width}" height="{mol_height}" fill="#f0f0f0" stroke="#cccccc"/>')
        svg_parts.append(f'<text x="{mol_width/2}" y="{mol_height/2}" font-family="Arial" font-size="16" fill="red" text-anchor="middle">PRB分子表示エラー</text>')
        svg_parts.append('</g>')
    
    # 分離線
    line_x = mol_width + 25
    svg_parts.append(f'<line x1="{line_x}" y1="{header_height}" x2="{line_x}" y2="{total_height - 20}" stroke="#CCCCCC" stroke-width="2" stroke-dasharray="5,5"/>')
    
    # SMILES情報（下部）
    smiles_y = total_height - 30
    smiles_font_size = 10
    
    # 分子A SMILES
    ref_smiles_display = smiles1[:60] + "..." if len(smiles1) > 60 else smiles1
    svg_parts.append(f'<text x="{ref_x}" y="{smiles_y}" font-family="Courier" font-size="{smiles_font_size}" fill="#666666" text-anchor="middle">分子A: {ref_smiles_display}</text>')
    
    # 分子B SMILES
    prb_smiles_display = smiles2[:60] + "..." if len(smiles2) > 60 else smiles2
    svg_parts.append(f'<text x="{prb_x}" y="{smiles_y}" font-family="Courier" font-size="{smiles_font_size}" fill="#666666" text-anchor="middle">分子B: {prb_smiles_display}</text>')
    
    svg_parts.append('</svg>')
    
    # 統合SVGを保存
    summary_svg = '\n'.join(svg_parts)
    summary_path = os.path.join(pair_dir, "pair_summary.svg")
    
    try:
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary_svg)
        print(f"統合SVG（正解ラベル情報付き）を保存しました: {summary_path}")
    except Exception as e:
        print(f"統合SVG保存でエラー: {e}")
        # デバッグ用：最初の100文字を表示
        print(f"SVG内容の最初の部分: {summary_svg[:200]}...")

# --- 分子ペアを処理する関数（修正版） ---
def process_molecule_pair(model, smiles1, smiles2, device, output_dir=None, pair_index=0, pair_info=None):
    """分子ペアに対してモデルを実行し、アテンションの可視化を行う"""
    # pair_infoから情報を取得（後方互換性のため）
    if pair_info is None:
        pair_info = {
            'ref_smiles': smiles1,
            'prb_smiles': smiles2,
            'true_label': None,
            'detailed_info': {}
        }
    
    true_label = pair_info.get('true_label')
    
    print(f"\n=== ペア {pair_index + 1} の処理 ===")
    print(f"REF-SMILES: {smiles1}")
    print(f"PRB-SMILES: {smiles2}")
    if true_label is not None:
        print(f"実際のラベル（生物学的等価体）: {true_label}")
    
    # CHEMBL IDがあれば表示
    detailed_info = pair_info.get('detailed_info', {})
    if 'REF-CID' in detailed_info:
        print(f"REF-CHEMBL ID: {detailed_info['REF-CID']}")
    if 'PRB-CID' in detailed_info:
        print(f"PRB-CHEMBL ID: {detailed_info['PRB-CID']}")
    if 'TID' in detailed_info:
        print(f"標的ID: {detailed_info['TID']}")
    
    data1 = smiles_to_data(smiles1)
    data2 = smiles_to_data(smiles2)
    
    if data1 is None or data2 is None:
        print("SMILESの変換に失敗しました。")
        return False

    batch1 = Batch.from_data_list([data1]).to(device)
    batch2 = Batch.from_data_list([data2]).to(device)
    
    with torch.no_grad():
        # ステップ1: 各分子でエンコーダーを個別に実行し、アテンション重みを取得
        model.encoder(batch1)
        attn_weights1 = model.encoder.attention_weights
        
        model.encoder(batch2)
        attn_weights2 = model.encoder.attention_weights
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # ペア毎にサブディレクトリを作成
        pair_dir = os.path.join(output_dir, f"pair_{pair_index + 1}")
        os.makedirs(pair_dir, exist_ok=True)
    
    # 可視化関数を呼び出す
    try:
        # REF分子の可視化
        ref_svg = visualize_molecule_attention(
            smiles1, attn_weights1,
            output_path=os.path.join(pair_dir, "ref_molecule_attention.svg") if output_dir else None
        )
        
        # PRB分子の可視化
        prb_svg = visualize_molecule_attention(
            smiles2, attn_weights2,
            output_path=os.path.join(pair_dir, "prb_molecule_attention.svg") if output_dir else None
        )
        
        # ペア全体の情報を含む統合SVGを作成
        if output_dir:
            create_pair_summary_svg(
                pair_dir, smiles1, smiles2, true_label, pair_index
            )
        
        # 詳細情報を保存
        if output_dir:
            save_detailed_pair_info(pair_dir, pair_info, pair_index)
        
        # 従来のSMILES情報ファイルも作成（後方互換性のため）
        if output_dir:
            with open(os.path.join(pair_dir, "smiles_info.txt"), "w") as f:
                f.write(f"REF-SMILES: {smiles1}\n")
                f.write(f"PRB-SMILES: {smiles2}\n")
                if true_label is not None:
                    f.write(f"実際のラベル（生物学的等価体）: {true_label}\n")
        
        print(f"ペア {pair_index + 1} の可視化が完了しました。")
        return True
        
    except Exception as e:
        print(f"ペア {pair_index + 1} の可視化でエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return False

# --- 詳細情報を保存する関数 ---
def save_detailed_pair_info(pair_dir, pair_info, pair_index):
    """分子ペアの詳細情報をTXTファイルに保存する"""
    info_path = os.path.join(pair_dir, "detailed_info.txt")
    
    try:
        with open(info_path, "w", encoding='utf-8') as f:
            f.write(f"=== 分子ペア {pair_index + 1} の詳細情報 ===\n\n")
            
            # 基本情報
            f.write("【基本情報】\n")
            f.write(f"REF-SMILES: {pair_info['ref_smiles']}\n")
            f.write(f"PRB-SMILES: {pair_info['prb_smiles']}\n")
            if pair_info.get('true_label') is not None:
                label_text = "生物学的等価体" if pair_info['true_label'] else "非等価体"
                f.write(f"実際のラベル: {label_text} ({pair_info['true_label']})\n")
            f.write(f"データセット内のインデックス: {pair_info['index']}\n\n")
            
            # 詳細情報
            if 'detailed_info' in pair_info and pair_info['detailed_info']:
                f.write("【詳細メタデータ】\n")
                detailed_info = pair_info['detailed_info']
                
                # CHEMBL ID情報
                if 'REF-CID' in detailed_info:
                    f.write(f"REF-CHEMBL ID: {detailed_info['REF-CID']}\n")
                if 'PRB-CID' in detailed_info:
                    f.write(f"PRB-CHEMBL ID: {detailed_info['PRB-CID']}\n")
                
                # アッセイ・標的情報
                if 'AID' in detailed_info:
                    f.write(f"アッセイID (AID): {detailed_info['AID']}\n")
                if 'TID' in detailed_info:
                    f.write(f"標的ID (TID): {detailed_info['TID']}\n")
                if 'STANDARD_TYPE' in detailed_info:
                    f.write(f"測定タイプ: {detailed_info['STANDARD_TYPE']}\n")
                
                # 活性値情報
                if 'REF-standard_value' in detailed_info:
                    f.write(f"REF分子の標準値: {detailed_info['REF-standard_value']}\n")
                if 'PRB-standard_value' in detailed_info:
                    f.write(f"PRB分子の標準値: {detailed_info['PRB-standard_value']}\n")
                if 'delta_value' in detailed_info:
                    f.write(f"活性差 (delta_value): {detailed_info['delta_value']}\n")
                
                # フラグメント情報
                if 'CUT_NUM' in detailed_info:
                    f.write(f"カット数: {detailed_info['CUT_NUM']}\n")
                if 'COMMON_FRAG' in detailed_info:
                    f.write(f"共通フラグメント: {detailed_info['COMMON_FRAG']}\n")
                if 'REF-FRAG' in detailed_info:
                    f.write(f"REFフラグメント: {detailed_info['REF-FRAG']}\n")
                if 'PRB-FRAG' in detailed_info:
                    f.write(f"PRBフラグメント: {detailed_info['PRB-FRAG']}\n")
                if 'SMIRKS' in detailed_info:
                    f.write(f"SMIRKS反応: {detailed_info['SMIRKS'][:200]}...\n")  # 長すぎる場合は省略
                
                f.write("\n")
            else:
                f.write("【詳細メタデータ】\n")
                f.write("利用可能な詳細情報がありません。\n\n")
            
            # ファイル生成時刻
            from datetime import datetime
            f.write(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        print(f"詳細情報を保存しました: {info_path}")
        return True
        
    except Exception as e:
        print(f"詳細情報保存でエラー: {e}")
        return False

# --- 複数ペアを処理する関数 ---
def process_multiple_pairs(model, pairs, device, output_dir):
    """複数の分子ペアを処理する"""
    successful_pairs = 0
    
    for i, pair in enumerate(pairs):
        success = process_molecule_pair(
            model, pair['ref_smiles'], pair['prb_smiles'], 
            device=device, output_dir=output_dir, pair_index=i, pair_info=pair
        )
        if success:
            successful_pairs += 1
    
    print(f"\n=== 処理完了 ===")
    print(f"総ペア数: {len(pairs)}")
    print(f"成功: {successful_pairs}")
    print(f"失敗: {len(pairs) - successful_pairs}")
    
    # 詳細情報を含む総合レポートも作成
    if output_dir and successful_pairs > 0:
        create_summary_report(pairs, output_dir, successful_pairs)

# --- 総合レポート作成関数 ---
def create_summary_report(pairs, output_dir, successful_pairs):
    """処理した全ペアの総合レポートを作成する"""
    report_path = os.path.join(output_dir, "analysis_summary.txt")
    
    try:
        with open(report_path, "w", encoding='utf-8') as f:
            f.write("=== 分子ペア解析総合レポート ===\n\n")
            
            # 基本統計
            f.write("【基本統計】\n")
            f.write(f"総処理ペア数: {len(pairs)}\n")
            f.write(f"成功した処理: {successful_pairs}\n")
            f.write(f"失敗した処理: {len(pairs) - successful_pairs}\n\n")
            
            # ラベル分布
            true_labels = [pair.get('true_label') for pair in pairs if pair.get('true_label') is not None]
            if true_labels:
                bio_equiv_count = sum(true_labels)
                f.write("【ラベル分布】\n")
                f.write(f"生物学的等価体ペア: {bio_equiv_count}\n")
                f.write(f"非等価体ペア: {len(true_labels) - bio_equiv_count}\n")
                f.write(f"等価体率: {bio_equiv_count / len(true_labels):.3f}\n\n")
            
            # 標的・アッセイ情報の集計
            targets = []
            assays = []
            standard_types = []
            
            for pair in pairs:
                detailed_info = pair.get('detailed_info', {})
                if 'TID' in detailed_info:
                    targets.append(detailed_info['TID'])
                if 'AID' in detailed_info:
                    assays.append(detailed_info['AID'])
                if 'STANDARD_TYPE' in detailed_info:
                    standard_types.append(detailed_info['STANDARD_TYPE'])
            
            if targets:
                unique_targets = set(targets)
                f.write("【標的情報】\n")
                f.write(f"ユニーク標的数: {len(unique_targets)}\n")
                f.write(f"対象標的ID: {', '.join(sorted(unique_targets))}\n\n")
            
            if assays:
                unique_assays = set(assays)
                f.write("【アッセイ情報】\n")
                f.write(f"ユニークアッセイ数: {len(unique_assays)}\n\n")
            
            if standard_types:
                unique_types = set(standard_types)
                f.write("【測定タイプ】\n")
                f.write(f"測定タイプ: {', '.join(sorted(unique_types))}\n\n")
            
            # 各ペアの概要
            f.write("【個別ペア概要】\n")
            for i, pair in enumerate(pairs):
                f.write(f"ペア {i+1}:\n")
                detailed_info = pair.get('detailed_info', {})
                
                if 'REF-CID' in detailed_info and 'PRB-CID' in detailed_info:
                    f.write(f"  CHEMBL ID: {detailed_info['REF-CID']} vs {detailed_info['PRB-CID']}\n")
                if 'TID' in detailed_info:
                    f.write(f"  標的ID: {detailed_info['TID']}\n")
                if pair.get('true_label') is not None:
                    label_text = "等価体" if pair['true_label'] else "非等価体"
                    f.write(f"  ラベル: {label_text}\n")
                if 'delta_value' in detailed_info:
                    f.write(f"  活性差: {detailed_info['delta_value']}\n")
                
                f.write("\n")
            
            # 生成時刻
            from datetime import datetime
            f.write(f"レポート生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        print(f"総合レポートを作成しました: {report_path}")
        
    except Exception as e:
        print(f"総合レポート作成でエラー: {e}")

# --- メイン実行ブロック ---
def main():
    parser = argparse.ArgumentParser(description='Visualize model attention for molecule pairs from CSV')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    
    # CSVファイル関連の引数
    parser.add_argument('--csv_path', type=str, help='Path to CSV file containing SMILES pairs')
    parser.add_argument('--n_pairs', type=int, default=5, help='Number of pairs to randomly sample (default: 5)')
    parser.add_argument('--ref_col', type=str, default='REF-SMILES', help='Column name for reference SMILES (default: REF-SMILES)')
    parser.add_argument('--prb_col', type=str, default='PRB-SMILES', help='Column name for probe SMILES (default: PRB-SMILES)')
    parser.add_argument('--random_seed', type=int, default=None, help='Random seed for reproducible sampling')
    
    # 標的指定関連の引数
    parser.add_argument('--target_id', type=str, default=None, help='Target ID to filter pairs (e.g., CHEMBL312)')
    parser.add_argument('--target_col', type=str, default='TID', help='Column name for target ID (default: TID)')
    parser.add_argument('--list_targets', action='store_true', help='List available targets and their pair counts')
    
    # 従来の単一ペア指定（CSVと併用不可）
    parser.add_argument('--smiles1', type=str, help='SMILES string for first molecule (alternative to CSV)')
    parser.add_argument('--smiles2', type=str, help='SMILES string for second molecule (alternative to CSV)')
    
    # その他の引数
    parser.add_argument('--output_dir', type=str, default='attention_results', help='Directory to save visualization results')
    parser.add_argument('--config', type=str, default=None, help='Path to config file (optional)')
    
    cmd_args = parser.parse_args()
    
    # 引数の検証
    if cmd_args.csv_path and (cmd_args.smiles1 or cmd_args.smiles2):
        raise ValueError("CSVファイル（--csv_path）と個別SMILES指定（--smiles1, --smiles2）は同時に使用できません。")
    
    if not cmd_args.csv_path and not (cmd_args.smiles1 and cmd_args.smiles2):
        raise ValueError("CSVファイル（--csv_path）または個別SMILES（--smiles1, --smiles2）のいずれかを指定してください。")
    
    if cmd_args.config and os.path.exists(cmd_args.config):
        with open(cmd_args.config, 'r') as f:
            config = json.load(f)
            for key, value in config.items():
                setattr(default_args, key, value)
    
    if cmd_args.csv_path:
        # 標的一覧表示モード
        if cmd_args.list_targets:
            print("利用可能な標的一覧を表示します...")
            list_available_targets(cmd_args.csv_path, cmd_args.target_col)
            return
        
        # 標的指定の場合の情報表示
        if cmd_args.target_id:
            print(f"標的 '{cmd_args.target_id}' を指定してサンプリングします...")
        else:
            print(f"全標的からランダムサンプリングします...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = load_model(cmd_args.model_path, default_args).to(device)
    
    if cmd_args.csv_path:
        # CSVファイルからペアを読み込んで処理
        print(f"CSVファイル '{cmd_args.csv_path}' から {cmd_args.n_pairs} ペアをサンプリングします...")
        pairs = load_and_sample_pairs(
            cmd_args.csv_path, 
            n_pairs=cmd_args.n_pairs,
            ref_col=cmd_args.ref_col,
            prb_col=cmd_args.prb_col,
            random_seed=cmd_args.random_seed,
            target_id=cmd_args.target_id,
            target_col=cmd_args.target_col
        )
        process_multiple_pairs(model, pairs, device=device, output_dir=cmd_args.output_dir)
    else:
        # 従来の単一ペア処理
        pair_info = {
            'ref_smiles': cmd_args.smiles1,
            'prb_smiles': cmd_args.smiles2,
            'true_label': None,
            'detailed_info': {}
        }
        process_molecule_pair(
            model, cmd_args.smiles1, cmd_args.smiles2, 
            device=device, output_dir=cmd_args.output_dir, pair_index=0, pair_info=pair_info
        )
    
    print(f"\n結果を {cmd_args.output_dir} に保存しました。")

if __name__ == "__main__":
    main()