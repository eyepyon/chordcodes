# src/utils/gpu_dataloader.py
"""
GPU最適化データローダー
CUDA 12.x対応完全版
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Any, Optional, Union
import pickle
import os
import glob
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
import logging
import random
from tqdm import tqdm
import time

logger = logging.getLogger(__name__)

class MIDIDatasetGPU(Dataset):
    """GPU最適化MIDIデータセット"""
    
    def __init__(self, 
                 data_dir: str,
                 max_sequence_length: int = 1024,
                 preload_to_memory: bool = True,
                 pin_memory: bool = True,
                 cache_dir: str = None,
                 composer_filter: List[str] = None):
        
        self.data_dir = Path(data_dir)
        self.max_seq_len = max_sequence_length
        self.preload = preload_to_memory
        self.pin_memory = pin_memory
        self.cache_dir = Path(cache_dir) if cache_dir else self.data_dir.parent / "cache"
        self.composer_filter = composer_filter or ['chopin', 'beethoven', 'bach', 'mozart']
        
        # キャッシュディレクトリ作成
        self.cache_dir.mkdir(exist_ok=True)
        
        # データファイル探索
        self.data_files = self._discover_data_files()
        logger.info(f"Found {len(self.data_files)} data files")
        
        # 語彙マッピング
        self.vocab = self._create_vocabulary()
        self.vocab_size = len(self.vocab)
        
        # メモリに事前ロード（オプション）
        if self.preload:
            self.memory_data = self._preload_data()
        
        logger.info(f"Dataset initialized: {len(self)} samples")
    
    def _discover_data_files(self) -> List[Dict[str, Any]]:
        """データファイル探索"""
        files = []
        
        for composer in self.composer_filter:
            composer_dir = self.data_dir / composer
            if composer_dir.exists():
                # MIDIファイル
                midi_files = list(composer_dir.glob("*.mid")) + list(composer_dir.glob("*.midi"))
                
                # 特徴量ファイル
                feature_files = list(composer_dir.glob("*_features.pkl"))
                
                for midi_file in midi_files:
                    # 対応する特徴量ファイルを探索
                    feature_file = None
                    base_name = midi_file.stem
                    
                    for feat_file in feature_files:
                        if base_name in feat_file.stem:
                            feature_file = feat_file
                            break
                    
                    files.append({
                        'midi_path': str(midi_file),
                        'feature_path': str(feature_file) if feature_file else None,
                        'composer': composer,
                        'file_id': len(files)
                    })
        
        return files
    
    def _create_vocabulary(self) -> Dict[str, int]:
        """語彙作成"""
        vocab = {
            '<pad>': 0,
            '<unk>': 1, 
            '<start>': 2,
            '<end>': 3
        }
        
        # 音符トークン (C0 = MIDI 12 から C8 = MIDI 108)
        for note in range(128):
            vocab[f'note_{note}'] = len(vocab)
        
        # ベロシティトークン
        for vel in range(128):
            vocab[f'vel_{vel}'] = len(vocab)
        
        # 時間トークン (16分音符単位)
        for time_step in range(256):
            vocab[f'time_{time_step}'] = len(vocab)
        
        # 作曲家トークン
        for composer in self.composer_filter:
            vocab[f'composer_{composer}'] = len(vocab)
        
        return vocab
    
    def _preload_data(self) -> List[Dict[str, Any]]:
        """データを並列でメモリにプリロード"""
        logger.info("Preloading data to memory...")
        
        memory_data = [None] * len(self.data_files)
        
        # 並列処理でロード
        max_workers = min(mp.cpu_count(), 8)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # タスク送信
            future_to_idx = {
                executor.submit(self._load_single_file, i): i 
                for i in range(len(self.data_files))
            }
            
            # 結果収集
            for future in tqdm(as_completed(future_to_idx), 
                             desc="Loading data", 
                             total=len(future_to_idx)):
                idx = future_to_idx[future]
                try:
                    data = future.result()
                    memory_data[idx] = data
                except Exception as e:
                    logger.error(f"Error loading file {idx}: {e}")
                    memory_data[idx] = None
        
        # None を除外
        memory_data = [data for data in memory_data if data is not None]
        
        logger.info(f"Preloaded {len(memory_data)} files to memory")
        return memory_data
    
    def _load_single_file(self, idx: int) -> Optional[Dict[str, Any]]:
        """単一ファイルをロード"""
        file_info = self.data_files[idx]
        
        try:
            # キャッシュチェック
            cache_path = self.cache_dir / f"processed_{file_info['file_id']}.pt"
            
            if cache_path.exists():
                return torch.load(cache_path)
            
            # MIDIファイル処理
            midi_data = self._process_midi_file(file_info['midi_path'])
            
            # 特徴量ファイル処理
            features = None
            if file_info['feature_path']:
                features = self._load_features(file_info['feature_path'])
            
            # データ統合
            processed_data = {
                'tokens': midi_data['tokens'],
                'composer': file_info['composer'],
                'features': features,
                'file_id': file_info['file_id'],
                'original_length': len(midi_data['tokens'])
            }
            
            # キャッシュ保存
            torch.save(processed_data, cache_path)
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error processing {file_info['midi_path']}: {e}")
            return None
    
    def _process_midi_file(self, midi_path: str) -> Dict[str, Any]:
        """MIDIファイルをトークンに変換"""
        try:
            import pretty_midi
            
            # MIDI読み込み
            midi = pretty_midi.PrettyMIDI(midi_path)
            
            tokens = [self.vocab['<start>']]
            
            # 各楽器の音符を処理
            for instrument in midi.instruments:
                if not instrument.is_drum:
                    # 音符を時間順にソート
                    notes = sorted(instrument.notes, key=lambda x: x.start)
                    
                    current_time = 0.0
                    
                    for note in notes:
                        # 時間進行
                        time_diff = note.start - current_time
                        if time_diff > 0:
                            time_steps = min(int(time_diff * 4), 255)  # 16分音符単位
                            tokens.append(self.vocab[f'time_{time_steps}'])
                        
                        # 音符
                        note_token = f'note_{note.pitch}'
                        if note_token in self.vocab:
                            tokens.append(self.vocab[note_token])
                        
                        # ベロシティ
                        vel_token = f'vel_{note.velocity}'
                        if vel_token in self.vocab:
                            tokens.append(self.vocab[vel_token])
                        
                        current_time = note.start
            
            tokens.append(self.vocab['<end>'])
            
            return {'tokens': tokens}
            
        except Exception as e:
            logger.error(f"Error processing MIDI {midi_path}: {e}")
            # エラー時はダミートークン
            return {'tokens': [self.vocab['<start>'], self.vocab['<end>']]}
    
    def _load_features(self, feature_path: str) -> Optional[Dict[str, Any]]:
        """特徴量ファイル読み込み"""
        try:
            with open(feature_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading features {feature_path}: {e}")
            return None
    
    def __len__(self) -> int:
        if self.preload:
            return len(self.memory_data)
        return len(self.data_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # データ取得
        if self.preload:
            data = self.memory_data[idx]
        else:
            data = self._load_single_file(idx)
        
        if data is None:
            # エラー時のダミーデータ
            return self._get_dummy_sample()
        
        # トークン処理
        tokens = data['tokens']
        
        # パディング・切り詰め
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
        else:
            # パディング
            pad_length = self.max_seq_len - len(tokens)
            tokens = tokens + [self.vocab['<pad>']] * pad_length
        
        # PyTorchテンソルに変換
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        
        # パディング部分のattention maskを0に
        pad_positions = input_ids == self.vocab['<pad>']
        attention_mask[pad_positions] = 0
        
        # 作曲家ID
        composer_names = ['chopin', 'beethoven', 'bach', 'mozart']
        composer_id = composer_names.index(data['composer']) if data['composer'] in composer_names else 0
        
        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'composer_ids': torch.tensor(composer_id, dtype=torch.long)
        }
        
        # ピン留めメモリ（GPU転送高速化）
        if self.pin_memory:
            result = {k: v.pin_memory() if isinstance(v, torch.Tensor) else v 
                     for k, v in result.items()}
        
        return result
    
    def _get_dummy_sample(self) -> Dict[str, torch.Tensor]:
        """ダミーサンプル生成（エラー時用）"""
        tokens = [self.vocab['<start>']] + [self.vocab['<pad>']] * (self.max_seq_len - 2) + [self.vocab['<end>']]
        
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)  # 全てマスク
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'composer_ids': torch.tensor(0, dtype=torch.long)
        }
    
    def get_vocab_size(self) -> int:
        """語彙サイズ取得"""
        return self.vocab_size
    
    def get_composer_stats(self) -> Dict[str, int]:
        """作曲家別統計"""
        stats = {}
        
        if self.preload:
            for data in self.memory_data:
                composer = data['composer']
                stats[composer] = stats.get(composer, 0) + 1
        else:
            for file_info in self.data_files:
                composer = file_info['composer']
                stats[composer] = stats.get(composer, 0) + 1
        
        return stats


class GPUCollator:
    """GPU最適化コライダー（バッチ処理）"""
    
    def __init__(self, 
                 pad_token_id: int = 0,
                 max_length: int = None,
                 padding_strategy: str = "longest"):
        self.pad_token_id = pad_token_id
        self.max_length = max_length
        self.padding_strategy = padding_strategy
    
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """バッチ処理"""
        if not batch:
            raise ValueError("Empty batch")
        
        # バッチサイズ
        batch_size = len(batch)
        
        # 最大長決定
        if self.padding_strategy == "longest":
            max_len = max(item['input_ids'].size(0) for item in batch)
            if self.max_length:
                max_len = min(max_len, self.max_length)
        else:
            max_len = self.max_length or 1024
        
        # テンソル初期化（GPU効率化のため事前確保）
        device = batch[0]['input_ids'].device
        dtype_long = torch.long
        
        # バッチテンソル作成
        input_ids = torch.full((batch_size, max_len), self.pad_token_id, 
                               dtype=dtype_long, device=device)
        attention_mask = torch.zeros((batch_size, max_len), 
                                   dtype=dtype_long, device=device)
        labels = torch.full((batch_size, max_len), -100, 
                           dtype=dtype_long, device=device)
        composer_ids = torch.zeros(batch_size, dtype=dtype_long, device=device)
        
        # データコピー（vectorized operations）
        for i, item in enumerate(batch):
            seq_len = min(item['input_ids'].size(0), max_len)
            
            input_ids[i, :seq_len] = item['input_ids'][:seq_len]
            attention_mask[i, :seq_len] = item['attention_mask'][:seq_len]
            labels[i, :seq_len] = item['labels'][:seq_len]
            composer_ids[i] = item['composer_ids']
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'composer_ids': composer_ids
        }


def create_optimized_dataloader(
    data_dir: str,
    batch_size: int = 16,
    max_sequence_length: int = 1024,
    num_workers: int = None,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
    composer_filter: List[str] = None,
    shuffle: bool = True
) -> DataLoader:
    """最適化されたデータローダー作成"""
    
    # デフォルトワーカー数設定
    if num_workers is None:
        num_workers = min(8, mp.cpu_count())
    
    # データセット作成
    dataset = MIDIDatasetGPU(
        data_dir=data_dir,
        max_sequence_length=max_sequence_length,
        preload_to_memory=True,
        pin_memory=pin_memory,
        composer_filter=composer_filter
    )
    
    # コライダー
    collator = GPUCollator(
        pad_token_id=0,
        max_length=max_sequence_length,
        padding_strategy="longest"
    )
    
    # データローダー
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collator,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        drop_last=True,  # バッチサイズを統一
        worker_init_fn=_worker_init_fn
    )
    
    logger.info(f"DataLoader created:")
    logger.info(f"  Dataset size: {len(dataset)}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Num workers: {num_workers}")
    logger.info(f"  Batches: {len(dataloader)}")
    
    # 作曲家統計表示
    stats = dataset.get_composer_stats()
    logger.info(f"  Composer distribution: {stats}")
    
    return dataloader


def _worker_init_fn(worker_id: int):
    """ワーカープロセス初期化関数"""
    # 各ワーカーで異なる乱数シード
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# データローダーテスト・ベンチマーク
class DataLoaderBenchmark:
    """データローダー性能ベンチマーク"""
    
    @staticmethod
    def benchmark_dataloader(dataloader: DataLoader, num_batches: int = 10):
        """データローダー性能測定"""
        logger.info(f"Benchmarking dataloader with {num_batches} batches...")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # ウォームアップ
        for i, batch in enumerate(dataloader):
            if i >= 2:  # 2バッチでウォームアップ
                break
        
        # 実測定
        start_time = time.time()
        total_samples = 0
        
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            
            # GPU転送テスト
            if device.type == "cuda":
                batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                torch.cuda.synchronize()  # 同期待ち
            
            total_samples += batch['input_ids'].size(0)
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        samples_per_sec = total_samples / elapsed
        batches_per_sec = num_batches / elapsed
        
        logger.info(f"Benchmark results:")
        logger.info(f"  Total time: {elapsed:.2f}s")
        logger.info(f"  Samples/sec: {samples_per_sec:.1f}")
        logger.info(f"  Batches/sec: {batches_per_sec:.1f}")
        logger.info(f"  Avg batch time: {elapsed/num_batches*1000:.1f}ms")
        
        return {
            'samples_per_sec': samples_per_sec,
            'batches_per_sec': batches_per_sec,
            'avg_batch_time_ms': elapsed/num_batches*1000
        }


def main():
    """テスト実行"""
    logging.basicConfig(level=logging.INFO)
    
    # テストデータディレクトリ
    data_dir = "data/processed"
    
    if not os.path.exists(data_dir):
        logger.warning(f"Data directory {data_dir} not found. Creating dummy structure...")
        os.makedirs(data_dir + "/chopin", exist_ok=True)
        
        # ダミーファイル作成（テスト用）
        dummy_tokens = [2, 100, 101, 102, 3]  # start, notes, end
        dummy_data = {'tokens': dummy_tokens}
        
        import pickle
        with open(data_dir + "/chopin/test.pkl", "wb") as f:
            pickle.dump(dummy_data, f)
    
    # データローダー作成
    dataloader = create_optimized_dataloader(
        data_dir=data_dir,
        batch_size=4,
        max_sequence_length=512,
        num_workers=2
    )
    
    # 最初のバッチをテスト
    logger.info("Testing first batch...")
    for batch in dataloader:
        logger.info(f"Batch shapes:")
        for key, tensor in batch.items():
            logger.info(f"  {key}: {tensor.shape} ({tensor.dtype})")
        
        # GPU転送テスト
        if torch.cuda.is_available():
            device = torch.device("cuda")
            batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
            logger.info("✅ GPU transfer successful")
        
        break
    
    # ベンチマーク実行
    benchmark = DataLoaderBenchmark()
    results = benchmark.benchmark_dataloader(dataloader, num_batches=5)
    
    logger.info("✅ DataLoader test completed!")


if __name__ == "__main__":
    main()