"""
sat dataset loader
------------------

This module implements the `SATDataset` class which prepares several SAT
benchmark datasets as PyG `InMemoryDataset` objects. 

The class handles downloading SAT datasets and supplementing labels via csv files. 
"""

import gc
import logging
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch_geometric.transforms as T
from sklearn.decomposition import PCA
from torch_geometric.data import Data, HeteroData, InMemoryDataset
from torch_geometric.io import fs
from tqdm import tqdm

from graphbench.helpers.download import _download_and_unpack


# (0) Constants
SMALL_N_VARS = 3_000
MEDIUM_N_VARS = 20_000
# SMALL_N_CLAUSES = 2000_000
# SMALL_N_VARS = 500_000
# MAX_TIME = 60
MAX_TIME = 6000000000
# SMALL_N_VARS = 100000000_000
SMALL_N_CLAUSES = 15_000
MEDIUM_N_CLAUSES = 90_000



# (i) helper functions

# -----------------------------------------------------------------------------#
# (a) Utilities
# -----------------------------------------------------------------------------#

logger = logging.getLogger(__name__)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)



@dataclass(frozen=True)
class _SourceSpec:
    url: str
    raw_folder: str  # folder name inside tmp/ where data will appear


def _sinusoidal_positional_encoding(positions: np.ndarray, dim: int = 10) -> np.ndarray:
    """
    Compute sinusoidal positional encodings as in the GraSS paper.
    
    Args:
        positions: Array of positions (e.g., clause indices)
        dim: Embedding dimension (default 10 as in GraSS)
    
    Returns:
        Array of shape (len(positions), dim) with positional encodings
    """
    positions = np.asarray(positions).reshape(-1, 1)
    div_term = np.exp(np.arange(0, dim, 2) * -(np.log(10000.0) / dim))
    pe = np.zeros((len(positions), dim))
    pe[:, 0::2] = np.sin(positions * div_term)
    pe[:, 1::2] = np.cos(positions * div_term[:dim//2] if dim % 2 else div_term)
    return pe


# Try to import numba for JIT compilation, fall back to pure numpy if not available
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Create no-op decorator if numba is not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range


@jit(nopython=True, cache=True)
def _compute_clause_pos_neg(clauses_flat: np.ndarray, clause_offsets: np.ndarray, 
                            num_pos: np.ndarray, num_neg: np.ndarray) -> None:
    """JIT-compiled function to compute positive/negative literal counts per clause."""
    n_clauses = len(clause_offsets) - 1
    for i in range(n_clauses):
        start = clause_offsets[i]
        end = clause_offsets[i + 1]
        pos_count = 0
        neg_count = 0
        for j in range(start, end):
            if clauses_flat[j] > 0:
                pos_count += 1
            else:
                neg_count += 1
        num_pos[i] = pos_count
        num_neg[i] = neg_count


@jit(nopython=True, cache=True)
def _accumulate_vcg_features(clauses_flat: np.ndarray, clause_offsets: np.ndarray,
                              is_horn_arr: np.ndarray, is_unit_arr: np.ndarray,
                              is_binary_arr: np.ndarray, is_ternary_arr: np.ndarray,
                              clause_lens: np.ndarray,
                              var_features: np.ndarray, var_clause_length_sum: np.ndarray,
                              var_clause_count: np.ndarray,
                              edge_src: np.ndarray, edge_dst: np.ndarray, 
                              edge_attrs: np.ndarray) -> int:
    """
    JIT-compiled function to accumulate VCG variable features and build edges.
    Returns the number of edges created.
    """
    edge_idx = 0
    n_clauses = len(clause_offsets) - 1
    
    for i in range(n_clauses):
        start = clause_offsets[i]
        end = clause_offsets[i + 1]
        clause_len = clause_lens[i]
        is_horn = is_horn_arr[i]
        is_unit = is_unit_arr[i]
        is_binary = is_binary_arr[i]
        is_ternary = is_ternary_arr[i]
        
        # Check for tautology (both x and -x in same clause)
        has_tautology = False
        for j in range(start, end):
            lit = clauses_flat[j]
            for k in range(j + 1, end):
                if clauses_flat[k] == -lit:
                    has_tautology = True
                    break
            if has_tautology:
                break
        
        if has_tautology:
            continue
        
        for j in range(start, end):
            lit = clauses_flat[j]
            node_id = abs(lit) - 1
            
            # Store edge
            edge_src[edge_idx] = node_id
            edge_dst[edge_idx] = i
            edge_attrs[edge_idx] = 1.0 if lit > 0 else -1.0
            edge_idx += 1
            
            # Update variable features
            if is_horn:
                var_features[node_id, 2] += 1  # horn_count
            if lit > 0:
                var_features[node_id, 3] += 1  # positive_count
            else:
                var_features[node_id, 4] += 1  # negative_count
            var_features[node_id, 5] += 1  # degree
            if is_binary:
                var_features[node_id, 7] += 1
            if is_unit:
                var_features[node_id, 8] += 1
            if is_ternary:
                var_features[node_id, 9] += 1
            
            var_clause_length_sum[node_id] += clause_len
            var_clause_count[node_id] += 1
    
    return edge_idx


@jit(nopython=True, cache=True)
def _accumulate_lcg_features(clauses_flat: np.ndarray, clause_offsets: np.ndarray,
                              n_vars: int,
                              is_horn_arr: np.ndarray, is_unit_arr: np.ndarray,
                              is_binary_arr: np.ndarray, is_ternary_arr: np.ndarray,
                              clause_lens: np.ndarray,
                              lit_features: np.ndarray, lit_clause_length_sum: np.ndarray,
                              lit_clause_count: np.ndarray,
                              edge_src: np.ndarray, edge_dst: np.ndarray,
                              edge_attrs: np.ndarray) -> int:
    """
    JIT-compiled function to accumulate LCG literal features and build edges.
    Returns the number of edges created.
    """
    edge_idx = 0
    n_clauses = len(clause_offsets) - 1
    
    for i in range(n_clauses):
        start = clause_offsets[i]
        end = clause_offsets[i + 1]
        clause_len = clause_lens[i]
        is_horn = is_horn_arr[i]
        is_unit = is_unit_arr[i]
        is_binary = is_binary_arr[i]
        is_ternary = is_ternary_arr[i]
        
        # Process unique literals in clause
        for j in range(start, end):
            var = clauses_flat[j]
            var_idx = abs(var) - 1
            
            if var > 0:
                node_id = var_idx
                other_node_id = var_idx + n_vars
            else:
                node_id = var_idx + n_vars
                other_node_id = var_idx
            
            edge_src[edge_idx] = node_id
            edge_dst[edge_idx] = i
            edge_attrs[edge_idx] = 1.0 if var > 0 else -1.0
            edge_idx += 1
            
            # Update literal features
            if is_horn:
                lit_features[node_id, 2] += 1
            lit_features[node_id, 3] += 1  # occurrence_count
            lit_features[other_node_id, 4] += 1  # complementary_occurrence
            lit_features[node_id, 5] += 1  # degree
            if is_binary:
                lit_features[node_id, 7] += 1
            if is_unit:
                lit_features[node_id, 8] += 1
            if is_ternary:
                lit_features[node_id, 9] += 1
            
            lit_clause_length_sum[node_id] += clause_len
            lit_clause_count[node_id] += 1
    
    return edge_idx


@jit(nopython=True, cache=True)
def _accumulate_vg_features(clauses_flat: np.ndarray, clause_offsets: np.ndarray,
                            is_horn_arr: np.ndarray, is_unit_arr: np.ndarray,
                            is_binary_arr: np.ndarray, is_ternary_arr: np.ndarray,
                            clause_lens: np.ndarray,
                            var_features: np.ndarray, var_clause_count: np.ndarray,
                            var_clause_length_sum: np.ndarray,
                            edge_buffer: np.ndarray) -> int:
    """
    JIT-compiled function to accumulate VG variable features and build edges.
    Returns the number of unique edges created.
    """
    edge_idx = 0
    n_clauses = len(clause_offsets) - 1
    
    for clause_idx in range(n_clauses):
        start = clause_offsets[clause_idx]
        end = clause_offsets[clause_idx + 1]
        clause_len = clause_lens[clause_idx]
        is_horn = is_horn_arr[clause_idx]
        is_unit = is_unit_arr[clause_idx]
        is_binary = is_binary_arr[clause_idx]
        is_ternary = is_ternary_arr[clause_idx]
        
        # Get absolute variable indices for this clause
        n_lits = end - start
        
        # Track clause participation for each variable
        for j in range(start, end):
            lit = clauses_flat[j]
            var_id = abs(lit) - 1
            var_clause_count[var_id] += 1
            var_clause_length_sum[var_id] += clause_len
            if is_horn:
                var_features[var_id, 0] += 1
            if lit > 0:
                var_features[var_id, 1] += 1
            else:
                var_features[var_id, 2] += 1
            if is_binary:
                var_features[var_id, 5] += 1
            if is_unit:
                var_features[var_id, 6] += 1
            if is_ternary:
                var_features[var_id, 7] += 1
        
        # Create edges between co-occurring variables
        for i in range(start, end):
            a = abs(clauses_flat[i]) - 1
            for j in range(i + 1, end):
                b = abs(clauses_flat[j]) - 1
                if a == b:
                    continue
                
                # Store edge with canonical ordering
                if a < b:
                    edge_buffer[edge_idx, 0] = a
                    edge_buffer[edge_idx, 1] = b
                else:
                    edge_buffer[edge_idx, 0] = b
                    edge_buffer[edge_idx, 1] = a
                edge_idx += 1
                
                # Update co-occurrence degree
                var_features[a, 3] += 1
                var_features[b, 3] += 1
    
    return edge_idx


def _flatten_clauses(clauses: List) -> tuple:
    """
    Flatten a list of clauses into a single array with offset indices.
    This allows efficient JIT-compiled iteration.
    
    Returns:
        clauses_flat: 1D array of all literals
        clause_offsets: Array where clause i spans [offsets[i], offsets[i+1])
        clause_lens: Array of clause lengths
    """
    total_lits = sum(len(c) for c in clauses)
    clauses_flat = np.zeros(total_lits, dtype=np.int32)
    clause_offsets = np.zeros(len(clauses) + 1, dtype=np.int32)
    clause_lens = np.zeros(len(clauses), dtype=np.int32)
    
    offset = 0
    for i, clause in enumerate(clauses):
        clause_offsets[i] = offset
        clause_lens[i] = len(clause)
        for j, lit in enumerate(clause):
            clauses_flat[offset + j] = lit
        offset += len(clause)
    clause_offsets[len(clauses)] = offset
    
    return clauses_flat, clause_offsets, clause_lens

class SATDataset(InMemoryDataset):
    def __init__(
        self,
        name: str,
        split: str,
        root: Union[str, Path],
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        generate: Optional[bool] = False,
        use_satzilla_features: Optional[bool] = False,
        cleanup_raw: bool = True,
        solver: Optional[str] = None,
        normalize_targets: Optional[bool] = True,

        # TODO: This should be removed in the future -- the user will download these files
        load_preprocessed = False,):
        


        #currently downloads everything at once for a single dataset. Up to the user to manually unpack it so far

        self.SOURCES: Dict[str, _SourceSpec] = {
        "sat_lcg_as": _SourceSpec(
            url="https://huggingface.co/datasets/log-rwth-aachen/Graphbench_SAT/resolve/main/data_small_lcg_no_trans.pt.xz",
            raw_folder="sat_lcg",
        ), 
        "sat_vcg_as": _SourceSpec(
            url="https://huggingface.co/datasets/log-rwth-aachen/Graphbench_SAT/resolve/main/data_small_vcg_no_trans.pt.xz",
            raw_folder="sat_vcg",
        ),
        "sat_vg_as": _SourceSpec(
            url="https://huggingface.co/datasets/log-rwth-aachen/Graphbench_SAT/resolve/main/data_small_vg_no_trans.pt.xz",
            raw_folder="sat_vg",
        ),
        "sat_lcg_epm": _SourceSpec(
            url="https://huggingface.co/datasets/log-rwth-aachen/Graphbench_SAT/resolve/main/data_small_lcg_no_trans.pt.xz",
            raw_folder="sat_lcg",
        ),
        "sat_vcg_epm": _SourceSpec(
            url="https://huggingface.co/datasets/log-rwth-aachen/Graphbench_SAT/resolve/main/data_small_vcg_no_trans.pt.xz",
            raw_folder="sat_vcg",
        ),
        "sat_vg_epm": _SourceSpec(
            url="https://huggingface.co/datasets/log-rwth-aachen/Graphbench_SAT/resolve/main/data_small_vg_no_trans.pt.xz",
            raw_folder="sat_vg",
        ),
    }
        self.SOURCE_CSV = _SourceSpec(
            url="https://huggingface.co/datasets/log-rwth-aachen/Graphbench_SAT/resolve/main/sat_csv.zip",
            raw_folder="sat_csv",
        )
        self.name_temp = name.replace("_"," ")
        """
        Initialize a SATDataset instance.

        Parameters
        - name (str): Dataset identifier, e.g. 'sat_vg_as'.
        - split (str): One of 'train', 'val', 'test'.
        - root (str|Path): Root dataset directory.
        - use_satzilla_features (bool): Whether to include satzilla meta-features.
        - generate (bool): If True, attempt to generate dataset programmatically (slow).

        Behavior
        The constructor will ensure supplementary CSVs are present (downloading
        them if necessary), determine the dataset type and graph encoding, and
        then attempt to load a cached processed file. If not found, call
        `_prepare()` to build the dataset from raw files.
        """
        csv_dir = Path(root) / self.SOURCE_CSV.raw_folder
        if not csv_dir.exists():
            print(f"Downloading supplementary CSV files to {csv_dir}...")
            _download_and_unpack(source=self.SOURCE_CSV, raw_dir=csv_dir, processed_dir=csv_dir / "processed", logger=logger)
        self.solver = solver
        self.instances_csv = pd.read_csv(Path(root) /"sat_csv"/ "instances_new.csv")
        #self.dataset_name = self.name_temp.lower().split(" ")[0]
        self.task_type = self.name_temp.lower().split(" ")[2]
        self.graph_type = self.name_temp.lower().split(" ")[1]
        self.formula_sizes = "small" #only small formula sizes for now 
        self.use_satzilla_features = use_satzilla_features

        self.runs = pd.read_csv(Path(root) /"sat_csv"/ "runs.csv", index_col=0)
        if self.formula_sizes == "small":
            self.instances_csv = self.instances_csv[
                (self.instances_csv["n_vars"] < SMALL_N_VARS)
                & (self.instances_csv["n_clauses"] < SMALL_N_CLAUSES)]

        elif self.formula_sizes == "medium":
            self.instances_csv = self.instances_csv[
                (self.instances_csv["n_vars"] < MEDIUM_N_VARS)
                & (self.instances_csv["n_clauses"] < MEDIUM_N_CLAUSES)]
        if self.use_satzilla_features:
            self.features = pd.read_csv(Path(root) / "sat_csv" / "features.csv")
            self.features.set_index("filename", inplace=True)
            pca = PCA(n_components=7)
            pca.fit(self.features)
            self.features = pd.DataFrame(
                    pca.transform(self.features), index=self.features.index
                )
        self.normalize_targets = normalize_targets
        self.target_mean = None
        self.target_std = None
        
        if self.task_type == "as":
            runs = self.runs.copy()
            runs.loc[runs["time"] < 0.05, "time"] = 0.05
            runs.loc[~runs["status"].str.contains("SAT|UNSAT"), "time"] = 5000 * 10
            runs_all = self.instances_csv.merge(runs, on="filename")
            runs_all = runs_all.pivot_table(index="filename", columns="solver_name", values="time")
            self.order = runs_all.sum().sort_values().index.tolist()
            
            # Compute normalization stats from log-transformed targets
            if self.normalize_targets:
                log_times = np.log10(runs_all.values)
                self.target_mean = float(np.nanmean(log_times))
                self.target_std = float(np.nanstd(log_times)) + 1e-6
        
        elif self.task_type == "epm" and self.normalize_targets:
            # Compute normalization stats for EPM task
            runs = self.runs.copy()
            if self.solver:
                solver_runs = runs[runs["solver_name"] == self.solver].copy()
            else:
                solver_runs = runs.copy()
            solver_runs.loc[solver_runs["time"] < 0.05, "time"] = 0.05
            solver_runs.loc[~solver_runs["status"].str.contains("SAT|UNSAT"), "time"] = 50_000
            log_times = np.log10(solver_runs["time"].values)
            self.target_mean = float(np.nanmean(log_times))
            self.target_std = float(np.nanstd(log_times)) + 1e-6
        
        self.name = name.lower()
        if self.name not in self.SOURCES:
            raise ValueError(f"Unsupported dataset name: {self.name}")
        assert split in ["train", "val", "test"], "Only 'train', 'val', 'test' splits are supported."


        self.generate = generate
        self.split = split
        self.source = self.SOURCES[self.name]
        self.cleanup_raw = cleanup_raw
        self.load_preprocessed = load_preprocessed

        # paths
        self.sat_dir = Path(root) / "sat"
        self._raw_dir = (self.sat_dir / self.SOURCES[self.name].raw_folder / "raw" )
        # Include time window & task in the processed filename to avoid collisions
        self.processed_path = self.sat_dir / self.SOURCES[self.name].raw_folder / "processed"
        self.graphs_dir = self.sat_dir / self.SOURCES[self.name].raw_folder 
        super().__init__(str(self.graphs_dir), transform, pre_transform)

        # process data if needed
        if self.processed_path.exists():
            self.load(self.processed_paths[0])
            return

        #self._prepare()  # (i) downloads, unpacks, load data + (ii) timestep handle + (e) subgraph + collate
        if self.cleanup_raw:
            self._cleanup()

    def create_variable_clause_graph(self, clauses, n_vars):
        """
        Create a variable-clause graph (VCG) with enhanced features inspired by GraSS.
        
        Variable features (12 dims):
            0: constant (placeholder, set to 0)
            1: constant (placeholder, set to 0)
            2: horn_count (number of Horn clauses this variable appears in)
            3: positive_count (appearances as positive literal)
            4: negative_count (appearances as negative literal)
            5: degree (total appearances)
            6: pos_neg_ratio (positive / negative appearances)
            7: binary_clause_count (appearances in binary clauses)
            8: unit_clause_count (appearances in unit clauses)
            9: ternary_clause_count (appearances in ternary clauses)
            10: normalized_freq (degree / n_clauses)
            11: avg_clause_length (average length of clauses containing this variable)
        
        Clause features (22 dims):
            0-9: sinusoidal positional encoding (10 dims)
            10: num_pos_literals
            11: num_neg_literals
            12: pos_neg_ratio
            13: clause_length
            14: is_unit (length == 1)
            15: is_binary (length == 2)
            16: is_ternary (length == 3)
            17: is_horn (at most 1 positive literal)
            18: normalized_length (clause_length / max_clause_length)
            19: clause_position_normalized (clause_idx / n_clauses)
            20: is_definite_horn (exactly 1 positive literal)
            21: is_negative_clause (all literals negative)
        """
        data = HeteroData()
        n_clauses = len(clauses)
        
        if n_clauses == 0:
            data["var"].x = torch.zeros((n_vars, 12), dtype=torch.float)
            data["clause"].x = torch.zeros((0, 22), dtype=torch.float)
            data["var", "in", "clause"].edge_index = torch.zeros((2, 0), dtype=torch.long)
            data["var", "in", "clause"].edge_attr = torch.zeros((0, 1), dtype=torch.float)
            data["clause", "contains", "var"].edge_index = torch.zeros((2, 0), dtype=torch.long)
            data["clause", "contains", "var"].edge_attr = torch.zeros((0, 1), dtype=torch.float)
            # Don't set num_edges for HeteroData - PyG computes it from edge indices
            return data
        
        # Flatten clauses for JIT-compiled processing
        clauses_flat, clause_offsets, clause_lens = _flatten_clauses(clauses)
        total_lits = len(clauses_flat)
        max_clause_length = clause_lens.max()
        
        # Pre-allocate arrays for clause features (computed by JIT function)
        num_pos_arr = np.zeros(n_clauses, dtype=np.int32)
        num_neg_arr = np.zeros(n_clauses, dtype=np.int32)
        
        # Use JIT-compiled function to compute pos/neg counts
        _compute_clause_pos_neg(clauses_flat, clause_offsets, num_pos_arr, num_neg_arr)
        
        # Compute clause-level boolean masks
        is_horn_arr = num_pos_arr <= 1
        is_unit_arr = clause_lens == 1
        is_binary_arr = clause_lens == 2 
        is_ternary_arr = clause_lens == 3
        is_definite_horn_arr = num_pos_arr == 1
        is_negative_arr = num_pos_arr == 0
        
        # Initialize clause features tensor and populate in bulk
        clause_x = np.zeros((n_clauses, 22), dtype=np.float32)
        clause_x[:, :10] = _sinusoidal_positional_encoding(np.arange(n_clauses), dim=10)
        clause_x[:, 10] = num_pos_arr
        clause_x[:, 11] = num_neg_arr
        clause_x[:, 12] = num_pos_arr / (num_neg_arr + 1e-6)
        clause_x[:, 13] = clause_lens
        clause_x[:, 14] = is_unit_arr.astype(np.float32)
        clause_x[:, 15] = is_binary_arr.astype(np.float32)
        clause_x[:, 16] = is_ternary_arr.astype(np.float32)
        clause_x[:, 17] = is_horn_arr.astype(np.float32)
        clause_x[:, 18] = clause_lens / max_clause_length
        clause_x[:, 19] = np.arange(n_clauses) / max(n_clauses - 1, 1)
        clause_x[:, 20] = is_definite_horn_arr.astype(np.float32)
        clause_x[:, 21] = is_negative_arr.astype(np.float32)
        
        data["clause"].x = torch.from_numpy(clause_x)
        
        # Initialize variable feature accumulators
        var_features = np.zeros((n_vars, 12), dtype=np.float32)
        var_clause_length_sum = np.zeros(n_vars, dtype=np.float32)
        var_clause_count = np.zeros(n_vars, dtype=np.int32)
        
        # Pre-allocate edge arrays
        edge_src = np.zeros(total_lits, dtype=np.int64)
        edge_dst = np.zeros(total_lits, dtype=np.int64)
        edge_attrs = np.zeros(total_lits, dtype=np.float32)
        
        # Use JIT-compiled function for feature accumulation and edge building
        edge_idx = _accumulate_vcg_features(
            clauses_flat, clause_offsets,
            is_horn_arr.view(np.uint8), is_unit_arr.view(np.uint8),
            is_binary_arr.view(np.uint8), is_ternary_arr.view(np.uint8),
            clause_lens, var_features, var_clause_length_sum, var_clause_count,
            edge_src, edge_dst, edge_attrs
        )
        
        # Trim edge arrays to actual size
        edge_src = edge_src[:edge_idx]
        edge_dst = edge_dst[:edge_idx]
        edge_attrs = edge_attrs[:edge_idx]
        
        # Compute derived variable features in bulk
        pos_count = var_features[:, 3]
        neg_count = var_features[:, 4]
        degree = var_features[:, 5]
        
        var_features[:, 6] = pos_count / (neg_count + 1e-6)  # pos_neg_ratio
        var_features[:, 10] = degree / max(n_clauses, 1)  # normalized_freq
        
        # avg_clause_length (only where variable appears)
        mask = var_clause_count > 0
        var_features[mask, 11] = var_clause_length_sum[mask] / var_clause_count[mask]
        
        data["var"].x = torch.from_numpy(var_features)
        
        # Create edge tensors
        edge_index = torch.from_numpy(np.stack([edge_src, edge_dst], axis=0))
        edge_attr = torch.from_numpy(edge_attrs).reshape(-1, 1)
        
        data["var", "in", "clause"].edge_index = edge_index
        data["var", "in", "clause"].edge_attr = edge_attr
        
        # Create clause -> variable edges (reverse)
        data["clause", "contains", "var"].edge_index = edge_index.flip(0)
        data["clause", "contains", "var"].edge_attr = edge_attr.clone()

        # Don't set num_nodes/num_edges for HeteroData - PyG computes them from node/edge indices
        return data


    def create_literal_clause_graph(self, clauses, n_vars):
        """
        Create a literal-clause graph (LCG) with enhanced features from the GraSS paper.
        
        Literal features (12 dims):
            0: is_positive (1 for positive literal, 0 for negative)
            1: is_negative (0 for positive literal, 1 for negative)
            2: horn_count (number of Horn clauses this literal appears in)
            3: occurrence_count (total appearances in clauses)
            4: complementary_occurrence (appearances of the complementary literal)
            5: degree (same as occurrence_count, for compatibility)
            6: pos_neg_ratio (occurrence / complementary_occurrence)
            7: binary_clause_count (appearances in binary clauses)
            8: unit_clause_count (appearances in unit clauses)
            9: ternary_clause_count (appearances in ternary clauses)
            10: normalized_freq (occurrence_count / n_clauses)
            11: avg_clause_length (average length of clauses containing this literal)
        
        Clause features (22 dims):
            0-9: sinusoidal positional encoding (10 dims, from GraSS paper)
            10: num_pos_literals
            11: num_neg_literals  
            12: pos_neg_ratio
            13: clause_length
            14: is_unit (length == 1)
            15: is_binary (length == 2)
            16: is_ternary (length == 3)
            17: is_horn (at most 1 positive literal)
            18: normalized_length (clause_length / max_clause_length)
            19: clause_position_normalized (clause_idx / n_clauses)
            20: is_definite_horn (exactly 1 positive literal)
            21: is_negative_clause (all literals negative)
        """
        data = HeteroData()
        n_clauses = len(clauses)
        
        if n_clauses == 0:
            data["literal"].x = torch.zeros((n_vars * 2, 12), dtype=torch.float)
            data["literal"].x[:n_vars, 0] = 1.0
            data["literal"].x[n_vars:, 1] = 1.0
            data["clause"].x = torch.zeros((0, 22), dtype=torch.float)
            data["literal", "in", "clause"].edge_index = torch.zeros((2, 0), dtype=torch.long)
            data["literal", "in", "clause"].edge_attr = torch.zeros((0, 1), dtype=torch.float)
            data["clause", "contains", "literal"].edge_index = torch.zeros((2, 0), dtype=torch.long)
            data["clause", "contains", "literal"].edge_attr = torch.zeros((0, 1), dtype=torch.float)
            lit_edges = np.stack([np.arange(n_vars), np.arange(n_vars, 2 * n_vars)], axis=0)
            lit_edges = np.concatenate([lit_edges, lit_edges[::-1]], axis=1)
            data["literal", "complement", "literal"].edge_index = torch.from_numpy(lit_edges.astype(np.int64))
            # Complement edges get a constant edge_attr (0.0) for consistency with other edge types
            data["literal", "complement", "literal"].edge_attr = torch.zeros((2 * n_vars, 1), dtype=torch.float)
            # Don't set num_edges for HeteroData - PyG computes it from edge indices
            return data
        
        # Flatten clauses for JIT-compiled processing
        clauses_flat, clause_offsets, clause_lens = _flatten_clauses(clauses)
        total_lits = len(clauses_flat)
        max_clause_length = clause_lens.max()
        
        # Pre-allocate arrays for clause features (computed by JIT function)
        num_pos_arr = np.zeros(n_clauses, dtype=np.int32)
        num_neg_arr = np.zeros(n_clauses, dtype=np.int32)
        
        # Use JIT-compiled function to compute pos/neg counts
        _compute_clause_pos_neg(clauses_flat, clause_offsets, num_pos_arr, num_neg_arr)
        
        # Compute clause-level boolean masks
        is_horn_arr = num_pos_arr <= 1
        is_unit_arr = clause_lens == 1
        is_binary_arr = clause_lens == 2
        is_ternary_arr = clause_lens == 3
        is_definite_horn_arr = num_pos_arr == 1
        is_negative_arr = num_pos_arr == 0
        
        # Build clause features in bulk using numpy
        clause_x = np.zeros((n_clauses, 22), dtype=np.float32)
        clause_x[:, :10] = _sinusoidal_positional_encoding(np.arange(n_clauses), dim=10)
        clause_x[:, 10] = num_pos_arr
        clause_x[:, 11] = num_neg_arr
        clause_x[:, 12] = num_pos_arr / (num_neg_arr + 1e-6)
        clause_x[:, 13] = clause_lens
        clause_x[:, 14] = is_unit_arr.astype(np.float32)
        clause_x[:, 15] = is_binary_arr.astype(np.float32)
        clause_x[:, 16] = is_ternary_arr.astype(np.float32)
        clause_x[:, 17] = is_horn_arr.astype(np.float32)
        clause_x[:, 18] = clause_lens / max_clause_length
        clause_x[:, 19] = np.arange(n_clauses) / max(n_clauses - 1, 1)
        clause_x[:, 20] = is_definite_horn_arr.astype(np.float32)
        clause_x[:, 21] = is_negative_arr.astype(np.float32)
        
        data["clause"].x = torch.from_numpy(clause_x)
        
        # Initialize literal feature accumulators
        lit_features = np.zeros((n_vars * 2, 12), dtype=np.float32)
        lit_features[:n_vars, 0] = 1.0  # is_positive
        lit_features[n_vars:, 1] = 1.0  # is_negative
        
        lit_clause_length_sum = np.zeros(n_vars * 2, dtype=np.float32)
        lit_clause_count = np.zeros(n_vars * 2, dtype=np.int32)
        
        # Pre-allocate edge arrays
        edge_src = np.zeros(total_lits, dtype=np.int64)
        edge_dst = np.zeros(total_lits, dtype=np.int64)
        edge_attrs = np.zeros(total_lits, dtype=np.float32)
        
        # Use JIT-compiled function for feature accumulation and edge building
        edge_idx = _accumulate_lcg_features(
            clauses_flat, clause_offsets, n_vars,
            is_horn_arr.view(np.uint8), is_unit_arr.view(np.uint8),
            is_binary_arr.view(np.uint8), is_ternary_arr.view(np.uint8),
            clause_lens, lit_features, lit_clause_length_sum, lit_clause_count,
            edge_src, edge_dst, edge_attrs
        )
        
        # Trim edge arrays
        edge_src = edge_src[:edge_idx]
        edge_dst = edge_dst[:edge_idx]
        edge_attrs = edge_attrs[:edge_idx]
        
        # Compute derived literal features in bulk
        occ = lit_features[:, 3]
        comp_occ = lit_features[:, 4]
        lit_features[:, 6] = occ / (comp_occ + 1e-6)
        lit_features[:, 10] = occ / max(n_clauses, 1)
        
        mask = lit_clause_count > 0
        lit_features[mask, 11] = lit_clause_length_sum[mask] / lit_clause_count[mask]
        
        data["literal"].x = torch.from_numpy(lit_features)
        
        # Create edge tensors
        edge_index = torch.from_numpy(np.stack([edge_src, edge_dst], axis=0))
        edge_attr = torch.from_numpy(edge_attrs).reshape(-1, 1)
        
        data["literal", "in", "clause"].edge_index = edge_index
        data["literal", "in", "clause"].edge_attr = edge_attr
        data["clause", "contains", "literal"].edge_index = edge_index.flip(0)
        data["clause", "contains", "literal"].edge_attr = edge_attr.clone()
        
        # Create complement edges using numpy
        lit_edges = np.stack([np.arange(n_vars), np.arange(n_vars, 2 * n_vars)], axis=0)
        lit_edges = np.concatenate([lit_edges, lit_edges[::-1]], axis=1)
        data["literal", "complement", "literal"].edge_index = torch.from_numpy(lit_edges.astype(np.int64))
        # Complement edges get a constant edge_attr (0.0) for consistency with other edge types
        data["literal", "complement", "literal"].edge_attr = torch.zeros((2 * n_vars, 1), dtype=torch.float)

        # Don't set num_nodes/num_edges for HeteroData - PyG computes them from node/edge indices
        return data


    def create_variable_graph(self, clauses, n_vars):
        """
        Create a variable graph (VG) with enhanced features inspired by GraSS.
        
        In this graph, variables are nodes and edges connect variables that 
        appear together in at least one clause.
        
        Variable features (12 dims):
            0: horn_count (number of Horn clauses this variable appears in)
            1: positive_count (appearances as positive literal)
            2: negative_count (appearances as negative literal)
            3: degree (total co-occurrence count)
            4: pos_neg_ratio (positive / negative appearances)
            5: binary_clause_count (appearances in binary clauses)
            6: unit_clause_count (appearances in unit clauses)
            7: ternary_clause_count (appearances in ternary clauses)
            8: normalized_freq (clause appearances / n_clauses)
            9: avg_clause_length (average length of clauses containing this variable)
            10: total_clause_appearances (number of distinct clauses)
            11: avg_co_occurrence_degree (average co-occurrence per clause)
        """
        n_clauses = len(clauses)
        
        if n_clauses == 0:
            data = Data(edge_index=torch.zeros((2, 0), dtype=torch.long))
            data.x = torch.zeros((n_vars, 12), dtype=torch.float)
            data.num_nodes = n_vars
            data.num_edges = 0
            return data
        
        # Flatten clauses for JIT-compiled processing
        clauses_flat, clause_offsets, clause_lens = _flatten_clauses(clauses)
        
        # Pre-allocate arrays for clause features (computed by JIT function)
        num_pos_arr = np.zeros(n_clauses, dtype=np.int32)
        num_neg_arr = np.zeros(n_clauses, dtype=np.int32)
        
        # Use JIT-compiled function to compute pos/neg counts
        _compute_clause_pos_neg(clauses_flat, clause_offsets, num_pos_arr, num_neg_arr)
        
        is_horn_arr = num_pos_arr <= 1
        is_unit_arr = clause_lens == 1
        is_binary_arr = clause_lens == 2
        is_ternary_arr = clause_lens == 3
        
        # Use numpy arrays for variable features
        var_features = np.zeros((n_vars, 12), dtype=np.float32)
        var_clause_count = np.zeros(n_vars, dtype=np.int32)
        var_clause_length_sum = np.zeros(n_vars, dtype=np.float32)
        
        # Estimate max edges: sum of C(clause_len, 2) for all clauses
        max_edges = sum(int(l * (l - 1) / 2) for l in clause_lens)
        edge_buffer = np.zeros((max_edges, 2), dtype=np.int64)
        
        # Use JIT-compiled function for feature accumulation and edge building
        n_edges = _accumulate_vg_features(
            clauses_flat, clause_offsets,
            is_horn_arr.view(np.uint8), is_unit_arr.view(np.uint8),
            is_binary_arr.view(np.uint8), is_ternary_arr.view(np.uint8),
            clause_lens, var_features, var_clause_count, var_clause_length_sum,
            edge_buffer
        )
        
        # Compute derived features in bulk
        pos_count = var_features[:, 1]
        neg_count = var_features[:, 2]
        var_features[:, 4] = pos_count / (neg_count + 1e-6)  # pos_neg_ratio
        var_features[:, 8] = var_clause_count / max(n_clauses, 1)  # normalized_freq
        
        mask = var_clause_count > 0
        var_features[mask, 9] = var_clause_length_sum[mask] / var_clause_count[mask]  # avg_clause_length
        var_features[:, 10] = var_clause_count  # total_clause_appearances
        var_features[mask, 11] = var_features[mask, 3] / var_clause_count[mask]  # avg_co_occurrence_degree
        
        # Deduplicate edges and convert to tensor
        if n_edges > 0:
            edge_buffer = edge_buffer[:n_edges]
            # Use numpy unique to deduplicate
            unique_edges = np.unique(edge_buffer, axis=0)
            # Add reverse edges to make graph undirected
            reverse_edges = unique_edges[:, ::-1]
            all_edges = np.concatenate([unique_edges, reverse_edges], axis=0)
            edge_index = torch.from_numpy(all_edges.T.copy()).contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        data = Data(edge_index=edge_index)
        data.x = torch.from_numpy(var_features)

        data.num_nodes = n_vars
        data.num_edges = edge_index.size(1)

        return data


    def create_clause_graph(self, clauses, n_vars):
        """
        Create a clause graph (CG) with enhanced features inspired by GraSS.
        
        In this graph, clauses are nodes and edges connect clauses that share
        a complementary literal (variable appears positive in one, negative in other).
        
        Clause features (22 dims, same as VCG/LCG for consistency):
            0-9: sinusoidal positional encoding (10 dims)
            10: num_pos_literals
            11: num_neg_literals
            12: pos_neg_ratio
            13: clause_length
            14: is_unit (length == 1)
            15: is_binary (length == 2)
            16: is_ternary (length == 3)
            17: is_horn (at most 1 positive literal)
            18: normalized_length (clause_length / max_clause_length)
            19: clause_position_normalized (clause_idx / n_clauses)
            20: is_definite_horn (exactly 1 positive literal)
            21: is_negative_clause (all literals negative)
        """
        n_clauses = len(clauses)
        
        if n_clauses == 0:
            data = Data(edge_index=torch.zeros((2, 0), dtype=torch.long))
            data.x = torch.zeros((0, 22), dtype=torch.float)
            data.num_nodes = 0
            data.num_edges = 0
            return data
        
        # Flatten clauses for JIT-compiled processing
        clauses_flat, clause_offsets, clause_lens = _flatten_clauses(clauses)
        max_clause_length = clause_lens.max()
        
        # Pre-allocate arrays for clause features (computed by JIT function)
        num_pos_arr = np.zeros(n_clauses, dtype=np.int32)
        num_neg_arr = np.zeros(n_clauses, dtype=np.int32)
        
        # Use JIT-compiled function to compute pos/neg counts
        _compute_clause_pos_neg(clauses_flat, clause_offsets, num_pos_arr, num_neg_arr)
        
        is_horn_arr = num_pos_arr <= 1
        is_unit_arr = clause_lens == 1
        is_binary_arr = clause_lens == 2
        is_ternary_arr = clause_lens == 3
        is_definite_horn_arr = num_pos_arr == 1
        is_negative_arr = num_pos_arr == 0
        
        # Build clause features in bulk using numpy
        x = np.zeros((n_clauses, 22), dtype=np.float32)
        x[:, :10] = _sinusoidal_positional_encoding(np.arange(n_clauses), dim=10)
        x[:, 10] = num_pos_arr
        x[:, 11] = num_neg_arr
        x[:, 12] = num_pos_arr / (num_neg_arr + 1e-6)
        x[:, 13] = clause_lens
        x[:, 14] = is_unit_arr.astype(np.float32)
        x[:, 15] = is_binary_arr.astype(np.float32)
        x[:, 16] = is_ternary_arr.astype(np.float32)
        x[:, 17] = is_horn_arr.astype(np.float32)
        x[:, 18] = clause_lens / max_clause_length
        x[:, 19] = np.arange(n_clauses) / max(n_clauses - 1, 1)
        x[:, 20] = is_definite_horn_arr.astype(np.float32)
        x[:, 21] = is_negative_arr.astype(np.float32)
        
        start = time.time()
        # Use defaultdict for faster literal-to-clause mapping
        clauses_for_lits = defaultdict(list)
        edges = []

        for cid, clause in enumerate(clauses):
            if cid % 10000 == 0 and cid > 0:
                if time.time() - start > MAX_TIME:
                    raise TimeoutError("Timeout during graph creation")
            
            # Build clause-to-literal mapping for edge detection
            for var in clause:
                neg_var = -var
                
                # If opposite polarity exists in another clause, create edge
                if neg_var in clauses_for_lits:
                    for nc in clauses_for_lits[neg_var]:
                        if nc < cid:
                            edges.append((nc, cid))

                clauses_for_lits[var].append(cid)

        # Convert edges to tensor efficiently
        if edges:
            edge_arr = np.array(edges, dtype=np.int64)
            edge_index = torch.from_numpy(edge_arr.T).contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        data = Data(edge_index=edge_index)
        data.x = torch.from_numpy(x)

        data.num_nodes = n_clauses
        data.num_edges = edge_index.size(1)

        return data


    # function from https://github.com/zhaoyu-li/G4SATBench
    def parse_cnf_file(self,file_path):
        with open(file_path, "r") as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            tokens = lines[i].strip().split()
            if len(tokens) < 1 or tokens[0] != "p":
                i += 1
            else:
                break

        if i == len(lines):
            return 0, []

        header = lines[i].strip().split()
        n_vars = int(header[2])
        clauses = []

        for line in lines[i + 1 :]:
            tokens = line.strip().split()
            clause = [int(s) for s in tokens[:-1]]
            clauses.append(clause)

        return n_vars, clauses


    def process_file(self,instance, graph_type, pre_transform=None):
    
        gc.collect()
        original_file_path = Path("/storage/work/graph_bench/dataset/raw_preprocessed") / instance["filename"]


        n_vars, clauses = self.parse_cnf_file(original_file_path)

        if graph_type == "vcg":
            data = self.create_variable_clause_graph(clauses, n_vars)
        elif graph_type == "cg":
            data = self.create_clause_graph(clauses, n_vars)
        elif graph_type == "lcg":
            data = self.create_literal_clause_graph(clauses, n_vars)
        elif graph_type == "vg":
            data = self.create_variable_graph(clauses, n_vars)
            
        if pre_transform is not None:
            data = pre_transform(data)
        
        fs.torch_save(data, os.path.join(tempfile.gettempdir(), f"{instance['filename']}.pt"))

    def get(self, idx):
        data = super().get(idx)
        assert data.is_undirected()
        instance = self.instances_csv.iloc[idx]
        times = self.runs.loc[instance["filename"]]

        if self.use_satzilla_features:
            features = self.features.loc[instance["filename"]]
            features = [features.to_list()] * data.x.size(0)
            feat_tensor = torch.tensor(features, dtype=torch.bfloat16)

        if self.task_type == "epm":
            y = times[times["solver_name"] == self.solver]["time"].values[0]

            if y < 0.05:
                y = 0.05

            status = times[times["solver_name"] == self.solver]["status"].values
            if status not in ["SAT", "UNSAT"]:
                y = 50_000

            y = np.log10(y)
            
            # Optionally normalize using precomputed stats
            if self.normalize_targets and self.target_mean is not None:
                y = (y - self.target_mean) / self.target_std
            
            if self.use_satzilla_features:
                data.x = feat_tensor.reshape(-1, 1)
            data.y = torch.tensor([y], dtype=torch.bfloat16)

            return data
        elif self.task_type == "as":
            y = []
            for solver in self.order:
                t = times[times["solver_name"] == solver]["time"].values[0]
                if t < 0.05:
                    t = 0.05
                status = times[times["solver_name"] == solver]["status"].values[0]
                if status not in ["SAT", "UNSAT"]:
                    t = 50_000
                y.append(t)
            y = np.array(y, dtype=np.float32)
            
            # Log-transform to compress the range (0.05s -> 50000s becomes ~-1.3 -> ~4.7)
            y = np.log10(y)
            
            # Optionally normalize using precomputed stats
            if self.normalize_targets and self.target_mean is not None:
                y = (y - self.target_mean) / self.target_std
            
            if self.use_satzilla_features:
                data.x = feat_tensor.reshape(-1, 1)
            data.y = torch.tensor(y, dtype=torch.bfloat16).unsqueeze(0)

            return data

    def denormalize_targets(self, y: torch.Tensor) -> torch.Tensor:
        """Convert normalized predictions back to log10(seconds) scale.
        
        Args:
            y: Normalized predictions from model
            
        Returns:
            Denormalized predictions in log10(seconds) scale
        """
        if not self.normalize_targets or self.target_mean is None:
            return y
        return y * self.target_std + self.target_mean
    
    def denormalize_to_seconds(self, y: torch.Tensor) -> torch.Tensor:
        """Convert normalized predictions back to seconds.
        
        Args:
            y: Normalized predictions from model
            
        Returns:
            Denormalized predictions in seconds (not log10)
        """
        log_y = self.denormalize_targets(y)
        return torch.pow(10, log_y)
    
    def get_normalization_stats(self) -> dict:
        """Get the normalization statistics used for targets.
        
        Returns:
            Dictionary with 'mean' and 'std' keys, or None if normalization is disabled.
        """
        if not self.normalize_targets or self.target_mean is None:
            return None
        return {"mean": self.target_mean, "std": self.target_std}

    def _generate(self) -> None:
        futures = []
        #generate the corresponding sat dataset
        with ProcessPoolExecutor(max_workers=64) as executor:
            cnt = 0
            for _, instance in tqdm(self.instances_csv.iterrows()):
                futures.append(executor.submit(self.process_file, instance.to_dict(), self.graph_type, self.pre_transform))
                cnt += 1
                # self.process_file(instance.to_dict(), self.graph_type, self.pre_transform, True)
            # futures = [
            #     executor.submit(process_file, instance.to_dict(), self.graph_type)
            #     for _, instance in self.instances_csv.iterrows()
            # ]

            print("Waiting for results...", flush=True)
            graphs = []
            for i, f in enumerate(tqdm(futures)):
                try:
                    f.result()
                except Exception as e:
                    file = self.instances_csv.iloc[i]
                    print(file, flush=True)
                    print(f"Error processing file: {e}")
                    import traceback

                    traceback.print_exc()
                    print("", flush=True)
                    raise e
            
        print("Combining results...", flush=True)
        graphs = [fs.torch_load(os.path.join(tempfile.gettempdir(), f"{instance['filename']}.pt")) for _, instance in self.instances_csv.iterrows()]
        
        return graphs 

    def _prepare(self) -> None:
        print("Processing...", flush=True)

        # (b) Download & unpack helpers

        if self.generate:
            data_list = self._generate()
            if self.pre_transform is not None:
               data_list = [self.pre_transform(d) for d in data_list]
            self.save(data_list, self.processed_paths[0])
            logger.info(f"Saved processed dataset -> {self.processed_path}")
        # else:
        #     _download_and_unpack(source=self.source, raw_dir=self._raw_dir, processed_dir=self.processed_path, logger=logger)

        #     loader = self._load_sat_graphs
        #     loader_kwargs = {}
        #     loader(**loader_kwargs)
        #     data_list = [self.get(i) for i in range(len(self))]
        #     if self.pre_transform is not None:
        #         data_list = [self.pre_transform(d) for d in data_list]
        


    def _cleanup(self) -> None:
        if self._raw_dir.exists():
            logger.info(f"Cleaning up: {self._raw_dir}")
            # remove only the dataset-specific temp folder
            for p in sorted(self._raw_dir.rglob("*"), reverse=True):
                try:
                    p.unlink()
                except IsADirectoryError:
                    pass
            try:
                self._raw_dir.rmdir()
            except OSError:
                # not empty due to shared artifacts; leave it
                pass

    def _load_sat_graphs(self) -> List[Data]:
        filepaths = self._find_matching_files(directory=self._raw_dir, size=self.formula_sizes, graph_type=self.graph_type)
        self.load(filepaths[0])
        return 

    def _find_matching_files(self,directory, size, graph_type):
        """
        Returns a list of filenames matching the convention in the directory.
        """

        pattern = f"data_{size}_{graph_type}_features.pt"
        return [os.path.join(directory, fname)
                for fname in os.listdir(directory)
                if fname == pattern]

    # --- InMemoryDataset API (not used directly but kept for PyG hygiene) -----

    @property
    def raw_file_names(self) -> List[str]:  # unused, we drive our own cache
        return []

    @property
    def processed_file_names(self) -> List[str]:  # unused, we drive our own cache
        return ["data.pt"]

    def process(self):
        self._prepare()
        return 