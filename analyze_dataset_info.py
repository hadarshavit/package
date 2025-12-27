"""
Analyze SAT dataset information.

This script creates SATDataset instances (with generate=False) and extracts:
- Node features and their dimensions (data.x / data.node_type specific features)
- Edge features and their dimensions (data.edge_attr)
- Targets and target dimensions (data.y)
- Other important attributes in the dataset
"""

import argparse
from pathlib import Path
import numpy as np
from torch_geometric.utils import to_undirected
from graphbench.datasets.sat import SATDataset


def analyze_hetero_data(data, name):
    """Analyze a HeteroData object (VCG or LCG)."""
    print(f"\n{'='*60}")
    print(f"Dataset: {name}")
    print(f"{'='*60}")
    
    print(f"\nData type: HeteroData")
    
    # Node types and features
    print(f"\n--- Node Types and Features ---")
    for node_type in data.node_types:
        node_store = data[node_type]
        if hasattr(node_store, 'x') and node_store.x is not None:
            x = node_store.x
            print(f"\n  '{node_type}' nodes:")
            print(f"    - Number of nodes: {x.shape[0]}")
            print(f"    - Feature dimension: {x.shape[1]}")
            print(f"    - data['{node_type}'].x shape: {tuple(x.shape)}")
            print(f"    - dtype: {x.dtype}")
    
    # Edge types and features
    print(f"\n--- Edge Types and Features ---")
    for edge_type in data.edge_types:
        edge_store = data[edge_type]
        src, rel, dst = edge_type
        
        print(f"\n  Edge type: ('{src}', '{rel}', '{dst}')")
        
        if hasattr(edge_store, 'edge_index') and edge_store.edge_index is not None:
            edge_index = edge_store.edge_index
            print(f"    - Number of edges: {edge_index.shape[1]}")
        
        if hasattr(edge_store, 'edge_attr') and edge_store.edge_attr is not None:
            edge_attr = edge_store.edge_attr
            print(f"    - Edge attr shape: {tuple(edge_attr.shape)}")
            print(f"    - Edge attr dimension: {edge_attr.shape[1] if edge_attr.dim() > 1 else 1}")
            print(f"    - dtype: {edge_attr.dtype}")
        else:
            print(f"    - Edge attr: None (no edge features)")
    
    # After conversion to homogeneous
    print(f"\n--- After to_homogeneous() conversion ---")
    assert data.is_undirected()
    homo_data = data.to_homogeneous()
    # Note: homo_data may not be strictly undirected after conversion due to 
    # how PyG merges edge types with different edge_attr values
    print(f"  - data.x shape: {tuple(homo_data.x.shape)}")
    print(f"  - data.edge_index shape: {tuple(homo_data.edge_index.shape)}")
    if hasattr(homo_data, 'edge_attr') and homo_data.edge_attr is not None:
        print(f"  - data.edge_attr shape: {tuple(homo_data.edge_attr.shape)}")
    if hasattr(homo_data, 'node_type') and homo_data.node_type is not None:
        print(f"  - data.node_type shape: {tuple(homo_data.node_type.shape)}")
    if hasattr(homo_data, 'edge_type') and homo_data.edge_type is not None:
        print(f"  - data.edge_type shape: {tuple(homo_data.edge_type.shape)}")
    
    return homo_data


def analyze_homo_data(data, name):
    """Analyze a homogeneous Data object (VG or CG)."""
    print(f"\n{'='*60}")
    print(f"Dataset: {name}")
    print(f"{'='*60}")
    
    print(f"\nData type: Data (homogeneous)")
    
    # Node features
    print(f"\n--- Node Features ---")
    if hasattr(data, 'x') and data.x is not None:
        x = data.x
        print(f"  - data.x shape: {tuple(x.shape)}")
        print(f"  - Number of nodes: {x.shape[0]}")
        print(f"  - Feature dimension: {x.shape[1]}")
        print(f"  - dtype: {x.dtype}")
    else:
        print(f"  - data.x: None")
    
    # Edge features
    print(f"\n--- Edge Features ---")
    if hasattr(data, 'edge_index') and data.edge_index is not None:
        print(f"  - data.edge_index shape: {tuple(data.edge_index.shape)}")
        print(f"  - Number of edges: {data.edge_index.shape[1]}")
    
    if hasattr(data, 'edge_attr') and data.edge_attr is not None:
        edge_attr = data.edge_attr
        print(f"  - data.edge_attr shape: {tuple(edge_attr.shape)}")
        print(f"  - Edge attr dimension: {edge_attr.shape[1] if edge_attr.dim() > 1 else 1}")
        print(f"  - dtype: {edge_attr.dtype}")
    else:
        print(f"  - data.edge_attr: None (no edge features)")
    
    return data


def analyze_targets(dataset, name):
    """Analyze target information from the dataset."""
    print(f"\n--- Targets (data.y) ---")
    
    # Get a sample to check targets
    sample = dataset.get(0)
    
    if hasattr(sample, 'y') and sample.y is not None:
        y = sample.y
        print(f"  - data.y shape: {tuple(y.shape)}")
        print(f"  - Target dimension: {y.shape[-1] if y.dim() > 0 else 1}")
        print(f"  - dtype: {y.dtype}")
        
        # Task-specific info
        if dataset.task_type == "as":
            print(f"  - Task: Algorithm Selection (AS)")
            print(f"  - Target: Log10-transformed solver runtimes for {len(dataset.order)} solvers")
            print(f"  - Solvers: {dataset.order}")
        elif dataset.task_type == "epm":
            print(f"  - Task: Empirical Performance Modeling (EPM)")
            print(f"  - Target: Log10-transformed runtime for solver: {dataset.solver}")
            
        # Normalization info
        if dataset.normalize_targets:
            stats = dataset.get_normalization_stats()
            if stats:
                print(f"  - Normalization: enabled")
                print(f"  - Target mean: {stats['mean']:.4f}")
                print(f"  - Target std: {stats['std']:.4f}")
    else:
        print(f"  - data.y: None")


def print_feature_descriptions():
    """Print detailed feature descriptions for all graph types."""
    
    print("\n" + "="*80)
    print("FEATURE DESCRIPTIONS")
    print("="*80)
    
    # VCG Variable Features
    print("""
┌──────────────────────────────────────────────────────────────────────────────┐
│ VARIABLE-CLAUSE GRAPH (VCG) - HeteroData                                     │
├──────────────────────────────────────────────────────────────────────────────┤
│ Variable node features (data['var'].x) - 12 dimensions:                      │
│   0: constant (placeholder, set to 0)                                        │
│   1: constant (placeholder, set to 0)                                        │
│   2: horn_count (number of Horn clauses this variable appears in)            │
│   3: positive_count (appearances as positive literal)                        │
│   4: negative_count (appearances as negative literal)                        │
│   5: degree (total appearances)                                              │
│   6: pos_neg_ratio (positive / negative appearances)                         │
│   7: binary_clause_count (appearances in binary clauses)                     │
│   8: unit_clause_count (appearances in unit clauses)                         │
│   9: ternary_clause_count (appearances in ternary clauses)                   │
│  10: normalized_freq (degree / n_clauses)                                    │
│  11: avg_clause_length (average length of clauses containing this variable)  │
├──────────────────────────────────────────────────────────────────────────────┤
│ Clause node features (data['clause'].x) - 22 dimensions:                     │
│   0-9: sinusoidal positional encoding (10 dims)                              │
│  10: num_pos_literals                                                        │
│  11: num_neg_literals                                                        │
│  12: pos_neg_ratio                                                           │
│  13: clause_length                                                           │
│  14: is_unit (length == 1)                                                   │
│  15: is_binary (length == 2)                                                 │
│  16: is_ternary (length == 3)                                                │
│  17: is_horn (at most 1 positive literal)                                    │
│  18: normalized_length (clause_length / max_clause_length)                   │
│  19: clause_position_normalized (clause_idx / n_clauses)                     │
│  20: is_definite_horn (exactly 1 positive literal)                           │
│  21: is_negative_clause (all literals negative)                              │
├──────────────────────────────────────────────────────────────────────────────┤
│ Edge types:                                                                  │
│   - ('var', 'in', 'clause'): variable -> clause edges                        │
│   - ('clause', 'contains', 'var'): clause -> variable edges (reverse)        │
│ Edge features (edge_attr): 1D - polarity (+1 pos, -1 neg)                    │
└──────────────────────────────────────────────────────────────────────────────┘
""")

    # LCG Literal Features
    print("""
┌──────────────────────────────────────────────────────────────────────────────┐
│ LITERAL-CLAUSE GRAPH (LCG) - HeteroData                                      │
├──────────────────────────────────────────────────────────────────────────────┤
│ Literal node features (data['literal'].x) - 12 dimensions:                   │
│   0: is_positive (1 for positive literal, 0 for negative)                    │
│   1: is_negative (0 for positive literal, 1 for negative)                    │
│   2: horn_count (number of Horn clauses this literal appears in)             │
│   3: occurrence_count (total appearances in clauses)                         │
│   4: complementary_occurrence (appearances of the complementary literal)     │
│   5: degree (same as occurrence_count, for compatibility)                    │
│   6: pos_neg_ratio (occurrence / complementary_occurrence)                   │
│   7: binary_clause_count (appearances in binary clauses)                     │
│   8: unit_clause_count (appearances in unit clauses)                         │
│   9: ternary_clause_count (appearances in ternary clauses)                   │
│  10: normalized_freq (occurrence_count / n_clauses)                          │
│  11: avg_clause_length (average length of clauses containing this literal)   │
├──────────────────────────────────────────────────────────────────────────────┤
│ Clause node features (data['clause'].x) - 22 dimensions:                     │
│   Same as VCG clause features (see above)                                    │
├──────────────────────────────────────────────────────────────────────────────┤
│ Edge types:                                                                  │
│   - ('literal', 'in', 'clause'): literal -> clause edges                     │
│   - ('clause', 'contains', 'literal'): clause -> literal edges (reverse)     │
│   - ('literal', 'complement', 'literal'): complementary literal edges        │
│ Edge features (edge_attr): 1D - polarity (+1 pos, -1 neg)                    │
│ Note: There are 2*n_vars literal nodes (n_vars positive + n_vars negative)   │
└──────────────────────────────────────────────────────────────────────────────┘
""")

    # VG Variable Features
    print("""
┌──────────────────────────────────────────────────────────────────────────────┐
│ VARIABLE GRAPH (VG) - Data (homogeneous)                                     │
├──────────────────────────────────────────────────────────────────────────────┤
│ Node features (data.x) - 12 dimensions:                                      │
│   0: horn_count (number of Horn clauses this variable appears in)            │
│   1: positive_count (appearances as positive literal)                        │
│   2: negative_count (appearances as negative literal)                        │
│   3: degree (total co-occurrence count)                                      │
│   4: pos_neg_ratio (positive / negative appearances)                         │
│   5: binary_clause_count (appearances in binary clauses)                     │
│   6: unit_clause_count (appearances in unit clauses)                         │
│   7: ternary_clause_count (appearances in ternary clauses)                   │
│   8: normalized_freq (clause appearances / n_clauses)                        │
│   9: avg_clause_length (average length of clauses containing this variable)  │
│  10: total_clause_appearances (number of distinct clauses)                   │
│  11: avg_co_occurrence_degree (average co-occurrence per clause)             │
├──────────────────────────────────────────────────────────────────────────────┤
│ Edges: Connect variables that appear together in at least one clause         │
│ Edge features: None (data.edge_attr is not set)                              │
└──────────────────────────────────────────────────────────────────────────────┘
""")

    # CG Clause Features
    print("""
┌──────────────────────────────────────────────────────────────────────────────┐
│ CLAUSE GRAPH (CG) - Data (homogeneous)                                       │
├──────────────────────────────────────────────────────────────────────────────┤
│ Node features (data.x) - 22 dimensions:                                      │
│   0-9: sinusoidal positional encoding (10 dims)                              │
│  10: num_pos_literals                                                        │
│  11: num_neg_literals                                                        │
│  12: pos_neg_ratio                                                           │
│  13: clause_length                                                           │
│  14: is_unit (length == 1)                                                   │
│  15: is_binary (length == 2)                                                 │
│  16: is_ternary (length == 3)                                                │
│  17: is_horn (at most 1 positive literal)                                    │
│  18: normalized_length (clause_length / max_clause_length)                   │
│  19: clause_position_normalized (clause_idx / n_clauses)                     │
│  20: is_definite_horn (exactly 1 positive literal)                           │
│  21: is_negative_clause (all literals negative)                              │
├──────────────────────────────────────────────────────────────────────────────┤
│ Edges: Connect clauses that share a complementary literal                    │
│        (variable appears positive in one clause, negative in other)          │
│ Edge features: None (data.edge_attr is not set)                              │
└──────────────────────────────────────────────────────────────────────────────┘
""")

    # Target info
    print("""
┌──────────────────────────────────────────────────────────────────────────────┐
│ TARGET INFORMATION                                                           │
├──────────────────────────────────────────────────────────────────────────────┤
│ Algorithm Selection (AS) task - e.g., sat_vcg_as, sat_lcg_as:                │
│   - data.y shape: (1, num_solvers)                                           │
│   - Contains log10-transformed runtimes for each solver                      │
│   - Normalized if normalize_targets=True                                     │
│   - Use dataset.order to get solver names in order                           │
│   - Use dataset.denormalize_targets() to reverse normalization               │
│   - Use dataset.denormalize_to_seconds() to get original seconds             │
├──────────────────────────────────────────────────────────────────────────────┤
│ Empirical Performance Modeling (EPM) task - e.g., sat_vcg_epm, sat_lcg_epm:  │
│   - data.y shape: (1,)                                                       │
│   - Contains log10-transformed runtime for specific solver                   │
│   - Normalized if normalize_targets=True                                     │
│   - Use dataset.solver to get the solver name                                │
│   - Use dataset.denormalize_targets() to reverse normalization               │
│   - Use dataset.denormalize_to_seconds() to get original seconds             │
└──────────────────────────────────────────────────────────────────────────────┘
""")

    # Other attributes
    print("""
┌──────────────────────────────────────────────────────────────────────────────┐
│ OTHER IMPORTANT DATASET ATTRIBUTES                                           │
├──────────────────────────────────────────────────────────────────────────────┤
│ Dataset attributes:                                                          │
│   - dataset.task_type: 'as' or 'epm'                                         │
│   - dataset.graph_type: 'vcg', 'lcg', 'vg', or 'cg'                          │
│   - dataset.solver: Solver name (for EPM tasks)                              │
│   - dataset.order: List of solver names in target order (for AS tasks)       │
│   - dataset.normalize_targets: Whether targets are normalized                │
│   - dataset.target_mean: Mean used for normalization                         │
│   - dataset.target_std: Std used for normalization                           │
│   - dataset.use_satzilla_features: Whether satzilla meta-features are used   │
│                                                                              │
│ Data attributes (per graph):                                                 │
│   - data.num_nodes: Total number of nodes                                    │
│   - data.num_edges: Total number of edges                                    │
│   - data.y: Target tensor                                                    │
│                                                                              │
│ Utility methods:                                                             │
│   - dataset.get_normalization_stats(): Returns {'mean': ..., 'std': ...}     │
│   - dataset.denormalize_targets(y): Reverse normalization (log10 scale)      │
│   - dataset.denormalize_to_seconds(y): Convert to actual seconds             │
└──────────────────────────────────────────────────────────────────────────────┘
""")


def analyze_dataset_statistics(dataset, dataset_name):
    """Analyze statistics across the entire dataset.
    
    Computes:
    - Node features and dimensions
    - Edge features and dimensions
    - Target information and dimensions
    - Other important attributes
    - Min/max/average/median for nodes and edges
    """
    print(f"\n{'='*80}")
    print(f"DATASET-WIDE STATISTICS: {dataset_name}")
    print(f"{'='*80}")
    
    num_samples = len(dataset)
    print(f"\nTotal number of samples: {num_samples}")
    
    # Collect statistics across all graphs
    node_counts = []
    edge_counts = []
    
    # Get first sample to understand structure
    sample = dataset[0]
    is_hetero = hasattr(sample, 'node_types')
    
    print(f"\n--- Feature Summary ---")
    
    if is_hetero:
        print(f"Graph type: HeteroData")
        print(f"\nNode types and features:")
        for node_type in sample.node_types:
            node_store = sample[node_type]
            if hasattr(node_store, 'x') and node_store.x is not None:
                x = node_store.x
                print(f"  '{node_type}' nodes:")
                print(f"    - Feature dimension (data['{node_type}'].x): {x.shape[1]}")
                print(f"    - dtype: {x.dtype}")
        
        print(f"\nEdge types and features:")
        for edge_type in sample.edge_types:
            edge_store = sample[edge_type]
            src, rel, dst = edge_type
            print(f"  Edge type ('{src}', '{rel}', '{dst}'):")
            if hasattr(edge_store, 'edge_attr') and edge_store.edge_attr is not None:
                edge_attr = edge_store.edge_attr
                dim = edge_attr.shape[1] if edge_attr.dim() > 1 else 1
                print(f"    - Edge attr dimension: {dim}")
                print(f"    - dtype: {edge_attr.dtype}")
            else:
                print(f"    - Edge attr: None (no edge features)")
    else:
        print(f"Graph type: Data (homogeneous)")
        print(f"\nNode features:")
        if hasattr(sample, 'x') and sample.x is not None:
            x = sample.x
            print(f"  - Feature dimension (data.x): {x.shape[1]}")
            print(f"  - dtype: {x.dtype}")
        else:
            print(f"  - data.x: None")
        
        print(f"\nEdge features:")
        if hasattr(sample, 'edge_attr') and sample.edge_attr is not None:
            edge_attr = sample.edge_attr
            dim = edge_attr.shape[1] if edge_attr.dim() > 1 else 1
            print(f"  - Edge attr dimension: {dim}")
            print(f"  - dtype: {edge_attr.dtype}")
        else:
            print(f"  - data.edge_attr: None (no edge features)")
    
    # Target information
    print(f"\n--- Target Summary ---")
    if hasattr(sample, 'y') and sample.y is not None:
        y = sample.y
        target_dim = y.shape[-1] if y.dim() > 0 and len(y.shape) > 1 else 1
        print(f"  - Target dimension (data.y): {target_dim}")
        print(f"  - Target shape: {tuple(y.shape)}")
        print(f"  - dtype: {y.dtype}")
        
        if dataset.task_type == "as":
            print(f"  - Task: Algorithm Selection (AS)")
            print(f"  - Number of solvers: {len(dataset.order)}")
            print(f"  - Solvers: {dataset.order}")
        elif dataset.task_type == "epm":
            print(f"  - Task: Empirical Performance Modeling (EPM)")
            print(f"  - Solver: {dataset.solver}")
    else:
        print(f"  - data.y: None")
    
    # Other important attributes
    print(f"\n--- Other Important Attributes ---")
    print(f"  - dataset.task_type: '{dataset.task_type}'")
    print(f"  - dataset.graph_type: '{dataset.graph_type}'")
    print(f"  - dataset.normalize_targets: {dataset.normalize_targets}")
    if hasattr(dataset, 'use_satzilla_features'):
        print(f"  - dataset.use_satzilla_features: {dataset.use_satzilla_features}")
    if dataset.normalize_targets:
        stats = dataset.get_normalization_stats()
        if stats:
            print(f"  - Normalization mean: {stats['mean']:.4f}")
            print(f"  - Normalization std: {stats['std']:.4f}")
    
    # Collect node and edge counts across all samples
    print(f"\n--- Computing Node/Edge Statistics Across Dataset ---")
    print(f"Iterating over {num_samples} samples...")
    
    for i in range(num_samples):
        data = dataset[i]
        
        # Get node count
        if hasattr(data, 'num_nodes') and data.num_nodes is not None:
            num_nodes = int(data.num_nodes) if hasattr(data.num_nodes, 'item') else data.num_nodes
        elif is_hetero:
            num_nodes = sum(data[nt].x.shape[0] for nt in data.node_types 
                          if hasattr(data[nt], 'x') and data[nt].x is not None)
        elif hasattr(data, 'x') and data.x is not None:
            num_nodes = data.x.shape[0]
        else:
            num_nodes = 0
        
        # Get edge count
        if hasattr(data, 'num_edges') and data.num_edges is not None:
            num_edges = int(data.num_edges) if hasattr(data.num_edges, 'item') else data.num_edges
        elif is_hetero:
            num_edges = sum(data[et].edge_index.shape[1] for et in data.edge_types 
                          if hasattr(data[et], 'edge_index') and data[et].edge_index is not None)
        elif hasattr(data, 'edge_index') and data.edge_index is not None:
            num_edges = data.edge_index.shape[1]
        else:
            num_edges = 0
        
        node_counts.append(num_nodes)
        edge_counts.append(num_edges)
    
    node_counts = np.array(node_counts)
    edge_counts = np.array(edge_counts)
    
    # Compute and print statistics
    print(f"\n--- Node Statistics ---")
    print(f"  - Minimum nodes: {np.min(node_counts)}")
    print(f"  - Maximum nodes: {np.max(node_counts)}")
    print(f"  - Average nodes: {np.mean(node_counts):.2f}")
    print(f"  - Median nodes: {np.median(node_counts):.2f}")
    print(f"  - Std nodes: {np.std(node_counts):.2f}")
    
    print(f"\n--- Edge Statistics ---")
    print(f"  - Minimum edges: {np.min(edge_counts)}")
    print(f"  - Maximum edges: {np.max(edge_counts)}")
    print(f"  - Average edges: {np.mean(edge_counts):.2f}")
    print(f"  - Median edges: {np.median(edge_counts):.2f}")
    print(f"  - Std edges: {np.std(edge_counts):.2f}")
    
    return {
        'num_samples': num_samples,
        'node_counts': node_counts,
        'edge_counts': edge_counts,
        'is_hetero': is_hetero
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze SAT dataset information")
    parser.add_argument("--root", type=str, default="/storage/work/graph_bench/dataset",
                        help="Root directory for datasets")
    parser.add_argument("--name", type=str, 
                        default="sat_lcg_as", 
                        help="Dataset name to analyze (e.g., small_vcg, small_lcg, small_vg, small_cg)")
    parser.add_argument("--descriptions-only", action="store_true",
                        help="Only print feature descriptions without loading datasets")
    parser.add_argument("--skip-descriptions", action="store_true",
                        help="Skip printing feature descriptions")
    args = parser.parse_args()
    root = args.root + f"_{args.name}"
    
    # Print feature descriptions unless skipped
    if not args.skip_descriptions:
        print_feature_descriptions()
    
    if args.descriptions_only:
        print("\n[Note: --descriptions-only flag set, skipping dataset loading]")
        return
    
    for dataset_name in [args.name]:
        
        try:
            print(f"\nLoading {dataset_name}...")
            print(f"Root: {root}")
            dataset = SATDataset(
                name=dataset_name,
                split="train",
                root=root,
                generate=False
            )
            
            # Compute dataset-wide statistics
            analyze_dataset_statistics(dataset, dataset_name)
            
            # Also show single sample analysis for detailed structure info
            print(f"\n{'='*80}")
            print(f"SINGLE SAMPLE ANALYSIS (graph 0)")
            print(f"{'='*80}")
            
            sample_graph = dataset[0]
            
            # Analyze based on graph type
            if dataset.graph_type in ["vcg", "lcg"]:
                analyze_hetero_data(sample_graph, dataset_name)
            else:
                analyze_homo_data(sample_graph, dataset_name)
            
            analyze_targets(dataset, dataset_name)
                
        except Exception as e:
            print(f"\n[Error loading {dataset_name}: {e}]")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
