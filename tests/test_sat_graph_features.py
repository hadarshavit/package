"""
Tests for SAT graph creation methods with GraSS features.

This module verifies that the graph features are calculated correctly
and the graph structures are valid for all three graph types:
- Literal-Clause Graph (LCG)
- Variable-Clause Graph (VCG)
- Variable Graph (VG)
"""

import numpy as np
import pytest
import torch

# Import the module under test
import sys
sys.path.insert(0, '/home/shavit/graphbench_package')

from graphbench.datasets.sat import (
    SATDataset,
    _sinusoidal_positional_encoding,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def simple_cnf():
    """
    Simple CNF formula: (x1 ∨ ¬x2 ∨ x3) ∧ (¬x1 ∨ x2) ∧ (x3)
    
    Clauses: [[1, -2, 3], [-1, 2], [3]]
    - Clause 0: 3 literals, 2 positive, 1 negative, NOT Horn
    - Clause 1: 2 literals, 1 positive, 1 negative, Horn (binary)
    - Clause 2: 1 literal, 1 positive, 0 negative, Horn (unit)
    """
    return {
        'clauses': [[1, -2, 3], [-1, 2], [3]],
        'n_vars': 3,
        'n_clauses': 3,
    }


@pytest.fixture
def horn_cnf():
    """
    Pure Horn formula: (¬x1 ∨ ¬x2 ∨ x3) ∧ (¬x1) ∧ (x2)
    
    All clauses have at most 1 positive literal.
    """
    return {
        'clauses': [[-1, -2, 3], [-1], [2]],
        'n_vars': 3,
        'n_clauses': 3,
    }


@pytest.fixture
def dataset_stub():
    """Create a minimal dataset stub for testing graph creation methods."""
    # Create a stub instance without calling __init__
    ds = object.__new__(SATDataset)
    return ds


# =============================================================================
# Positional Encoding Tests
# =============================================================================

class TestPositionalEncoding:
    """Tests for the sinusoidal positional encoding function."""
    
    def test_output_shape(self):
        """Test that output has correct shape."""
        positions = np.arange(10)
        pe = _sinusoidal_positional_encoding(positions, dim=10)
        assert pe.shape == (10, 10), f"Expected (10, 10), got {pe.shape}"
    
    def test_different_dimensions(self):
        """Test with different embedding dimensions."""
        positions = np.arange(5)
        for dim in [4, 8, 10, 16]:
            pe = _sinusoidal_positional_encoding(positions, dim=dim)
            assert pe.shape == (5, dim), f"Expected (5, {dim}), got {pe.shape}"
    
    def test_values_bounded(self):
        """Test that all values are in [-1, 1] range."""
        positions = np.arange(100)
        pe = _sinusoidal_positional_encoding(positions, dim=10)
        assert np.all(pe >= -1) and np.all(pe <= 1), "Values should be in [-1, 1]"
    
    def test_first_position_pattern(self):
        """Test that position 0 has expected sin pattern (sin(0) = 0)."""
        pe = _sinusoidal_positional_encoding(np.array([0]), dim=10)
        # sin(0) = 0 for all frequencies
        assert pe[0, 0] == pytest.approx(0, abs=1e-6), "sin(0) should be 0"
        # Second position is scaled, so just verify it's not zero
        assert pe[0, 1] != 0, "cos term should not be zero"
    
    def test_unique_positions(self):
        """Test that different positions produce different encodings."""
        positions = np.arange(5)
        pe = _sinusoidal_positional_encoding(positions, dim=10)
        # Each row should be unique
        for i in range(5):
            for j in range(i + 1, 5):
                assert not np.allclose(pe[i], pe[j]), f"Positions {i} and {j} should differ"


# =============================================================================
# Literal-Clause Graph Tests
# =============================================================================

class TestLiteralClauseGraph:
    """Tests for create_literal_clause_graph method."""
    
    def test_node_counts(self, dataset_stub, simple_cnf):
        """Test correct number of nodes."""
        data = dataset_stub.create_literal_clause_graph(
            simple_cnf['clauses'], simple_cnf['n_vars']
        )
        n_vars = simple_cnf['n_vars']
        n_clauses = simple_cnf['n_clauses']
        
        # 2 * n_vars literals + n_clauses
        expected_literal_nodes = 2 * n_vars
        expected_clause_nodes = n_clauses
        
        assert data["literal"].x.shape[0] == expected_literal_nodes, \
            f"Expected {expected_literal_nodes} literal nodes"
        assert data["clause"].x.shape[0] == expected_clause_nodes, \
            f"Expected {expected_clause_nodes} clause nodes"
    
    def test_feature_dimensions(self, dataset_stub, simple_cnf):
        """Test correct feature dimensions."""
        data = dataset_stub.create_literal_clause_graph(
            simple_cnf['clauses'], simple_cnf['n_vars']
        )
        
        assert data["literal"].x.shape[1] == 12, \
            f"Literal features should be 12-dim, got {data['literal'].x.shape[1]}"
        assert data["clause"].x.shape[1] == 22, \
            f"Clause features should be 22-dim, got {data['clause'].x.shape[1]}"
    
    def test_positive_negative_indicators(self, dataset_stub, simple_cnf):
        """Test that positive/negative literal indicators are correct."""
        data = dataset_stub.create_literal_clause_graph(
            simple_cnf['clauses'], simple_cnf['n_vars']
        )
        n_vars = simple_cnf['n_vars']
        
        # First n_vars are positive literals (dim 0 = 1, dim 1 = 0)
        for i in range(n_vars):
            assert data["literal"].x[i, 0] == 1.0, f"Literal {i} should be positive"
            assert data["literal"].x[i, 1] == 0.0, f"Literal {i} should not be negative"
        
        # Next n_vars are negative literals (dim 0 = 0, dim 1 = 1)
        for i in range(n_vars, 2 * n_vars):
            assert data["literal"].x[i, 0] == 0.0, f"Literal {i} should not be positive"
            assert data["literal"].x[i, 1] == 1.0, f"Literal {i} should be negative"
    
    def test_complement_edges(self, dataset_stub, simple_cnf):
        """Test that complement edges connect pos/neg literals of same variable."""
        data = dataset_stub.create_literal_clause_graph(
            simple_cnf['clauses'], simple_cnf['n_vars']
        )
        n_vars = simple_cnf['n_vars']
        
        assert ("literal", "complement", "literal") in data.edge_types, \
            "Should have complement edge type"
        
        comp_edges = data["literal", "complement", "literal"].edge_index
        assert comp_edges.shape[1] == 2 * n_vars, \
            f"Should have {2 * n_vars} complement edges (bidirectional)"
    
    def test_clause_positional_encoding(self, dataset_stub, simple_cnf):
        """Test that clauses have positional encoding in first 10 dims."""
        data = dataset_stub.create_literal_clause_graph(
            simple_cnf['clauses'], simple_cnf['n_vars']
        )
        
        # First 10 dims should be positional encoding (not all zeros)
        pos_enc = data["clause"].x[:, :10]
        assert not torch.allclose(pos_enc, torch.zeros_like(pos_enc)), \
            "Positional encoding should not be all zeros"
        
        # Values should be in [-1, 1]
        assert pos_enc.min() >= -1 and pos_enc.max() <= 1, \
            "Positional encoding values should be in [-1, 1]"
    
    def test_horn_clause_detection(self, dataset_stub, horn_cnf):
        """Test that Horn clauses are correctly identified."""
        data = dataset_stub.create_literal_clause_graph(
            horn_cnf['clauses'], horn_cnf['n_vars']
        )
        
        # All clauses in horn_cnf are Horn clauses (dim 17 should be 1)
        for i in range(horn_cnf['n_clauses']):
            assert data["clause"].x[i, 17] == 1.0, \
                f"Clause {i} should be marked as Horn"
    
    def test_unit_binary_ternary_detection(self, dataset_stub, simple_cnf):
        """Test detection of unit, binary, and ternary clauses."""
        data = dataset_stub.create_literal_clause_graph(
            simple_cnf['clauses'], simple_cnf['n_vars']
        )
        
        # Clause 0: ternary (3 literals)
        assert data["clause"].x[0, 16] == 1.0, "Clause 0 should be ternary"
        assert data["clause"].x[0, 14] == 0.0, "Clause 0 should not be unit"
        assert data["clause"].x[0, 15] == 0.0, "Clause 0 should not be binary"
        
        # Clause 1: binary (2 literals)
        assert data["clause"].x[1, 15] == 1.0, "Clause 1 should be binary"
        assert data["clause"].x[1, 14] == 0.0, "Clause 1 should not be unit"
        assert data["clause"].x[1, 16] == 0.0, "Clause 1 should not be ternary"
        
        # Clause 2: unit (1 literal)
        assert data["clause"].x[2, 14] == 1.0, "Clause 2 should be unit"
        assert data["clause"].x[2, 15] == 0.0, "Clause 2 should not be binary"
        assert data["clause"].x[2, 16] == 0.0, "Clause 2 should not be ternary"
    
    def test_edge_connectivity(self, dataset_stub, simple_cnf):
        """Test that edges correctly connect literals to clauses."""
        data = dataset_stub.create_literal_clause_graph(
            simple_cnf['clauses'], simple_cnf['n_vars']
        )
        
        # Check edge types exist
        assert ("literal", "in", "clause") in data.edge_types
        assert ("clause", "contains", "literal") in data.edge_types
        
        edges = data["literal", "in", "clause"].edge_index
        
        # Count total edges: clause 0 has 3, clause 1 has 2, clause 2 has 1 = 6 total
        assert edges.shape[1] == 6, f"Expected 6 edges, got {edges.shape[1]}"


# =============================================================================
# Variable-Clause Graph Tests
# =============================================================================

class TestVariableClauseGraph:
    """Tests for create_variable_clause_graph method."""
    
    def test_node_counts(self, dataset_stub, simple_cnf):
        """Test correct number of nodes."""
        data = dataset_stub.create_variable_clause_graph(
            simple_cnf['clauses'], simple_cnf['n_vars']
        )
        
        assert data["var"].x.shape[0] == simple_cnf['n_vars'], \
            f"Expected {simple_cnf['n_vars']} variable nodes"
        assert data["clause"].x.shape[0] == simple_cnf['n_clauses'], \
            f"Expected {simple_cnf['n_clauses']} clause nodes"
    
    def test_feature_dimensions(self, dataset_stub, simple_cnf):
        """Test correct feature dimensions."""
        data = dataset_stub.create_variable_clause_graph(
            simple_cnf['clauses'], simple_cnf['n_vars']
        )
        
        assert data["var"].x.shape[1] == 12, \
            f"Variable features should be 12-dim, got {data['var'].x.shape[1]}"
        assert data["clause"].x.shape[1] == 22, \
            f"Clause features should be 22-dim, got {data['clause'].x.shape[1]}"
    
    def test_positive_negative_counts(self, dataset_stub, simple_cnf):
        """Test that positive/negative appearance counts are correct."""
        data = dataset_stub.create_variable_clause_graph(
            simple_cnf['clauses'], simple_cnf['n_vars']
        )
        
        # x1 appears: positive in clause 0, negative in clause 1
        # dim 3 = positive_count, dim 4 = negative_count
        assert data["var"].x[0, 3] == 1.0, "x1 positive count should be 1"
        assert data["var"].x[0, 4] == 1.0, "x1 negative count should be 1"
        
        # x2 appears: negative in clause 0, positive in clause 1
        assert data["var"].x[1, 3] == 1.0, "x2 positive count should be 1"
        assert data["var"].x[1, 4] == 1.0, "x2 negative count should be 1"
        
        # x3 appears: positive in clause 0, positive in clause 2
        assert data["var"].x[2, 3] == 2.0, "x3 positive count should be 2"
        assert data["var"].x[2, 4] == 0.0, "x3 negative count should be 0"
    
    def test_clause_positional_encoding(self, dataset_stub, simple_cnf):
        """Test that clauses have positional encoding."""
        data = dataset_stub.create_variable_clause_graph(
            simple_cnf['clauses'], simple_cnf['n_vars']
        )
        
        pos_enc = data["clause"].x[:, :10]
        assert not torch.allclose(pos_enc, torch.zeros_like(pos_enc)), \
            "Positional encoding should not be all zeros"
    
    def test_edge_attr_sign(self, dataset_stub, simple_cnf):
        """Test that edge attributes correctly indicate literal sign."""
        data = dataset_stub.create_variable_clause_graph(
            simple_cnf['clauses'], simple_cnf['n_vars']
        )
        
        edge_attr = data["var", "in", "clause"].edge_attr
        
        # All edge attributes should be +1 or -1
        assert torch.all((edge_attr == 1) | (edge_attr == -1)), \
            "Edge attributes should be +1 or -1"


# =============================================================================
# Variable Graph Tests
# =============================================================================

class TestVariableGraph:
    """Tests for create_variable_graph method."""
    
    def test_node_count(self, dataset_stub, simple_cnf):
        """Test correct number of nodes."""
        data = dataset_stub.create_variable_graph(
            simple_cnf['clauses'], simple_cnf['n_vars']
        )
        
        assert data.x.shape[0] == simple_cnf['n_vars'], \
            f"Expected {simple_cnf['n_vars']} nodes"
    
    def test_feature_dimensions(self, dataset_stub, simple_cnf):
        """Test correct feature dimensions."""
        data = dataset_stub.create_variable_graph(
            simple_cnf['clauses'], simple_cnf['n_vars']
        )
        
        assert data.x.shape[1] == 12, \
            f"Variable features should be 12-dim, got {data.x.shape[1]}"
    
    def test_edge_structure(self, dataset_stub, simple_cnf):
        """Test that edges connect co-occurring variables."""
        data = dataset_stub.create_variable_graph(
            simple_cnf['clauses'], simple_cnf['n_vars']
        )
        
        # Clause 0 has vars {0, 1, 2} -> edges (0,1), (0,2), (1,2)
        # Clause 1 has vars {0, 1} -> edge (0,1) already exists
        # Clause 2 has var {2} -> no new edges
        # Total unique edges: 3
        
        assert data.edge_index.shape[1] == 3, \
            f"Expected 3 edges, got {data.edge_index.shape[1]}"
    
    def test_normalized_frequency(self, dataset_stub, simple_cnf):
        """Test that normalized frequency is computed correctly."""
        data = dataset_stub.create_variable_graph(
            simple_cnf['clauses'], simple_cnf['n_vars']
        )
        
        # dim 8 = normalized_freq = clause_appearances / n_clauses
        # x1 appears in 2 clauses, x2 in 2, x3 in 2
        # normalized_freq = 2/3 ≈ 0.667
        for i in range(3):
            assert data.x[i, 8] == pytest.approx(2/3, abs=0.01), \
                f"Variable {i} normalized_freq should be ~0.667"
    
    def test_avg_clause_length(self, dataset_stub, simple_cnf):
        """Test that average clause length is computed correctly."""
        data = dataset_stub.create_variable_graph(
            simple_cnf['clauses'], simple_cnf['n_vars']
        )
        
        # dim 9 = avg_clause_length
        # x1 in clauses 0 (len 3) and 1 (len 2) -> avg = 2.5
        # x2 in clauses 0 (len 3) and 1 (len 2) -> avg = 2.5
        # x3 in clauses 0 (len 3) and 2 (len 1) -> avg = 2.0
        
        assert data.x[0, 9] == pytest.approx(2.5, abs=0.01), "x1 avg clause length"
        assert data.x[1, 9] == pytest.approx(2.5, abs=0.01), "x2 avg clause length"
        assert data.x[2, 9] == pytest.approx(2.0, abs=0.01), "x3 avg clause length"


# =============================================================================
# Empty/Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_single_clause(self, dataset_stub):
        """Test with a single clause."""
        clauses = [[1, 2, 3]]
        n_vars = 3
        
        data = dataset_stub.create_literal_clause_graph(clauses, n_vars)
        assert data["clause"].x.shape[0] == 1
        assert data["literal"].x.shape[0] == 6
    
    def test_single_variable(self, dataset_stub):
        """Test with a single variable."""
        clauses = [[1], [-1]]
        n_vars = 1
        
        data = dataset_stub.create_literal_clause_graph(clauses, n_vars)
        assert data["literal"].x.shape[0] == 2  # pos and neg
        assert data["clause"].x.shape[0] == 2
    
    def test_all_unit_clauses(self, dataset_stub):
        """Test with all unit clauses."""
        clauses = [[1], [-2], [3]]
        n_vars = 3
        
        data = dataset_stub.create_variable_clause_graph(clauses, n_vars)
        
        # All clauses should be marked as unit (dim 14)
        for i in range(3):
            assert data["clause"].x[i, 14] == 1.0, f"Clause {i} should be unit"
    
    def test_variable_graph_no_cooccurrence(self, dataset_stub):
        """Test VG with no co-occurring variables (all unit clauses)."""
        clauses = [[1], [2], [3]]
        n_vars = 3
        
        data = dataset_stub.create_variable_graph(clauses, n_vars)
        
        # No edges since no variables co-occur
        assert data.edge_index.shape[1] == 0, "Should have no edges"


# =============================================================================
# Comprehensive Structural Tests
# =============================================================================

class TestLCGStructureComprehensive:
    """Comprehensive structural tests for Literal-Clause Graph."""
    
    @pytest.fixture
    def known_cnf(self):
        """
        Well-defined CNF: (x1 ∨ ¬x2) ∧ (¬x1 ∨ x3) ∧ (x2)
        
        Clause 0: [1, -2] -> literals: pos_x1 (0), neg_x2 (4)
        Clause 1: [-1, 3] -> literals: neg_x1 (3), pos_x3 (2)
        Clause 2: [2]     -> literals: pos_x2 (1)
        
        Literal indices (n_vars=3):
            pos_x1=0, pos_x2=1, pos_x3=2, neg_x1=3, neg_x2=4, neg_x3=5
        """
        return {
            'clauses': [[1, -2], [-1, 3], [2]],
            'n_vars': 3,
            'n_clauses': 3,
        }
    
    @pytest.fixture
    def dataset_stub(self):
        return object.__new__(SATDataset)
    
    def test_exact_literal_to_clause_edges(self, dataset_stub, known_cnf):
        """Test exact edge connectivity from literals to clauses."""
        data = dataset_stub.create_literal_clause_graph(
            known_cnf['clauses'], known_cnf['n_vars']
        )
        
        edges = data["literal", "in", "clause"].edge_index
        edge_set = set(zip(edges[0].tolist(), edges[1].tolist()))
        
        # Expected edges: (literal_id, clause_id)
        # Clause 0 [1, -2]: pos_x1(0)->c0, neg_x2(4)->c0
        # Clause 1 [-1, 3]: neg_x1(3)->c1, pos_x3(2)->c1
        # Clause 2 [2]: pos_x2(1)->c2
        expected = {(0, 0), (4, 0), (3, 1), (2, 1), (1, 2)}
        
        assert edge_set == expected, f"Expected {expected}, got {edge_set}"
    
    def test_exact_complement_edges(self, dataset_stub, known_cnf):
        """Test exact complement edge connectivity."""
        data = dataset_stub.create_literal_clause_graph(
            known_cnf['clauses'], known_cnf['n_vars']
        )
        n_vars = known_cnf['n_vars']
        
        comp_edges = data["literal", "complement", "literal"].edge_index
        edge_set = set(zip(comp_edges[0].tolist(), comp_edges[1].tolist()))
        
        # Expected: pos<->neg for each variable (bidirectional)
        # pos_x1(0) <-> neg_x1(3)
        # pos_x2(1) <-> neg_x2(4)
        # pos_x3(2) <-> neg_x3(5)
        expected = {
            (0, 3), (3, 0),  # x1
            (1, 4), (4, 1),  # x2
            (2, 5), (5, 2),  # x3
        }
        
        assert edge_set == expected, f"Expected {expected}, got {edge_set}"
    
    def test_reverse_edges_match(self, dataset_stub, known_cnf):
        """Test that clause->literal edges are reverse of literal->clause."""
        data = dataset_stub.create_literal_clause_graph(
            known_cnf['clauses'], known_cnf['n_vars']
        )
        
        lit_to_clause = data["literal", "in", "clause"].edge_index
        clause_to_lit = data["clause", "contains", "literal"].edge_index
        
        # clause_to_lit should be lit_to_clause with rows flipped
        assert torch.equal(clause_to_lit[0], lit_to_clause[1])
        assert torch.equal(clause_to_lit[1], lit_to_clause[0])
    
    def test_edge_attrs_match_sign(self, dataset_stub, known_cnf):
        """Test that edge attributes correctly indicate literal polarity."""
        data = dataset_stub.create_literal_clause_graph(
            known_cnf['clauses'], known_cnf['n_vars']
        )
        n_vars = known_cnf['n_vars']
        
        edges = data["literal", "in", "clause"].edge_index
        edge_attr = data["literal", "in", "clause"].edge_attr
        
        for i in range(edges.shape[1]):
            lit_id = edges[0, i].item()
            attr = edge_attr[i].item()
            
            # Positive literals (0 to n_vars-1) should have attr=1
            # Negative literals (n_vars to 2*n_vars-1) should have attr=-1
            if lit_id < n_vars:
                assert attr == 1.0, f"Positive literal {lit_id} should have attr=1"
            else:
                assert attr == -1.0, f"Negative literal {lit_id} should have attr=-1"
    
    def test_all_literal_features_complete(self, dataset_stub, known_cnf):
        """Test that all 12 literal feature dimensions are populated."""
        data = dataset_stub.create_literal_clause_graph(
            known_cnf['clauses'], known_cnf['n_vars']
        )
        
        lit_features = data["literal"].x
        
        # Check each feature dimension exists and has reasonable values
        # Dim 0-1: pos/neg indicators (checked elsewhere)
        # Dim 2: horn_count >= 0
        assert (lit_features[:, 2] >= 0).all(), "horn_count should be >= 0"
        # Dim 3: occurrence_count >= 0
        assert (lit_features[:, 3] >= 0).all(), "occurrence_count should be >= 0"
        # Dim 5: degree >= 0
        assert (lit_features[:, 5] >= 0).all(), "degree should be >= 0"
        # Dim 10: normalized_freq in [0, 1]
        assert (lit_features[:, 10] >= 0).all() and (lit_features[:, 10] <= 1).all(), \
            "normalized_freq should be in [0, 1]"
    
    def test_all_clause_features_complete(self, dataset_stub, known_cnf):
        """Test that all 22 clause feature dimensions are populated."""
        data = dataset_stub.create_literal_clause_graph(
            known_cnf['clauses'], known_cnf['n_vars']
        )
        
        clause_features = data["clause"].x
        
        # Dim 0-9: positional encoding in [-1, 1]
        pos_enc = clause_features[:, :10]
        assert (pos_enc >= -1).all() and (pos_enc <= 1).all()
        
        # Dim 10-11: num_pos/num_neg >= 0
        assert (clause_features[:, 10] >= 0).all()
        assert (clause_features[:, 11] >= 0).all()
        
        # Dim 13: clause_length >= 1
        assert (clause_features[:, 13] >= 1).all()
        
        # Dim 14-16: mutually exclusive unit/binary/ternary
        for i in range(known_cnf['n_clauses']):
            indicators = clause_features[i, 14:17].tolist()
            assert sum(indicators) <= 1, "At most one of unit/binary/ternary"
        
        # Dim 18: normalized_length in (0, 1]
        assert (clause_features[:, 18] > 0).all() and (clause_features[:, 18] <= 1).all()
        
        # Dim 19: clause_position_normalized in [0, 1]
        assert (clause_features[:, 19] >= 0).all() and (clause_features[:, 19] <= 1).all()


class TestVCGStructureComprehensive:
    """Comprehensive structural tests for Variable-Clause Graph."""
    
    @pytest.fixture
    def known_cnf(self):
        """Same as LCG test fixture."""
        return {
            'clauses': [[1, -2], [-1, 3], [2]],
            'n_vars': 3,
            'n_clauses': 3,
        }
    
    @pytest.fixture
    def dataset_stub(self):
        return object.__new__(SATDataset)
    
    def test_exact_variable_to_clause_edges(self, dataset_stub, known_cnf):
        """Test exact edge connectivity from variables to clauses."""
        data = dataset_stub.create_variable_clause_graph(
            known_cnf['clauses'], known_cnf['n_vars']
        )
        
        edges = data["var", "in", "clause"].edge_index
        edge_set = set(zip(edges[0].tolist(), edges[1].tolist()))
        
        # Expected edges: (var_id, clause_id)
        # Clause 0 [1, -2]: x1(0)->c0, x2(1)->c0
        # Clause 1 [-1, 3]: x1(0)->c1, x3(2)->c1
        # Clause 2 [2]: x2(1)->c2
        expected = {(0, 0), (1, 0), (0, 1), (2, 1), (1, 2)}
        
        assert edge_set == expected, f"Expected {expected}, got {edge_set}"
    
    def test_edge_attrs_match_polarity(self, dataset_stub, known_cnf):
        """Test edge attrs match literal polarity in original clause."""
        data = dataset_stub.create_variable_clause_graph(
            known_cnf['clauses'], known_cnf['n_vars']
        )
        
        edges = data["var", "in", "clause"].edge_index
        edge_attr = data["var", "in", "clause"].edge_attr
        
        # Build expected polarity map: (var, clause) -> polarity
        # Clause 0: x1=+1, x2=-1
        # Clause 1: x1=-1, x3=+1
        # Clause 2: x2=+1
        expected_polarity = {
            (0, 0): 1.0,   # x1 positive in clause 0
            (1, 0): -1.0,  # x2 negative in clause 0
            (0, 1): -1.0,  # x1 negative in clause 1
            (2, 1): 1.0,   # x3 positive in clause 1
            (1, 2): 1.0,   # x2 positive in clause 2
        }
        
        for i in range(edges.shape[1]):
            var_id = edges[0, i].item()
            clause_id = edges[1, i].item()
            actual_attr = edge_attr[i].item()
            expected_attr = expected_polarity[(var_id, clause_id)]
            assert actual_attr == expected_attr, \
                f"Edge ({var_id}, {clause_id}) expected {expected_attr}, got {actual_attr}"
    
    def test_reverse_edges_match(self, dataset_stub, known_cnf):
        """Test that clause->var edges are reverse of var->clause."""
        data = dataset_stub.create_variable_clause_graph(
            known_cnf['clauses'], known_cnf['n_vars']
        )
        
        var_to_clause = data["var", "in", "clause"].edge_index
        clause_to_var = data["clause", "contains", "var"].edge_index
        
        assert torch.equal(clause_to_var[0], var_to_clause[1])
        assert torch.equal(clause_to_var[1], var_to_clause[0])
    
    def test_variable_degree_matches_edges(self, dataset_stub, known_cnf):
        """Test that variable degree feature matches actual edge count."""
        data = dataset_stub.create_variable_clause_graph(
            known_cnf['clauses'], known_cnf['n_vars']
        )
        
        edges = data["var", "in", "clause"].edge_index
        var_features = data["var"].x
        
        # Count actual edges per variable
        for var_id in range(known_cnf['n_vars']):
            edge_count = (edges[0] == var_id).sum().item()
            degree_feature = var_features[var_id, 5].item()  # dim 5 = degree
            assert edge_count == degree_feature, \
                f"Var {var_id}: edge_count={edge_count}, degree_feature={degree_feature}"


class TestVGStructureComprehensive:
    """Comprehensive structural tests for Variable Graph."""
    
    @pytest.fixture
    def known_cnf(self):
        """
        CNF: (x1 ∨ x2 ∨ x3) ∧ (x1 ∨ x4) ∧ (x2 ∨ x3)
        
        Co-occurrences:
        - Clause 0: (1,2), (1,3), (2,3)
        - Clause 1: (1,4)  
        - Clause 2: (2,3) [duplicate edge]
        
        Unique edges: (0,1), (0,2), (0,3), (1,2), (1,3)
        Note: (1,2) from clause 2 already exists from clause 0
        """
        return {
            'clauses': [[1, 2, 3], [1, 4], [2, 3]],
            'n_vars': 4,
            'n_clauses': 3,
        }
    
    @pytest.fixture
    def dataset_stub(self):
        return object.__new__(SATDataset)
    
    def test_exact_cooccurrence_edges(self, dataset_stub, known_cnf):
        """Test exact co-occurrence edge set."""
        data = dataset_stub.create_variable_graph(
            known_cnf['clauses'], known_cnf['n_vars']
        )
        
        edges = data.edge_index
        edge_set = set()
        for i in range(edges.shape[1]):
            a, b = edges[0, i].item(), edges[1, i].item()
            edge_set.add((min(a, b), max(a, b)))
        
        # Expected unique edges (0-indexed, ordered)
        # Clause 0 [1,2,3]: (0,1), (0,2), (1,2)
        # Clause 1 [1,4]: (0,3)
        # Clause 2 [2,3]: (1,2) already exists
        expected = {(0, 1), (0, 2), (0, 3), (1, 2)}
        
        assert edge_set == expected, f"Expected {expected}, got {edge_set}"
    
    def test_no_self_loops(self, dataset_stub, known_cnf):
        """Test that there are no self-loops."""
        data = dataset_stub.create_variable_graph(
            known_cnf['clauses'], known_cnf['n_vars']
        )
        
        edges = data.edge_index
        for i in range(edges.shape[1]):
            assert edges[0, i] != edges[1, i], f"Self-loop at edge {i}"
    
    def test_no_duplicate_edges(self, dataset_stub, known_cnf):
        """Test that there are no duplicate edges."""
        data = dataset_stub.create_variable_graph(
            known_cnf['clauses'], known_cnf['n_vars']
        )
        
        edges = data.edge_index
        edge_list = []
        for i in range(edges.shape[1]):
            a, b = edges[0, i].item(), edges[1, i].item()
            edge_list.append((min(a, b), max(a, b)))
        
        assert len(edge_list) == len(set(edge_list)), "Duplicate edges found"
    
    def test_variable_not_in_any_clause(self, dataset_stub):
        """Test handling of variables that don't appear in any clause."""
        clauses = [[1, 2]]  # x3 is defined but not used
        n_vars = 3
        
        data = dataset_stub.create_variable_graph(clauses, n_vars)
        
        # x3 (index 2) should have all zero features except indicators
        assert data.x[2, 3] == 0, "Unused var should have 0 co-occurrence degree"
        assert data.x[2, 8] == 0, "Unused var should have 0 normalized_freq"
    
    def test_co_occurrence_degree_calculation(self, dataset_stub):
        """Test that co-occurrence degree is calculated correctly."""
        # In clause (1,2,3): each var gets +2 co-occurrences
        # (x1 with x2, x1 with x3, x2 with x1, x2 with x3, etc.)
        clauses = [[1, 2, 3]]
        n_vars = 3
        
        data = dataset_stub.create_variable_graph(clauses, n_vars)
        
        # Each variable co-occurs with 2 others in this clause
        for i in range(3):
            assert data.x[i, 3] == 2.0, f"Var {i} should have co-occurrence degree 2"
    
    def test_horn_count_calculation(self, dataset_stub):
        """Test that horn_count counts number of Horn clauses, not co-occurrences."""
        # Clause 0: [1, 2] -> Not Horn (2 pos literals)
        # Clause 1: [-1, -2] -> Horn
        # Clause 2: [1] -> Horn
        clauses = [[1, 2], [-1, -2], [1]]
        n_vars = 2
        
        data = dataset_stub.create_variable_graph(clauses, n_vars)
        
        # x1 (idx 0):
        # - in c0 (Not Horn) -> +0
        # - in c1 (Horn) -> +1
        # - in c2 (Horn) -> +1
        # Total horn_count = 2
        
        # x2 (idx 1):
        # - in c0 (Not Horn) -> +0
        # - in c1 (Horn) -> +1
        # Total horn_count = 1
        
        assert data.x[0, 0] == 2.0, f"x1 horn_count expected 2.0, got {data.x[0, 0]}"
        assert data.x[1, 0] == 1.0, f"x2 horn_count expected 1.0, got {data.x[1, 0]}"


class TestCGStructureComprehensive:
    """Comprehensive structural tests for Clause Graph."""
    
    @pytest.fixture
    def simple_cnf(self):
        """
        CNF: (x1 ∨ x2) ∧ (¬x1 ∨ x3) ∧ (¬x2 ∨ ¬x3)
        
        Clause 0: [1, 2]
        Clause 1: [-1, 3]
        Clause 2: [-2, -3]
        
        Edges (connect clauses sharing complementary literal):
        - c0 and c1 share x1 (pos in c0, neg in c1) -> Edge (0, 1)
        - c0 and c2 share x2 (pos in c0, neg in c2) -> Edge (0, 2)
        - c1 and c2 share x3 (pos in c1, neg in c2) -> Edge (1, 2)
        """
        return {
            'clauses': [[1, 2], [-1, 3], [-2, -3]],
            'n_vars': 3,
            'n_clauses': 3,
        }
    
    @pytest.fixture
    def dataset_stub(self):
        return object.__new__(SATDataset)
        
    def test_node_counts(self, dataset_stub, simple_cnf):
        pass # Covered generally, but specific logic check here
        data = dataset_stub.create_clause_graph(
            simple_cnf['clauses'], simple_cnf['n_vars']
        )
        assert data.num_nodes == 3

    def test_feature_dimensions(self, dataset_stub, simple_cnf):
        data = dataset_stub.create_clause_graph(
            simple_cnf['clauses'], simple_cnf['n_vars']
        )
        assert data.x.shape == (3, 22), f"Expected (3, 22), got {data.x.shape}"
        
    def test_connectivity(self, dataset_stub, simple_cnf):
        """Test that edges connect clauses with complementary literals."""
        data = dataset_stub.create_clause_graph(
            simple_cnf['clauses'], simple_cnf['n_vars']
        )
        
        edges = data.edge_index
        edge_set = set()
        for i in range(edges.shape[1]):
            a, b = edges[0, i].item(), edges[1, i].item()
            edge_set.add(tuple(sorted((a, b))))
            
        expected = {(0, 1), (0, 2), (1, 2)}
        assert edge_set == expected, f"Expected {expected}, got {edge_set}"
        
    def test_positional_encoding(self, dataset_stub, simple_cnf):
        data = dataset_stub.create_clause_graph(
            simple_cnf['clauses'], simple_cnf['n_vars']
        )
        pos_enc = data.x[:, :10]
        assert (pos_enc >= -1).all() and (pos_enc <= 1).all()
        assert not torch.allclose(pos_enc, torch.zeros_like(pos_enc))

    def test_feature_values(self, dataset_stub, simple_cnf):
        data = dataset_stub.create_clause_graph(
            simple_cnf['clauses'], simple_cnf['n_vars']
        )
        
        # Clause 0: [1, 2] -> Binary (len 2), not Horn (2 pos)
        assert data.x[0, 13] == 2.0, "c0 length should be 2"
        assert data.x[0, 15] == 1.0, "c0 should be binary"
        assert data.x[0, 17] == 0.0, "c0 should NOT be Horn"
        
        # Clause 1: [-1, 3] -> Binary, Horn (1 pos)
        assert data.x[1, 15] == 1.0, "c1 should be binary"
        assert data.x[1, 17] == 1.0, "c1 should be Horn"
        
        # Clause 2: [-2, -3] -> Binary, Horn (0 pos)
        assert data.x[2, 15] == 1.0, "c2 should be binary"
        assert data.x[2, 17] == 1.0, "c2 should be Horn"
        assert data.x[2, 21] == 1.0, "c2 should be negative clause"


class TestGraphProperties:
    """Tests for general graph properties that should hold for all types."""
    
    @pytest.fixture
    def dataset_stub(self):
        return object.__new__(SATDataset)
    
    def test_lcg_num_nodes_property(self, dataset_stub):
        """Test that num_nodes property is correct for LCG."""
        clauses = [[1, -2, 3], [-1, 2]]
        n_vars = 3
        
        data = dataset_stub.create_literal_clause_graph(clauses, n_vars)
        
        expected_nodes = 2 * n_vars + len(clauses)  # literals + clauses
        assert data.num_nodes == expected_nodes
    
    def test_vcg_num_nodes_property(self, dataset_stub):
        """Test that num_nodes property is correct for VCG."""
        clauses = [[1, -2, 3], [-1, 2]]
        n_vars = 3
        
        data = dataset_stub.create_variable_clause_graph(clauses, n_vars)
        
        expected_nodes = n_vars + len(clauses)  # vars + clauses
        assert data.num_nodes == expected_nodes
    
    def test_vg_num_nodes_property(self, dataset_stub):
        """Test that num_nodes property is correct for VG."""
        clauses = [[1, -2, 3], [-1, 2]]
        n_vars = 3
        
        data = dataset_stub.create_variable_graph(clauses, n_vars)
        
        assert data.num_nodes == n_vars
    
    def test_features_not_nan(self, dataset_stub):
        """Test that no features are NaN."""
        clauses = [[1, -2, 3], [-1, 2], [3]]
        n_vars = 3
        
        # LCG
        data = dataset_stub.create_literal_clause_graph(clauses, n_vars)
        assert not torch.isnan(data["literal"].x).any()
        assert not torch.isnan(data["clause"].x).any()
        
        # VCG
        data = dataset_stub.create_variable_clause_graph(clauses, n_vars)
        assert not torch.isnan(data["var"].x).any()
        assert not torch.isnan(data["clause"].x).any()
        
        # VG
        data = dataset_stub.create_variable_graph(clauses, n_vars)
        assert not torch.isnan(data.x).any()
    
    def test_features_not_inf(self, dataset_stub):
        """Test that no features are infinite."""
        clauses = [[1, -2, 3], [-1, 2], [3]]
        n_vars = 3
        
        # LCG
        data = dataset_stub.create_literal_clause_graph(clauses, n_vars)
        assert not torch.isinf(data["literal"].x).any()
        assert not torch.isinf(data["clause"].x).any()
        
        # VCG
        data = dataset_stub.create_variable_clause_graph(clauses, n_vars)
        assert not torch.isinf(data["var"].x).any()
        assert not torch.isinf(data["clause"].x).any()
        
        # VG
        data = dataset_stub.create_variable_graph(clauses, n_vars)
        assert not torch.isinf(data.x).any()


class TestLargerFormulas:
    """Tests with larger formulas to catch scaling issues."""
    
    @pytest.fixture
    def dataset_stub(self):
        return object.__new__(SATDataset)
    
    def test_medium_formula_lcg(self, dataset_stub):
        """Test LCG with a medium-sized formula."""
        n_vars = 20
        # Random-ish clauses
        clauses = [
            [1, -2, 3], [-4, 5], [6, -7, 8, -9], [10],
            [-1, -2, -3], [4, 5, 6], [-7, 8], [9, -10],
            [11, -12, 13, -14, 15], [-16, 17], [18, -19, 20],
        ]
        
        data = dataset_stub.create_literal_clause_graph(clauses, n_vars)
        
        assert data["literal"].x.shape == (2 * n_vars, 12)
        assert data["clause"].x.shape == (len(clauses), 22)
        assert data.num_nodes == 2 * n_vars + len(clauses)
    
    def test_medium_formula_vcg(self, dataset_stub):
        """Test VCG with a medium-sized formula."""
        n_vars = 20
        clauses = [
            [1, -2, 3], [-4, 5], [6, -7, 8, -9], [10],
            [-1, -2, -3], [4, 5, 6], [-7, 8], [9, -10],
        ]
        
        data = dataset_stub.create_variable_clause_graph(clauses, n_vars)
        
        assert data["var"].x.shape == (n_vars, 12)
        assert data["clause"].x.shape == (len(clauses), 22)
    
    def test_medium_formula_vg(self, dataset_stub):
        """Test VG with a medium-sized formula."""
        n_vars = 20
        clauses = [
            [1, -2, 3], [-4, 5], [6, -7, 8, -9], [10],
            [-1, -2, -3], [4, 5, 6], [-7, 8], [9, -10],
        ]
        
        data = dataset_stub.create_variable_graph(clauses, n_vars)
        
        assert data.x.shape == (n_vars, 12)
        assert data.num_nodes == n_vars


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
