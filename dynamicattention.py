import numpy as np
from typing import List, Tuple

class DynamicAttention:
    def __init__(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray, lazy_threshold: int = 10):
        """
        Initialize with matrices Q (query), K (key), and V (value).
        
        Args:
            Q (np.ndarray): Query matrix (n x d)
            K (np.ndarray): Key matrix (n x d)
            V (np.ndarray): Value matrix (n x d)
            lazy_threshold (int): Number of updates to accumulate before forced recalculation
        """
        self.Q = Q
        self.K = K
        self.V = V
        self.n, self.d = Q.shape
        self.lazy_threshold = lazy_threshold
        self.pending_updates_K: List[Tuple[int, int, float]] = []
        self.pending_updates_V: List[Tuple[int, int, float]] = []
        self.num_updates = 0
        
        self._initialize_matrices()

    def _initialize_matrices(self):
        """Initialize the attention-related matrices."""
        self.A = np.exp(np.dot(self.Q, self.K.T))
        self.D = np.diag(self.A.sum(axis=1))
        self.D_inv = np.linalg.inv(self.D)
        self.att_matrix = self.D_inv @ self.A @ self.V

    def apply_lazy_updates(self):
        """Apply all the lazy updates stored for both K and V matrices."""
        if not self.pending_updates_K and not self.pending_updates_V:
            return

        for i, j, delta in self.pending_updates_K:
            self.K[i, j] += delta
            self.A[i, :] = np.exp(np.dot(self.Q[i, :], self.K.T))
            self.D[i, i] = self.A[i, :].sum()

        for i, j, delta in self.pending_updates_V:
            self.V[i, j] += delta

        self.D_inv = np.linalg.inv(self.D)
        self.att_matrix = self.D_inv @ self.A @ self.V

        self.pending_updates_K.clear()
        self.pending_updates_V.clear()
        self.num_updates = 0

    def _lazy_update(self, updates: List[Tuple[int, int, float]], matrix: str):
        """Generic method for lazy updates to K or V matrices."""
        pending_updates = self.pending_updates_K if matrix == 'K' else self.pending_updates_V
        pending_updates.extend(updates)
        self.num_updates += len(updates)
        
        if self.num_updates >= self.lazy_threshold:
            self.apply_lazy_updates()

    def lazy_update_K(self, updates: List[Tuple[int, int, float]]):
        """Update entries of K lazily."""
        self._lazy_update(updates, 'K')

    def lazy_update_V(self, updates: List[Tuple[int, int, float]]):
        """Update entries of V lazily."""
        self._lazy_update(updates, 'V')

    def query(self, i: int, j: int) -> float:
        """
        Query the attention matrix for the value at position (i, j).
        If there are pending lazy updates, apply them first.
        """
        self.apply_lazy_updates()
        return self.att_matrix[i, j]

    def get_attention_matrix(self) -> np.ndarray:
        """Return the current attention matrix, applying any pending updates."""
        self.apply_lazy_updates()
        return self.att_matrix

# Example usage
def main():
    np.random.seed(42)  # For reproducibility
    n, d = 5, 3  # Increased dimensions for a more interesting example
    Q = np.random.rand(n, d)
    K = np.random.rand(n, d)
    V = np.random.rand(n, d)

    print("Initial matrices:")
    print("Q:", Q)
    print("K:", K)
    print("V:", V)

    dynamic_attention = DynamicAttention(Q, K, V, lazy_threshold=3)

    print("\nInitial Attention Matrix:")
    print(dynamic_attention.get_attention_matrix())

    # Perform some lazy updates on K and V
    k_updates = [(1, 1, 0.5), (2, 0, -0.3), (0, 2, 0.2)]
    v_updates = [(2, 1, -0.1), (3, 0, 0.4), (4, 2, -0.2)]

    print("\nApplying lazy updates to K:", k_updates)
    dynamic_attention.lazy_update_K(k_updates)

    print("Applying lazy updates to V:", v_updates)
    dynamic_attention.lazy_update_V(v_updates)

    print("\nQuerying specific positions:")
    print("Position (1, 1):", dynamic_attention.query(1, 1))
    print("Position (2, 0):", dynamic_attention.query(2, 0))

    print("\nFinal Attention Matrix:")
    print(dynamic_attention.get_attention_matrix())

    # Demonstrate the effect of updates
    print("\nDemonstrating the effect of updates:")
    original_K = K.copy()
    original_V = V.copy()

    for i, j, delta in k_updates:
        original_K[i, j] += delta
    for i, j, delta in v_updates:
        original_V[i, j] += delta

    print("Manual calculation of attention matrix:")
    A = np.exp(np.dot(Q, original_K.T))
    D = np.diag(A.sum(axis=1))
    D_inv = np.linalg.inv(D)
    manual_att_matrix = D_inv @ A @ original_V
    print(manual_att_matrix)

    print("\nDifference between manual calculation and DynamicAttention:")
    print(np.abs(manual_att_matrix - dynamic_attention.get_attention_matrix()).max())

if __name__ == "__main__":
    main()
