"""
Deep learning interface placeholder for future AI traffic prediction.

Phase 2 reserves this abstract interface. Actual model integration is
future work. The DummyTrafficPredictor returns constant free-flow results.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np


class TrafficPredictorInterface(ABC):
    """Abstract interface for ML-based traffic prediction models."""

    @abstractmethod
    def predict_congestion(
        self,
        edge_ids: List[Tuple[int, int, int]],
        future_timestamps: List[int],
    ) -> np.ndarray:
        """
        Predict congestion multipliers for given edges at future times.

        :param edge_ids: List of ``(u, v, key)`` edge identifiers.
        :param future_timestamps: List of UNIX timestamps (seconds since midnight).
        :returns: ``np.ndarray`` of shape ``(len(edge_ids), len(future_timestamps))``
                  with predicted congestion multipliers (1.0 = free flow).
        """
        ...

    @abstractmethod
    def train(self, historical_data_path: str) -> None:
        """Train the prediction model on historical traffic data."""
        ...


class DummyTrafficPredictor(TrafficPredictorInterface):
    """
    Placeholder predictor. Returns 1.0 (free flow) for all queries.

    Replace this with a real model (e.g., GCN, LSTM, Transformer) when
    historical traffic data is available.
    """

    def predict_congestion(
        self,
        edge_ids: List[Tuple[int, int, int]],
        future_timestamps: List[int],
    ) -> np.ndarray:
        return np.ones((len(edge_ids), len(future_timestamps)))

    def train(self, historical_data_path: str) -> None:
        pass
