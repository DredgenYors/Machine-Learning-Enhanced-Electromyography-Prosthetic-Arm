# filtering/pipeline_state.py
from dataclasses import dataclass, field
import numpy as np
from typing import Optional, List, Dict

@dataclass
class PipelineState:
    fs: int = 1000
    original_signal: Optional[np.ndarray] = None
    filtered_signal: Optional[np.ndarray] = None
    selected_channels: Optional[List[int]] = None
    exercise: Optional[np.ndarray] = None
    filter_history: List[str] = field(default_factory=list)

    extracted_features: Optional[Dict[str, np.ndarray]] = None
    feature_matrix: Optional[np.ndarray] = None
    window_centers: Optional[np.ndarray] = None

    window_size: int = 250
    overlap: float = 0.5

    @property
    def num_channels(self) -> int:
        if self.original_signal is None:
            return 0
        return self.original_signal.shape[1]
