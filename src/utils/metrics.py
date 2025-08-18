from typing import Any, Dict, Optional, Sequence

import numpy as np

from .io_utils import save_json_merge


class StatsRecorder:
    def __init__(self, stats_path: str) -> None:
        self.stats_path = stats_path

    def update(self, stage: str, stats: Dict[str, Any]) -> None:
        save_json_merge(self.stats_path, {stage: stats})

    def histogram_png(self, values: Sequence[float], path: str, title: Optional[str] = None) -> None:
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except Exception:
            return
        import os
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        arr = np.asarray(values, dtype=float)
        plt.figure(figsize=(6, 4))
        plt.hist(arr, bins=50, alpha=0.75)
        if title:
            plt.title(title)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()


