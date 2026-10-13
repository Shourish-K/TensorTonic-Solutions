import numpy as np

def create_sequence(start: float, stop: float, param: float, kind: str) -> np.ndarray:
    return (np.arange(start, stop, param, dtype=float) if kind == 'arange' else np.linspace(start, stop, param, dtype=float))
