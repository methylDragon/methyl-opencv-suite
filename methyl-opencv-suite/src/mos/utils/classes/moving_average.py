from collections import deque
import numpy as np

__all__ = [
    'MovingAverage'
]

class MovingAverage:
    """
    Compute moving average with or without window.
    """
    def __init__(self,
                 window=None,
                 initial_entry=None):
        self.window = window
        self.init = False
        self.entry_deque = None
        self.average = None
        self.data_shape = None

        if initial_entry is not None:
            initial_entry = np.array(initial_entry)
            self._initialise(initial_entry)

    def update(self, entry):
        entry = np.array(entry)

        if not self.init:
            self._initialise(entry)

        assert entry.shape == self.data_shape

        oldest_entry = self.entry_deque[0]
        self.average += (entry - oldest_entry) / self.window
        self.entry_deque.append(entry)

        return self.average

    def _initialise(self, entry):
        assert self.window is not None

        if entry is not None:
            self.data_shape = entry.shape

            self.entry_deque = deque(np.zeros((self.window, *entry.shape)),
                                     maxlen=self.window)
            self.entry_deque.appendleft(entry)

            self.average = np.zeros(entry.shape)
            self.init = True
