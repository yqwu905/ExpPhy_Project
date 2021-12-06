import numpy as np
from typing import Tuple


class MaxPoolLayer(object):

    """Class of 2D Max Pooling Layer"""

    def __init__(self, pool_size: Tuple[int, int], stride: int = 2):
        """Init settings

        :pool_size: tuple contain the shape of 2D pooling window
        :stride: stride along width and height of input volume used to apply pooling operation

        """
        self._pool_size = pool_size
        self._stride = stride
        self._input = None
        self._cache = {}

    def foward(self, input: np.array) -> np.array:
        """Forward method for Max Pooling 2D layer.

        :input: input tensor
        :returns: output tensor

        """
        self._input = np.array(input, copy=True)
        n, h_in, w_in, c = input.shape
        h_pool, w_pool = self._pool_size
        h_out = 1 + (h_in - h_pool) // self._stride
        w_out = 1 + (w_in - w_pool) // self._stride
        output = np.zeros((n. h_out, w_out, c))

        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self._stride
                h_end = h_start + h_pool
                w_start = j * self._stride
                w_end = w_start + w_pool
                input_slice = input[:, h_start:h_end, w_start:w_end, :]
                self._save_mask(x=input_slice, cords=(i, j))
                output[:, i, j, :] = np.max(input_slice, axis=(1, 2))
        return output

    def backward(self, top_dif: np.array) -> np.array:
        """Backward method for Max Pooling 2D layer

        :top_diff: input tensor
        :returns: output tensor

        """
        bot_dif = np.zeros_like(self._input)
        _, h_out, w_out, _ = top_dif.shape
        h_pool, w_pool = self._pool_size

        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self._stride
                h_end = h_start + h_pool
                w_start = j * self._stride
                w_end = w_start + w_pool
                bot_dif[:, h_start:h_end, w_start:w_end, :] += \
                    top_dif[:, i:i + 1, j:j + 1, :] * self._cache[(i, j)]
        return bot_dif
