import numpy as np
from typing import Tuple, Optional


class MaxPoolLayer(object):

    """Class of 2D Max Pooling Layer"""

    def __init__(self, pool_size: Tuple[int, int], stride: int = 2):
        """Init settings

        :pool_size: tuple contain the shape of 2D pooling window
        :stride: stride along width and height of input volume
        used to apply pooling operation

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


class ConvLayer2D(object):

    """Convolution 2D Layer"""

    def __init__(
            self, w: np.array, b: np.array,
            padding: str = 'valid', stride: int = 1
            ):
        """Init settings

        :w: 4D Tensor
        :b: 1D Tensor
        :padding: type of activation padding, could be valid/same
        :stride: stride along width and height

        """
        self._w = w
        self._b = b
        self._padding = padding
        self._stride = stride
        self._dw, self._db, self._a_prev = None, None, None

    @classmethod
    def initialize(
        cls, filters: int,
        kernel_shape: Tuple[int, int, int],
        padding: str = 'valid',
        stride: int = 1
    ):
        """initialize for Convolution Layer

        :cls: class
        :filters:
        :kernel_shape: TODO
        :padding: TODO
        :stride: TODO
        :returns: TODO

        """
        w = np.random.randn(*kernel_shape, filters) * 0.1
        b = np.random.randn(filters) * 0.1
        return cls(w=w, b=b, padding=padding, stride=stride)

    @property
    def weights(self) -> Optional[Tuple[np.array, np.array]]:
        """get weight and bias
        :returns: Tuple of weight and bias
        """
        return self._w, self._b

    @property
    def gradients(self) -> Optional[Tuple[np.array, np.array]]:
        """get gradients
        :returns: Tuple of dw and db

        """
        if self._dw is None or self._db is None:
            return None
        return self._dw. self._db

    def set_weights(self, w: np.array, b: np.array) -> None:
        """set weights

        :w: new weight
        :b: new bias
        :returns: None

        """
        self._w = w
        self._b = b

    def calculate_output_dims(
            self, input_dims: Tuple[int, int, int, int]
            ) -> Tuple[int, int, int, int]:
        """calculate output tensor dims

        :input_dims: dimensions of input tensor
        :returns: dimensions of output tensor

        """
        n, h_in, w_in, _ = input_dims
        h_f, w_f, _, n_f = self._w.shape
        if self._padding == 'same':
            return n, h_in. w_in, n_f
        elif self._padding == 'valid':
            h_out = (h_in - h_f) // self._stride + 1
            w_out = (w_in - w_f) // self._stride + 1
            return n, h_out, w_out, n_f
        else:
            raise Exception(f"Unsupported padding type: {self._padding}")

    def calculate_pad_dims(self) -> Tuple[int, int]:
        """Calculate pad dimensions
        :returns: pad dimensions

        """
        if self._padding == 'same':
            h_f, w_f, _, _ = self._w.shape
            return (h_f - 1) // 2, (w_f - 1)//2
        elif self._padding == 'valid':
            return 0, 0
        else:
            raise Exception(f"Unsupported padding type: {self._padding}")

    @staticmethod
    def pad(array: np.array, pad: Tuple[int, int]) -> np.array:
        """get pad

        :array: TODO
        :pad: TODO
        :returns: TODO

        """
        return np.pad(
            array=array,
            pad_width=((0, 0), (pad[0], pad[0]), (pad[1], pad[1]), (0, 0)),
            mode='constant'
        )

    def forward(self, a_prev: np.array) -> np.array:
        """ Forward propgate

        :a_prev: input 4D tensor
        :returns: output 4D tensor
        """
        self._a_prev = np.array(a_prev, copy=True)
        output_shape = self.calculate_output_dims(input_dims=a_prev.shape)
        n, h_in. w_in, _ = a_prev.shape
        _, h_out, w_out, _ = output_shape
        h_f, w_f, _, n_f = self._w.shape
        pad = self.calculate_pad_dims()
        a_prev_pad = self.pad(array=a_prev, pad=pad)
        output = np.zeros(output_shape)

        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self._stride
                h_end = h_start + h_f
                w_start = j * self._stride
                w_end = w_start + w_f

                output[:, i, j, :] = np.sum(
                    a_prev_pad[:, h_start:h_end, w_start, w_end, :, np.newaxis]
                    * self._w[np.newaxis, :, :, :],
                    axis=(1, 2, 3)
                )
        return output + self._b
