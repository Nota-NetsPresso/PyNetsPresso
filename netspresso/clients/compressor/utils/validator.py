from netspresso.enums.compression import CompressionMethod


class CompressionParamsValidator:
    def __init__(self, compression_method, layers):
        self.compression_method = compression_method
        self.layers = layers
        self.supported_method = list(CompressionMethod.__members__.keys())

    def validate(self):
        compression_methods = {
            "PR_L2": self._validate_pr_ratio,
            "PR_GM": self._validate_pr_ratio,
            "PR_NN": self._validate_pr_ratio,
            "PR_ID": self._validate_pr_index,
            "FD_TK": self._validate_fd_rank2,
            "FD_SVD": self._validate_fd_rank1,
            "FD_CP": self._validate_fd_rank1,
        }

        validation_method = compression_methods.get(self.compression_method)
        if validation_method:
            validation_method()
        else:
            raise ValueError(
                f"Invalid compression_method: {self.compression_method}. Please choose from {self.supported_method}."
            )

    def _validate_pr_ratio(self):
        for layer in self.layers:
            if not layer.use:
                continue
            if len(layer.values) != 1:
                raise ValueError(f"The number of values should be 1, but got {len(layer.values)}.")
            if not isinstance(layer.values[0], float):
                raise ValueError(f"The type of the input value should be float, but got {layer.values}.")
            if not 0.0 < layer.values[0] <= 1.0:
                raise ValueError(f"The range of input value should be 0.0 < x <= 1.0, but got {layer.values}.")

    def _validate_pr_index(self):
        for layer in self.layers:
            if not layer.use:
                continue
            if layer.channels[0] <= len(layer.values):
                raise ValueError(
                    f"The number of values should be less than {layer.channels[0]}, but got {len(layer.values)}."
                )
            if not all(isinstance(x, int) for x in layer.values):
                raise ValueError(f"The type of the input values should be integer, but got {layer.values}")
            if not all(0 <= x < layer.channels[0] for x in layer.values):
                raise ValueError(
                    f"The range of input values should be 0 <= x < out channels({layer.channels[0]}), but got {layer.values}"
                )

    def _validate_fd_rank2(self):
        for layer in self.layers:
            if not layer.use:
                continue
            if len(layer.values) != 2:
                raise ValueError(f"The number of values should be 2, but got {len(layer.values)}.")
            if not all(isinstance(x, int) for x in layer.values):
                raise ValueError(f"The type of the input values should be integer, but got {layer.values}")
            if not all(x > 0 and x <= channel for x, channel in zip(layer.values, layer.channels)):
                raise ValueError(
                    f"The range of input values should be 0 < x <= channels({layer.channels}), but got {layer.values}"
                )

    def _validate_fd_rank1(self):
        for layer in self.layers:
            if not layer.use:
                continue
            if len(layer.values) != 1:
                raise ValueError(f"The number of values should be 1, but got {len(layer.values)}.")
            if not isinstance(layer.values[0], int):
                raise ValueError(f"The type of the input values should be integer, but got {layer.values}")
            if not 0 < layer.values[0] <= min(layer.channels):
                raise ValueError(
                    f"The range of input values should be 0 < x < min(in channels, out channels)({min(layer.channels)}), but got {layer.values}"
                )
