from pathlib import Path

import numpy as np


def quantize_linear_parameters(
    weights: np.ndarray,
    bias: np.ndarray,
    input_range: tuple[float, float] = (-3.0, 3.0),
) -> dict[str, np.ndarray]:
    """Quantize a linear layer for int8 inputs and per-output int8 weights."""
    weights = np.asarray(weights, dtype=np.float32)
    bias = np.asarray(bias, dtype=np.float32)

    if weights.ndim != 2 or bias.shape != (weights.shape[0],):
        raise ValueError("Expected weights shaped (outputs, inputs) and one bias per output.")
    if not np.isfinite(weights).all() or not np.isfinite(bias).all():
        raise ValueError("Weights and bias must contain only finite values.")

    input_min, input_max = input_range
    if not np.isfinite([input_min, input_max]).all() or input_min >= 0 or input_max <= 0:
        raise ValueError("Input range must be finite and include zero.")
    if not np.isclose(input_min, -input_max):
        raise ValueError("INT8 input range must be symmetric around zero.")

    input_scale = np.float32(input_max / 127.0)
    max_abs_weights = np.max(np.abs(weights), axis=1)
    weight_scales = np.where(max_abs_weights > 0, max_abs_weights / 127.0, 1.0)
    quantized_weights = np.clip(
        np.rint(weights / weight_scales[:, np.newaxis]), -127, 127
    ).astype(np.int8)

    bias_scale = input_scale * weight_scales
    quantized_bias_float = np.rint(bias / bias_scale)
    int32_max = np.iinfo(np.int32).max
    if np.any(np.abs(quantized_bias_float) > int32_max):
        raise OverflowError("Quantized bias does not fit in int32.")
    quantized_bias = quantized_bias_float.astype(np.int32)

    return {
        "weights": quantized_weights,
        "bias": quantized_bias,
        "input_scale": np.asarray(input_scale, dtype=np.float32),
        "weight_scales": weight_scales.astype(np.float32),
        "input_range": np.asarray(input_range, dtype=np.float32),
    }


def predict_int8_linear(
    inputs: np.ndarray,
    quantized_model: dict[str, np.ndarray],
) -> np.ndarray:
    """Run integer accumulation and return dequantized logits."""
    inputs = np.asarray(inputs, dtype=np.float32)
    if inputs.ndim == 1:
        inputs = inputs.reshape(1, -1)
    if inputs.ndim != 2 or inputs.shape[1] != quantized_model["weights"].shape[1]:
        raise ValueError("Inputs must have shape (batch, model input features).")
    if not np.isfinite(inputs).all():
        raise ValueError("Inputs must contain only finite values.")

    input_min, input_max = quantized_model["input_range"]
    clipped_inputs = np.clip(inputs, input_min, input_max)
    quantized_inputs = np.clip(
        np.rint(clipped_inputs / quantized_model["input_scale"]), -127, 127
    ).astype(np.int8)

    accumulators = (
        quantized_inputs.astype(np.int32)
        @ quantized_model["weights"].astype(np.int32).T
    )
    accumulators += quantized_model["bias"]
    output_scales = (
        quantized_model["input_scale"] * quantized_model["weight_scales"]
    )
    return accumulators.astype(np.float32) * output_scales


def export_int8_linear(
    weights: np.ndarray,
    bias: np.ndarray,
    output_path: str | Path,
    input_range: tuple[float, float] = (-3.0, 3.0),
) -> tuple[dict[str, np.ndarray], int]:
    """Write a compressed, portable NPZ containing the quantized layer."""
    output_path = Path(output_path)
    if output_path.suffix.lower() != ".npz":
        raise ValueError("INT8 model output path must use the .npz extension.")

    quantized_model = quantize_linear_parameters(weights, bias, input_range)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as model_file:
        np.savez_compressed(model_file, **quantized_model)
    return quantized_model, output_path.stat().st_size
