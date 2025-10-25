"""
Convert a LLaMA model checkpoint to a GGML-compatible binary file.

Steps:
1. Load model parameters and weights from a PyTorch checkpoint.
2. Load the SentencePiece tokenizer.
3. Write model hyperparameters and vocabulary to the GGML file.
4. Write each model tensor (with metadata) to the GGML file.

Usage:
    python convert-pth-to-ggml.py <model_dir> [ftype]
    
Arguments:
    model_dir : Path to the model directory (must contain params.json, tokenizer.model, and .pth file)
    ftype     : 0 for float32, 1 for float16 (default = 1)
"""

import sys
import json
import struct
import numpy as np
import torch
from sentencepiece import SentencePieceProcessor
from pathlib import Path


def write_header(fout, hparams, ftype):
    """Write GGML file header with model hyperparameters."""
    fout.write(struct.pack("i", 0x67676D6C))  # magic: 'ggml'
    fout.write(struct.pack("i", hparams["vocab_size"]))
    fout.write(struct.pack("i", hparams["dim"]))
    fout.write(struct.pack("i", hparams["multiple_of"]))
    fout.write(struct.pack("i", hparams["n_heads"]))
    fout.write(struct.pack("i", hparams["n_layers"]))
    fout.write(struct.pack("i", 64))  # rotary dimension
    fout.write(struct.pack("i", ftype))


def write_vocab(fout, tokenizer):
    """Write vocabulary to GGML file."""
    vocab_size = tokenizer.vocab_size()
    for i in range(vocab_size):
        token = tokenizer.id_to_piece(i)
        token_bytes = token.encode("utf-8")
        fout.write(struct.pack("i", len(token_bytes)))
        fout.write(token_bytes)


def write_tensor(fout, name, tensor, ftype):
    """Write a single tensor (with metadata) to the GGML file."""
    data = tensor.half().numpy().squeeze()  # force float16 since bfloat16 is not compactible on numpy
    n_dims = len(data.shape)

    # Default: use float16 unless user requested float32 or tensor is 1D
    ftype_cur = 1
    if ftype == 0 or n_dims == 1:
        data = data.astype(np.float32)
        ftype_cur = 0

    encoded_name = name.encode("utf-8")
    fout.write(struct.pack("iii", n_dims, len(encoded_name), ftype_cur))
    for dim in reversed(data.shape):
        fout.write(struct.pack("i", dim))
    fout.write(encoded_name)

    data.tofile(fout)


def main():
    if len(sys.argv) < 2:
        print("Usage: python convert-ckpt-to-ggml.py <model_dir> [ftype]")
        print("  ftype == 0 → float32")
        print("  ftype == 1 → float16 (default)")
        sys.exit(1)

    model_dir = Path(sys.argv[1])
    ftype = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    if ftype not in (0, 1):
        print(f"Invalid ftype: {ftype}")
        sys.exit(1)

    ftype_str = ["f32", "f16"]
    output_path = model_dir / f"ggml-model-{ftype_str[ftype]}.bin"

    hparams_path = model_dir / "params.json"
    model_path = model_dir / "consolidated.00.pth"
    tokenizer_path = model_dir / "tokenizer.model"

    with open(hparams_path, "r") as f:
        hparams = json.load(f)

    tokenizer = SentencePieceProcessor(model_file=str(tokenizer_path))
    hparams["vocab_size"] = tokenizer.vocab_size()

    print("Loaded model parameters:")
    print(json.dumps(hparams, indent=2))

    model = torch.load(model_path, map_location="cpu")

    with open(output_path, "wb") as fout:
        write_header(fout, hparams, ftype)

        print("\nWriting vocabulary...")
        write_vocab(fout, tokenizer)

        print("\nWriting model tensors...")
        for name, tensor in model.items():
            if name.endswith("freqs"):  # skip RoPE frequency buffers
                continue
            print(f"  Processing: {name} {tuple(tensor.shape)} ({tensor.dtype})")
            write_tensor(fout, name, tensor, ftype)

    print(f"\nOutput file written to: {output_path}\n")


if __name__ == "__main__":
    main()
