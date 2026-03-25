#!/usr/bin/env python3
"""Export all-MiniLM-L6-v2 to ONNX format for lightweight deployment.

Usage:
    pip install sentence-transformers onnx onnxruntime
    python export_model_to_onnx.py [--output-dir ./onnx_model]

This only needs to be run once. The exported model can then be used
by the MCP server without requiring PyTorch or sentence-transformers.
"""
import argparse
import json
import os


def export(output_dir: str = "./onnx_model"):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Error: sentence-transformers is required for export.")
        print("Install with: pip install sentence-transformers")
        raise SystemExit(1)

    try:
        import torch
    except ImportError:
        print("Error: torch is required for export.")
        raise SystemExit(1)

    print("Loading model: all-MiniLM-L6-v2 ...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    os.makedirs(output_dir, exist_ok=True)

    # Save tokenizer
    tokenizer = model.tokenizer
    tokenizer.save_pretrained(output_dir)

    # Export transformer to ONNX
    transformer = model[0].auto_model
    transformer.eval()

    dummy_input = tokenizer(
        "This is a test sentence for export.",
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
    )

    onnx_path = os.path.join(output_dir, "model.onnx")
    print(f"Exporting ONNX model to {onnx_path} ...")

    torch.onnx.export(
        transformer,
        (dummy_input["input_ids"], dummy_input["attention_mask"]),
        onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "last_hidden_state": {0: "batch", 1: "sequence"},
        },
        opset_version=14,
    )

    # Save pooling config
    pooling_config = {"pooling_mode": "mean"}
    with open(os.path.join(output_dir, "pooling_config.json"), "w") as f:
        json.dump(pooling_config, f)

    model_size = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"Export complete! Model size: {model_size:.1f} MB")
    print(f"Files saved to: {output_dir}/")
    print(f"Set ONNX_MODEL_PATH={os.path.abspath(output_dir)} to use with server.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export embedding model to ONNX")
    parser.add_argument(
        "--output-dir", default="./onnx_model",
        help="Directory to save model and tokenizer (default: ./onnx_model)",
    )
    args = parser.parse_args()
    export(args.output_dir)
