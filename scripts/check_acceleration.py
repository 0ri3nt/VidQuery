"""Report optional Torch acceleration without assuming that a GPU exists."""


def main() -> None:
    try:
        import torch
    except ImportError:
        print("Torch is not installed; VidQuery will use CPU-safe/base components.")
        return

    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available; optional ML components will run on CPU.")


if __name__ == "__main__":
    main()
