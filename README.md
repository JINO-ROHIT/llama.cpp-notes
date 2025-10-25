# nano llama-cpp

The idea is to learn and implement a more simpler version of `llama.cpp` from the earlier legacy commits of the repository.
Reference - https://github.com/ggml-org/llama.cpp

Currently supports only for metal architecture.

### Download the model weights from here and put them under `models`

```
https://huggingface.co/meta-llama/Llama-2-7b/
```

### How to perform inference

1. convert the weights to ggml format first.

```python3
python3 convert-pth-to-ggml.py <model_dir>
```



