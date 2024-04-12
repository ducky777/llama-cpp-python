python llama_cpp/server --model "/Users/sengwee.ngui/Library/CloudStorage/OneDrive-TemusPte.Ltd/Documents/projects/SuperAdapters/data/llms/mistral-fwd-john-doe-ckpt-158-200.gguf" --n_gpu_layers 64 --n_ctx 8192 --n_batch 2048 --last_n_tokens_size 4000

python llama_cpp/server --model "/Users/sengwee.ngui/Library/CloudStorage/OneDrive-TemusPte.Ltd/Documents/projects/SuperAdapters/data/llms/mistral-fwd-instruct-v0.2-v0.0.1.gguf" --n_gpu_layers 64 --n_ctx 8192 --n_batch 2048 --last_n_tokens_size 4000 --host 0.0.0.0

python3 llama_cpp/server --model "../data/mistral-fwd-john-doe-ckpt-158-200.gguf" --n_gpu_layers 64 --n_ctx 8192 --n_batch 2048 --last_n_tokens_size 4000