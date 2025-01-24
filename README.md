# Hibiki(ひびき)

<img src="assets/hibiki.png" alt="hibiki" width="350"/>

### Usage: 
```shell
git clone https://github.com/xutianyi1999/hibiki.git
cd hibiki
cargo build --release --features cuda
hibiki -b 0.0.0.0:30000 -m /media/nvme/models/Meta-Llama-3-70B-Instruct-Q6_K.gguf/Meta-Llama-3-70B-Instruct-Q6_K-00001-of-00002.gguf -d /media/nvme/models/Meta-Llama-3-8B-Instruct.Q2_K.gguf -t llama3 --model-name llama3
```
