## Triton ASR Client

A command-line client for the Triton ASR service.


### Installation

To get started with this project, you can clone the repository and install the required packages using `pip`.

   ```
   https://github.com/yuekaizhang/Triton-ASR-Client.git
   cd Triton-ASR-Client
   pip install -r requirements.txt
   ```

### Usage

```
client.py [-h] [--server-addr SERVER_ADDR] [--server-port SERVER_PORT]
                 [--manifest-dir MANIFEST_DIR] [--audio-path AUDIO_PATH]
                 [--model-name {transducer,attention_rescoring,streaming_wenet,infer_pipeline}]
                 [--num-tasks NUM_TASKS] [--log-interval LOG_INTERVAL]
                 [--compute-cer] [--streaming] [--simulate-streaming]
                 [--chunk_size CHUNK_SIZE] [--context CONTEXT]
                 [--encoder_right_context ENCODER_RIGHT_CONTEXT]
                 [--subsampling SUBSAMPLING] [--stats_file STATS_FILE]
```

#### Optional Arguments

* `-h, --help`: show this help message and exit
* `--server-addr SERVER_ADDR`: Address of the server (default: localhost)
* `--server-port SERVER_PORT`: gRPC port of the triton server, default is 8001 (default: 8001)
* `--manifest-dir MANIFEST_DIR`: Path to the manifest dir which includes wav.scp trans.txt files. (default: ./datasets/aishell1_test)
* `--audio-path AUDIO_PATH`: Path to a single audio file. It can't be specified at the same time with --manifest-dir (default: None)
* `--model-name {transducer,attention_rescoring,streaming_wenet,infer_pipeline}`: Triton model_repo module name to request: transducer for k2, attention_rescoring for wenet offline, streaming_wenet for wenet streaming, infer_pipeline for paraformer large offline (default: transducer)
* `--num-tasks NUM_TASKS`: Number of concurrent tasks for sending (default: 50)
* `--log-interval LOG_INTERVAL`: Controls how frequently we print the log. (default: 5)
* `--compute-cer`: True to compute CER, e.g., for Chinese. False to compute WER, e.g., for English words. (default: False)
* `--streaming`: True for streaming ASR. (default: False)
* `--simulate-streaming`: True for strictly simulate streaming ASR. Threads will sleep to simulate the real speaking scene. (default: False)
* `--chunk_size CHUNK_SIZE`: Parameter for streaming ASR, chunk size default is 16 (default: 16)
* `--context CONTEXT`: Subsampling context for wenet (default: -1)
* `--encoder_right_context ENCODER_RIGHT_CONTEXT`: Encoder right context for k2 streaming (default: 2)
* `--subsampling SUBSAMPLING`: Subsampling rate (default: 4)
* `--stats_file STATS_FILE`: Output of [stats analysis](poe://www.poe.com/_api/key_phrase?phrase=stats%20analysis&prompt=Tell%20me%20more%20about%20stats%20analysis.) in human readable format (default: ./stats_summary.txt)



### List of Supported Triton ASR Server
| Model Repo | Description | Source | HuggingFace Link |
| --- | --- | --- | -- |
| [Whisper Large-v2 Onnx](https://github.com/openai/whisper/tree/main) | Offline ASR Onnx FP16 |Openai | yuekai/model_repo_whisper_large_v2  |
| [Conformer Onnx](https://github.com/wenet-e2e/wenet/tree/main/runtime/gpu/model_repo) | Offline ASR Onnx FP16 |Wenet | yuekai/model_repo_conformer_aishell_wenet  |
| [Conformer Tensorrt](https://github.com/wenet-e2e/wenet/tree/main/runtime/gpu/tensorrt/model_repo_stateful_trt) |Streaming ASR Tensorrt FP16 |Wenet |  |
| [Conformer FasterTransformer](https://github.com/wenet-e2e/wenet/tree/main/runtime/gpu/tensorrt_fastertransformer/model_repo_ft) | Offline ASR FasterTransformer FP16|Wenet |  |
| [Conformer CUDA-TLG decoder](https://github.com/wenet-e2e/wenet/tree/main/runtime/gpu/cuda_decoders/model_repo_cuda_decoder) | Offline ASR with CUDA Decoders|Wenet | speechai/model_repo_conformer_aishell_wenet_tlg |
| [Offline Conformer Onnx](https://github.com/k2-fsa/sherpa/tree/master/triton/model_repo_offline) | Offline ASR Onnx FP16 |k2 |wd929/k2_conformer_offline_onnx_model_repo
| [Offline Conformer TensorRT](https://github.com/k2-fsa/sherpa/tree/master/triton/model_repo_offline) | Offline ASR TensorRT FP16 |k2 |wd929/k2_conformer_offline_trt_model_repo |
| [Streaming Conformer Onnx](https://github.com/k2-fsa/sherpa/tree/master/triton/model_repo_streaming) | Streaming ASR Onnx FP16 |k2 | |
| [Zipformer Onnx](https://github.com/k2-fsa/sherpa/tree/master/triton/zipformer/model_repo_offline_bs) | Offline ASR Onnx FP16 with Blank Skip |k2 | |
| [Paraformer Onnx](https://github.com/alibaba-damo-academy/FunASR/tree/main/funasr/runtime/triton_gpu/model_repo_paraformer_large_offline) | Offline ASR FP32 |FunASR | |


