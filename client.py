#!/usr/bin/env python3
# Copyright      2022  Xiaomi Corp.        (authors: Fangjun Kuang)
#                2023  Nvidia              (authors: Yuekai Zhang)
#                2023  Recurrent.ai        (authors: Songtao Shi)
# See LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This script supports to load manifest files in kaldi format and sends it to the server
for decoding, in parallel.

Usage:
# For offline icefall server
python3 client.py \
    --compute-cer  # For Chinese, we use CER to evaluate the model 

# For streaming icefall server
python3 client.py \
    --streaming \
    --compute-cer

# For simulate streaming mode icefall server
python3 client.py \
    --simulate-streaming \
    --compute-cer

# For offline wenet server
python3 client.py \
    --server-addr localhost \
    --compute-cer \
    --model-name attention_rescoring \
    --num-tasks 300 \
    --manifest-dir ./datasets/aishell1_test

# For streaming wenet server
python3 client.py \
    --server-addr localhost \
    --streaming \
    --compute-cer \
    --context 7 \
    --model-name streaming_wenet \
    --num-tasks 300 \
    --manifest-dir ./datasets/aishell1_test

# For simulate streaming mode wenet server
python3 client.py \
    --server-addr localhost \
    --simulate-streaming \
    --compute-cer \
    --context 7 \
    --model-name streaming_wenet \
    --num-tasks 300 \
    --manifest-dir ./datasets/aishell1_test

# For offlien paraformer server
python3 client.py \
    --server-addr localhost \
    --compute-cer \
    --model-name infer_pipeline \
    --num-tasks $num_task \
    --manifest-dir ./datasets/aishell1_test

# For offlien whisper server
python3 client.py \
    --server-addr localhost \
    --model-name whisper \
    --num-tasks $num_task \
    --whisper-prompt "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>" \
    --manifest-dir ./datasets/mini_en

# For offline sensevoice server
python3 client.py \
    --server-addr localhost \
    --server-port 10086 \
    --model-name sensevoice \
    --num-tasks $num_task \
    --manifest-dir ./datasets/mini_zh
"""

import argparse
import asyncio
import json
import math
import os
import re
import time
import types
from pathlib import Path

import numpy as np
import soundfile
import tritonclient
import tritonclient.grpc.aio as grpcclient
from tritonclient.utils import np_to_triton_dtype

from utils import (
    download_and_extract,
    store_transcripts,
    write_error_stats,
    write_triton_stats,
)

DEFAULT_MANIFEST_DIR = "./datasets/aishell1_test"


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--server-addr",
        type=str,
        default="localhost",
        help="Address of the server",
    )

    parser.add_argument(
        "--server-port",
        type=int,
        default=8001,
        help="Grpc port of the triton server, default is 8001",
    )

    parser.add_argument(
        "--manifest-dir",
        type=str,
        default=DEFAULT_MANIFEST_DIR,
        help="Path to the manifest dir which includes wav.scp trans.txt files.",
    )

    parser.add_argument(
        "--audio-path",
        type=str,
        help="Path to a single audio file. It can't be specified at the same time with --manifest-dir",
    )

    parser.add_argument(
        "--whisper-prompt",
        type=str,
        default="<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
        help="e.g. <|startofprev|>My hot words<|startoftranscript|><|en|><|transcribe|><|notimestamps|>, please check https://arxiv.org/pdf/2305.11095.pdf also.",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="transducer",
        choices=[
            "transducer",
            "attention_rescoring",
            "streaming_wenet",
            "infer_pipeline",
            "whisper",
            "sensevoice",
        ],
        help="triton model_repo module name to request: transducer for k2, attention_rescoring for wenet offline, streaming_wenet for wenet streaming, infer_pipeline for paraformer large offline",
    )

    parser.add_argument(
        "--num-tasks",
        type=int,
        default=50,
        help="Number of concurrent tasks for sending",
    )

    parser.add_argument(
        "--log-interval",
        type=int,
        default=5,
        help="Controls how frequently we print the log.",
    )

    parser.add_argument(
        "--compute-cer",
        action="store_true",
        default=False,
        help="""True to compute CER, e.g., for Chinese.
        False to compute WER, e.g., for English words.
        """,
    )

    parser.add_argument(
        "--streaming",
        action="store_true",
        default=False,
        help="""True for streaming ASR.
        """,
    )

    parser.add_argument(
        "--simulate-streaming",
        action="store_true",
        default=False,
        help="""True for strictly simulate streaming ASR.
        Threads will sleep to simulate the real speaking scene.
        """,
    )

    parser.add_argument(
        "--chunk_size",
        type=int,
        required=False,
        default=16,
        help="Parameter for streaming ASR, chunk size default is 16",
    )

    parser.add_argument(
        "--context",
        type=int,
        required=False,
        default=-1,
        help="subsampling context for wenet",
    )

    parser.add_argument(
        "--encoder_right_context",
        type=int,
        required=False,
        default=2,
        help="encoder right context for k2 streaming",
    )

    parser.add_argument(
        "--subsampling",
        type=int,
        required=False,
        default=4,
        help="subsampling rate",
    )

    parser.add_argument(
        "--stats_file",
        type=str,
        required=False,
        default="./stats_summary.txt",
        help="output of stats anaylasis in human readable format",
    )

    return parser.parse_args()


def load_manifests(dir_path):
    dir_path = Path(dir_path)
    wav_scp_path = dir_path / "wav.scp"
    transcripts_path = dir_path / "trans.txt"

    # Check if the files exist, and raise an error if they don't
    if not wav_scp_path.exists():
        raise ValueError(f"{wav_scp_path} does not exist")
    if not transcripts_path.exists():
        raise ValueError(f"{transcripts_path} does not exist")

    # Load the audio file paths into a dictionary
    with open(wav_scp_path, "r") as f:
        wav_dict = {}
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"Invalid line: {line}")
            wav_dict[parts[0]] = parts[1]

    # Load the transcripts into a dictionary
    with open(transcripts_path, "r") as f:
        trans_dict = {}
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"Invalid line: {line}")
            trans_dict[parts[0]] = " ".join(parts[1:])

    # Combine the two dictionaries into a list of dictionaries
    data = []
    for k, v in wav_dict.items():
        assert k in trans_dict, f"Could not find transcript for {k}"
        data.append(
            {"audio_filepath": str(dir_path / v), "text": trans_dict[k], "id": k}
        )

    return data


def split_data(data, k):
    n = len(data)
    if n < k:
        print(
            f"Warning: the length of the input list ({n}) is less than k ({k}). Setting k to {n}."
        )
        k = n

    quotient = n // k
    remainder = n % k

    result = []
    start = 0
    for i in range(k):
        if i < remainder:
            end = start + quotient + 1
        else:
            end = start + quotient

        result.append(data[start:end])
        start = end

    return result


def load_audio(wav_path):
    waveform, sample_rate = soundfile.read(wav_path)
    assert sample_rate == 16000, f"Only support 16k sample rate, but got {sample_rate}"
    return waveform, sample_rate


async def send(
    dps: list,
    name: str,
    triton_client: tritonclient.grpc.aio.InferenceServerClient,
    protocol_client: types.ModuleType,
    log_interval: int,
    compute_cer: bool,
    model_name: str,
    padding_duration: int = 10,
):
    total_duration = 0.0
    results = []
    latency_data = []

    for i, dp in enumerate(dps):
        if i % log_interval == 0:
            print(f"{name}: {i}/{len(dps)}")

        waveform, sample_rate = load_audio(dp["audio_filepath"])
        duration = int(len(waveform) / sample_rate)

        # padding to nearset 10 seconds
        samples = np.zeros(
            (
                1,
                padding_duration * sample_rate * ((duration // padding_duration) + 1),
            ),
            dtype=np.float32,
        )

        samples[0, : len(waveform)] = waveform

        lengths = np.array([[len(waveform)]], dtype=np.int32)

        inputs = [
            protocol_client.InferInput(
                "WAV", samples.shape, np_to_triton_dtype(samples.dtype)
            ),
            protocol_client.InferInput(
                "WAV_LENS", lengths.shape, np_to_triton_dtype(lengths.dtype)
            ),
        ]
        inputs[0].set_data_from_numpy(samples)
        inputs[1].set_data_from_numpy(lengths)
        if model_name == "sensevoice":
            language = np.array([[0]], dtype=np.int32)
            text_norm = np.array([[15]], dtype=np.int32)
            inputs.append(
                protocol_client.InferInput(
                    "LANGUAGE",
                    language.shape,
                    np_to_triton_dtype(np.int32),
                )
            )
            inputs.append(
                protocol_client.InferInput(
                    "TEXT_NORM",
                    text_norm.shape,
                    np_to_triton_dtype(np.int32),
                )
            )
            inputs[2].set_data_from_numpy(language)
            inputs[3].set_data_from_numpy(text_norm)
        outputs = [protocol_client.InferRequestedOutput("TRANSCRIPTS")]
        sequence_id = 10086 + i
        start = time.time()
        response = await triton_client.infer(
            model_name, inputs, request_id=str(sequence_id), outputs=outputs
        )

        decoding_results = response.as_numpy("TRANSCRIPTS")[0]
        if type(decoding_results) == np.ndarray:
            decoding_results = b" ".join(decoding_results).decode("utf-8")
        else:
            # For wenet
            decoding_results = decoding_results.decode("utf-8")
        # remove special tokens in sensevoice results e.g. <|zh|><|NEUTRAL|><|Speech|><|woitn|>大学生利用漏洞免费吃肯德基祸刑
        # <|*|>, using re
        decoding_results = re.sub(r"<\|.*?\|>", "", decoding_results)
        end = time.time() - start
        latency_data.append((end, duration))
        total_duration += duration

        if compute_cer:
            ref = dp["text"].split()
            hyp = decoding_results.split()
            ref = list("".join(ref))
            hyp = list("".join(hyp))
            results.append((dp["id"], ref, hyp))
        else:
            results.append(
                (
                    dp["id"],
                    dp["text"].split(),
                    decoding_results.split(),
                )
            )

    return total_duration, results, latency_data


async def send_streaming(
    dps: list,
    name: str,
    triton_client: tritonclient.grpc.aio.InferenceServerClient,
    protocol_client: types.ModuleType,
    log_interval: int,
    compute_cer: bool,
    model_name: str,
    first_chunk_in_secs: float,
    other_chunk_in_secs: float,
    task_index: int,
    simulate_mode: bool = False,
):
    total_duration = 0.0
    results = []
    latency_data = []

    for i, dp in enumerate(dps):
        if i % log_interval == 0:
            print(f"{name}: {i}/{len(dps)}")

        waveform, sample_rate = load_audio(dp["audio_filepath"])
        duration = int(len(waveform) / sample_rate)

        wav_segs = []

        j = 0
        while j < len(waveform):
            if j == 0:
                stride = int(first_chunk_in_secs * sample_rate)
                wav_segs.append(waveform[j : j + stride])
            else:
                stride = int(other_chunk_in_secs * sample_rate)
                wav_segs.append(waveform[j : j + stride])
            j += len(wav_segs[-1])

        sequence_id = task_index + 10086

        for idx, seg in enumerate(wav_segs):
            chunk_len = len(seg)

            if simulate_mode:
                await asyncio.sleep(chunk_len / sample_rate)

            chunk_start = time.time()
            if idx == 0:
                chunk_samples = int(first_chunk_in_secs * sample_rate)
                expect_input = np.zeros((1, chunk_samples), dtype=np.float32)
            else:
                chunk_samples = int(other_chunk_in_secs * sample_rate)
                expect_input = np.zeros((1, chunk_samples), dtype=np.float32)

            expect_input[0][0:chunk_len] = seg
            input0_data = expect_input
            input1_data = np.array([[chunk_len]], dtype=np.int32)

            inputs = [
                protocol_client.InferInput(
                    "WAV",
                    input0_data.shape,
                    np_to_triton_dtype(input0_data.dtype),
                ),
                protocol_client.InferInput(
                    "WAV_LENS",
                    input1_data.shape,
                    np_to_triton_dtype(input1_data.dtype),
                ),
            ]

            inputs[0].set_data_from_numpy(input0_data)
            inputs[1].set_data_from_numpy(input1_data)

            outputs = [protocol_client.InferRequestedOutput("TRANSCRIPTS")]
            end = False
            if idx == len(wav_segs) - 1:
                end = True

            response = await triton_client.infer(
                model_name,
                inputs,
                outputs=outputs,
                sequence_id=sequence_id,
                sequence_start=idx == 0,
                sequence_end=end,
            )
            idx += 1

            decoding_results = response.as_numpy("TRANSCRIPTS")
            if type(decoding_results) == np.ndarray:
                decoding_results = b" ".join(decoding_results).decode("utf-8")
            else:
                # For wenet
                decoding_results = response.as_numpy("TRANSCRIPTS")[0].decode("utf-8")
            chunk_end = time.time() - chunk_start
            latency_data.append((chunk_end, chunk_len / sample_rate))

        total_duration += duration

        if compute_cer:
            ref = dp["text"].split()
            hyp = decoding_results.split()
            ref = list("".join(ref))
            hyp = list("".join(hyp))
            results.append((dp["id"], ref, hyp))
        else:
            results.append(
                (
                    dp["id"],
                    dp["text"].split(),
                    decoding_results.split(),
                )
            )  # noqa

    return total_duration, results, latency_data


async def send_whisper(
    dps: list,
    name: str,
    triton_client: tritonclient.grpc.aio.InferenceServerClient,
    protocol_client: types.ModuleType,
    log_interval: int,
    compute_cer: bool,
    model_name: str,
    padding_duration: int = 10,
    whisper_prompt: str = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
):
    total_duration = 0.0
    results = []
    latency_data = []
    task_id = int(name[5:])
    for i, dp in enumerate(dps):
        if i % log_interval == 0:
            print(f"{name}: {i}/{len(dps)}")

        waveform, sample_rate = load_audio(dp["audio_filepath"])
        duration = int(len(waveform) / sample_rate)

        # padding to nearset 10 seconds
        samples = np.zeros(
            (
                1,
                padding_duration * sample_rate * ((duration // padding_duration) + 1),
            ),
            dtype=np.float32,
        )

        samples[0, : len(waveform)] = waveform

        lengths = np.array([[len(waveform)]], dtype=np.int32)

        inputs = [
            protocol_client.InferInput(
                "WAV", samples.shape, np_to_triton_dtype(samples.dtype)
            ),
            protocol_client.InferInput("TEXT_PREFIX", [1, 1], "BYTES"),
        ]
        inputs[0].set_data_from_numpy(samples)

        input_data_numpy = np.array([whisper_prompt], dtype=object)
        input_data_numpy = input_data_numpy.reshape((1, 1))
        inputs[1].set_data_from_numpy(input_data_numpy)

        outputs = [protocol_client.InferRequestedOutput("TRANSCRIPTS")]
        sequence_id = 100000000 + i + task_id * 10
        start = time.time()
        response = await triton_client.infer(
            model_name, inputs, request_id=str(sequence_id), outputs=outputs
        )

        decoding_results = response.as_numpy("TRANSCRIPTS")[0]
        if type(decoding_results) == np.ndarray:
            decoding_results = b" ".join(decoding_results).decode("utf-8")
        else:
            # For wenet
            decoding_results = decoding_results.decode("utf-8")
        end = time.time() - start
        latency_data.append((end, duration))
        total_duration += duration

        if compute_cer:
            ref = dp["text"].split()
            hyp = decoding_results.split()
            ref = list("".join(ref))
            hyp = list("".join(hyp))
            results.append((dp["id"], ref, hyp))
        else:
            results.append(
                (
                    dp["id"],
                    dp["text"].split(),
                    decoding_results.split(),
                )
            )
        print(results[-1])

    return total_duration, results, latency_data


async def main():
    args = get_args()
    if args.audio_path:
        args.num_tasks = 1
        args.log_interval = 1
        dps_list = [
            [
                {
                    "audio_filepath": args.audio_path,
                    "text": "foo",
                    "id": 0,
                }
            ]
        ]
    else:
        if not any(Path(args.manifest_dir).rglob("*.wav")):
            if args.manifest_dir == DEFAULT_MANIFEST_DIR:
                download_and_extract(args.manifest_dir)
            raise ValueError(
                f"manifest_dir {args.manifest_dir} should contain wav files"
            )
        dps = load_manifests(args.manifest_dir)
        dps_list = split_data(dps, args.num_tasks)
        args.num_tasks = min(args.num_tasks, len(dps_list))

    url = f"{args.server_addr}:{args.server_port}"

    triton_client = grpcclient.InferenceServerClient(url=url, verbose=False)
    protocol_client = grpcclient

    if args.streaming or args.simulate_streaming:
        frame_shift_ms = 10
        frame_length_ms = 25
        add_frames = math.ceil((frame_length_ms - frame_shift_ms) / frame_shift_ms)
        # decode_window_length: input sequence length of streaming encoder
        if args.context > 0:
            # decode window length calculation for wenet
            decode_window_length = (
                args.chunk_size - 1
            ) * args.subsampling + args.context
        else:
            # decode window length calculation for icefall
            decode_window_length = (
                args.chunk_size + 2 + args.encoder_right_context
            ) * args.subsampling + 3

        first_chunk_ms = (decode_window_length + add_frames) * frame_shift_ms

    tasks = []
    start_time = time.time()
    for i in range(args.num_tasks):
        if args.streaming:
            assert not args.simulate_streaming
            task = asyncio.create_task(
                send_streaming(
                    dps=dps_list[i],
                    name=f"task-{i}",
                    triton_client=triton_client,
                    protocol_client=protocol_client,
                    log_interval=args.log_interval,
                    compute_cer=args.compute_cer,
                    model_name=args.model_name,
                    first_chunk_in_secs=first_chunk_ms / 1000,
                    other_chunk_in_secs=args.chunk_size
                    * args.subsampling
                    * frame_shift_ms
                    / 1000,
                    task_index=i,
                )
            )
        elif args.simulate_streaming:
            task = asyncio.create_task(
                send_streaming(
                    dps=dps_list[i],
                    name=f"task-{i}",
                    triton_client=triton_client,
                    protocol_client=protocol_client,
                    log_interval=args.log_interval,
                    compute_cer=args.compute_cer,
                    model_name=args.model_name,
                    first_chunk_in_secs=first_chunk_ms / 1000,
                    other_chunk_in_secs=args.chunk_size
                    * args.subsampling
                    * frame_shift_ms
                    / 1000,
                    task_index=i,
                    simulate_mode=True,
                )
            )
        else:
            if args.model_name == "whisper":
                task = asyncio.create_task(
                    send_whisper(
                        dps=dps_list[i],
                        name=f"task-{i}",
                        triton_client=triton_client,
                        protocol_client=protocol_client,
                        log_interval=args.log_interval,
                        compute_cer=args.compute_cer,
                        model_name=args.model_name,
                        whisper_prompt=args.whisper_prompt,
                    )
                )
            else:
                task = asyncio.create_task(
                    send(
                        dps=dps_list[i],
                        name=f"task-{i}",
                        triton_client=triton_client,
                        protocol_client=protocol_client,
                        log_interval=args.log_interval,
                        compute_cer=args.compute_cer,
                        model_name=args.model_name,
                    )
                )
        tasks.append(task)

    ans_list = await asyncio.gather(*tasks)

    end_time = time.time()
    elapsed = end_time - start_time

    results = []
    total_duration = 0.0
    latency_data = []
    for ans in ans_list:
        total_duration += ans[0]
        results += ans[1]
        latency_data += ans[2]

    rtf = elapsed / total_duration

    s = f"RTF: {rtf:.4f}\n"
    s += f"total_duration: {total_duration:.3f} seconds\n"
    s += f"({total_duration/3600:.2f} hours)\n"
    s += f"processing time: {elapsed:.3f} seconds " f"({elapsed/3600:.2f} hours)\n"

    latency_list = [chunk_end for (chunk_end, chunk_duration) in latency_data]
    latency_ms = sum(latency_list) / float(len(latency_list)) * 1000.0
    latency_variance = np.var(latency_list, dtype=np.float64) * 1000.0
    s += f"latency_variance: {latency_variance:.2f}\n"
    s += f"latency_50_percentile_ms: {np.percentile(latency_list, 50) * 1000.0:.2f}\n"
    s += f"latency_90_percentile_ms: {np.percentile(latency_list, 90) * 1000.0:.2f}\n"
    s += f"latency_95_percentile_ms: {np.percentile(latency_list, 95) * 1000.0:.2f}\n"
    s += f"latency_99_percentile_ms: {np.percentile(latency_list, 99) * 1000.0:.2f}\n"
    s += f"average_latency_ms: {latency_ms:.2f}\n"

    print(s)

    with open("rtf.txt", "w") as f:
        f.write(s)

    name = Path(args.manifest_dir).stem.split(".")[0]
    results = sorted(results)
    store_transcripts(filename=f"recogs-{name}.txt", texts=results)

    with open(f"errs-{name}.txt", "w") as f:
        write_error_stats(f, "test-set", results, enable_log=True)

    with open(f"errs-{name}.txt", "r") as f:
        print(f.readline())  # WER
        print(f.readline())  # Detailed errors

    if args.stats_file:
        stats = await triton_client.get_inference_statistics(
            model_name="", as_json=True
        )
        write_triton_stats(stats, args.stats_file)


if __name__ == "__main__":
    asyncio.run(main())
