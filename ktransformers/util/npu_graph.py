import time

import torch
import torch_npu
import sys
import os

from ktransformers.util.utils import USE_NPU_GRAPH
if USE_NPU_GRAPH:
    CAPTURE_PLUGIN_PATH = os.environ.get("CAPTURE_PLUGIN_PATH")
    if CAPTURE_PLUGIN_PATH is None:
        raise RuntimeError("env CAPTURE_PLUGIN_PATH not exist")

    sys.path.append(CAPTURE_PLUGIN_PATH)

    from libgraph_capture import graph_capture_init
    from libgraph_capture import graph_capture_destroy
    from libgraph_capture import graph_capture_begin
    from libgraph_capture import graph_capture_end
    from libgraph_capture import graph_capture_replay
    from libgraph_capture import graph_capture_launch_callback


class NpuGraph:
    def init(self):
        ret = graph_capture_init()
        if ret != 0:
            exit()

    def destroy(self):
        ret = graph_capture_destroy()
        if ret != 0:
            exit()

    def capture_begin(
            self,
            stream,
            capture_error_mode="global"):
        torch.npu.synchronize()
        torch.npu.empty_cache()
        ret = graph_capture_begin(stream, capture_error_mode)
        if ret != 0:
            exit()

    def capture_end(
            self,
            stream):
        ret = graph_capture_end(stream)
        if ret != 0:
            exit()

    def replay(
            self,
            stream):
        ret = graph_capture_replay(stream)
        if ret != 0:
            exit()

    def launch_callback(self, func, data, block, stream):
        graph_capture_launch_callback(func, data, block, stream)


class graph:
    def __init__(
            self,
            npu_graph: NpuGraph,
            pool,
            stream,
            capture_error_mode: str = "global"):
        self.npu_graph = npu_graph
        self.stream = stream.npu_stream

    def __enter__(self):
        self.npu_graph.capture_begin(self.stream)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.npu_graph.capture_end(self.stream)