# 部署

## 物理机安装

部署满血版DeepseekV3，需要机器物理内存能够存放下全部路由专家的权重，约400GB。

目前支持的NPU型号：**800I A2**。

在技术人员的支持下完成硬件安装。

## 系统安装

根据网页[昇腾兼容性查询助手](https://www.hiascend.com/hardware/compatibility)查询，选用系统Ubuntu 22.04 for aarch64，内核5.15.0-25-generic，并禁止系统自动更新。系统镜像获取链接：[ubuntu-old-releases](https://mirrors.aliyun.com/oldubuntu-releases/releases/22.04)。

## HDK安装

选择[Ascend HDK 25.0.RC1](https://www.hiascend.com/hardware/firmware-drivers/community?product=4&model=32&cann=8.1.RC1.beta1&driver=Ascend+HDK+25.0.RC1)进行安装，安装方式参考[昇腾社区HDK安装指导](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1beta1/softwareinst/instg/instg_0005.html?Mode=PmIns&InstallType=local&OS=Ubuntu&Software=cannToolKit)。


## Conda部署

建议按照最新[Installation Guide - kTransformers](https://kvcache-ai.github.io/ktransformers/en/install.html)部署开发环境，此处注意Python版本要求3.11（其他版本未验证），arm平台不需要安装cpufeature包。

安装conda/miniconda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
bash ~/Miniconda3-latest-Linux-aarch64.sh
```

部署Python环境：

```bash
conda create -n py311 python=3.11
conda activate py311
conda install -c conda-forge libstdcxx-ng  # 安装`GLIBCXX-3.4.32`
pip3 install numpy==1.26.4  # 适配torch/torch_npu
pip3 install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cpu
pip3 install packaging ninja transformers==4.43.2 fire protobuf attrs decorator cloudpickle ml-dtypes scipy tornado absl-py psutil
#pip3 install cpufeature  # only for x86
```

## CANN安装

选择[CANN 8.1.RC1.beta1](https://www.hiascend.com/developer/download/community/result?from=firmware&product=4&model=32&cann=8.1.RC1.beta1)进行安装，安装方式参考[昇腾社区CANN安装指导](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1beta1/softwareinst/instg/instg_0007.html?Mode=PmIns&InstallType=local&OS=Ubuntu&Software=cannToolKit)。

需要安装ToolKit，Kernel和NNAL。

## torch_npu(op-plugin)安装

获取最新的仓库代码：[op-plugin Gitee](https://gitee.com/ascend/op-plugin)

由于涉及新增算子，公网pypi内提供的torch_npu暂时无法直接使用，需要使用适配过的op-plugin来编译生成所需的torch_npu包，目前还无法从公网获取。  # TODO

在访问github和gitee的网络通畅时，执行下述代码完成编译和安装torch_npu：

```bash
cd op-plugin
source /usr/local/Ascend/ascend-toolkit/set_env.sh  # 以实际CANN安装路径为准
source /usr/local/Ascend/nnal/atb/set_env.sh  # 以实际NNAL安装路径为准
bash install.sh --python=3.11 --pytorch=v2.3.1-7.0.0  # 实际生成的torch_npu whl安装文件在{op-plugin项目地址}/dist中
```

## 权重准备

目前，为了满足性能和精度的要求，我们需要准备两份权重，并使用提供的权重合并脚本对权重进行合并，最终只会使用合并后的权重。

Q4权重：[DeepSeek-R1-Q4_K_M](https://modelscope.cn/models/unsloth/DeepSeek-R1-GGUF/files)

W8A8权重：[DeepSeek-R1-W8A8](https://modelers.cn/models/MindSpore-Lab/DeepSeek-R1-W8A8/tree/main)

使用[merge_safetensor_gguf.py](../../merge_tensors/merge_safetensor_gguf.py)来合并Q4和W8A8权重：

```bash
python merge_safetensor_gguf.py --safetensor_path /mnt/weights/DeepSeek-R1-Q4_K_M --gguf_path /mnt/weights/DeepSeek-R1-W8A8 --output_path /mnt/weights/DeepSeek-R1-q4km-w8a8 --safetensors_format w8a8
```

## 图下沉部署

图下沉所需的二进制文件随仓库给出：[ktransformers/util/npu_graph_so](../../ktransformers/util/npu_graph_so)。

部署图下沉功能，需要做相关文件替换，以arm平台为例：

```bash
mv /usr/local/Ascend/ascend-toolkit/latest/lib64/libruntime.so /usr/local/Ascend/ascend-toolkit/latest/lib64/libruntime.so.bak
cp ktransformers/util/npu_graph_so/arm/libruntime.so /usr/local/Ascend/ascend-toolkit/latest/lib64/libruntime.so
```

开启图下沉功能，需要添加如下环境变量：

```bash
export CAPTURE_PLUGIN_PATH=ktransformers/util/npu_graph_so/arm
export TASK_QUEUE_ENABLE=0  # 保证算子下发顺序有序
```


## kTransformers部署

将项目文件部署到机器上：

- 对于arm平台，注释掉`./requirements-local_chat.txt`中的`cpufeature`。
- 对于arm平台，做如下替换：
  ```bash
  cp ./for_arm/CMakeLists.txt ./csrc/ktransformers_ext/CMakeLists.txt
  cp ./for_arm/iqk_mul_mat.inc ./third_party/llamafile/iqk_mul_mat.inc
  cp ./for_arm/sgemm.cpp ./third_party/llamafile/sgemm.cpp
  cp ./for_arm/tinyblas_cpu_sgemm.inc ./third_party/llamafile/tinyblas_cpu_sgemm.inc
  cp ./for_arm/setup.py ./setup.py
  ```
- 执行`source /usr/local/Ascend/ascend-toolkit/set_env.sh`（以实际CANN-TOOLKIT安装路径为准）。
- 执行`apt install cmake libhwloc-dev pkg-config`安装依赖。
- 执行`bash install.sh`，等待安装完成。

此处给出示例local_chat的启动脚本（由于使用了相对路径，需将该脚本放至项目的根路径下）：

```bash
#!/bin/bash
export CAPTURE_PLUGIN_PATH=ktransformers/util/npu_graph_so/arm
export USE_MERGE=0
export INF_NAN_MODE_FORCE_DISABLE=1
export TASK_QUEUE_ENABLE=0
#export PROF_DECODE=1
#export PROF_PREFILL=1

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

torchrun --nproc_per_node 1 \
         --master_port 25565 \
         -m ktransformers.local_chat \
         --cpu_infer 20 \
         --model_path /mnt/weights/DeepSeek-R1-q4km-w8a8 \
         --gguf_path /mnt/weights/DeepSeek-R1-q4km-w8a8 \
         --optimize_config_path ./ktransformers/optimize/optimize_rules/DeepSeek-V3-Chat-800IA2-npu.yaml \
         --use_cuda_graph True \
         --max_new_tokens 500 \
         --tp 1
```

相关参数说明：

- `--model_path`：kTransformers原生参数，str，此处用来指定合并后的模型文件路径
- `--gguf_path`：kTransformers原生参数，str，此处用来指定合并后的模型文件路径
- `--cpu_infer`：kTransformers原生参数，int，用来控制CPU侧实际worker线程数，非必选
- `--optimize_config_path`：kTransformers原生参数，str，用来指定所用的模型优化配置文件，需要注意相对路径的使用，此处为**必选**
- `--use_cuda_graph`：kTransformers原生参数，bool，为True表示开启图下沉，为False表示关闭图下沉，非必选
- `--max_new_tokens`：kTransformers原生参数，int，当统计到输出的tokens数量达到该值时，会直接中止输出，非必选
- `--tp`：新增参数，int，用于开启tensor model parallel功能，目前local_chat只支持tp大小与ws大小相同（不支持local_chat使用多dp），非必选


# 其他问题

## 可能存在的其他依赖问题

ImportError: libhccl.so: cannot open shared object file: No such file or directory

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh  # 以实际CANN安装路径为准
```

ImportError: libascend_hal.so: cannot open shared object file: No such file or directory

```bash
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH  # 以实际Driver安装路径为准
```