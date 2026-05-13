"""
ASR 模块异常类型

不重新包装 ASRTranscription——直接复用 qwen_asr 的原生类型。
"""


class NotLoadedError(RuntimeError):
    """transcribe() 被调用但模型未加载。"""


class TranscriberOOMError(RuntimeError):
    """推理时 CUDA OOM，附段长和显存诊断信息。"""
