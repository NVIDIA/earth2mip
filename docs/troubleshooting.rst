Troubleshooting
===============


>  UserWarning: Specified provider 'CUDAExecutionProvider' is not in available provider names.Available providers: 'CPUExecutionProvider'

If you see this warning, the inference will fail on gpus. Fix the onnxruntime installation::

    pip uninstall onnxruntime onnxruntime-gpu
    pip install onnxruntime-gpu
