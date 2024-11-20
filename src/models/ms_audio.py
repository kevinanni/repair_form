from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# Create pipeline as a singleton to avoid reloading on each refresh
_inference_pipeline = None

def get_inference_pipeline():
    global _inference_pipeline
    if _inference_pipeline is None:
        _inference_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            model='iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
            model_revision="v2.0.5"
        )
    return _inference_pipeline

inference_pipeline = get_inference_pipeline()
