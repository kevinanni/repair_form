from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# Create pipelines as singletons to avoid reloading on each refresh
_asr_pipeline = None
_similarity_pipeline = None

def get_asr_pipeline():
    """Get speech recognition pipeline singleton"""
    global _asr_pipeline
    if _asr_pipeline is None:
        _asr_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            model='iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
            model_revision="v2.0.5"
        )
    return _asr_pipeline

def get_similarity_pipeline():
    """Get sentence similarity pipeline singleton"""
    global _similarity_pipeline
    if _similarity_pipeline is None:
        _similarity_pipeline = pipeline(
            task=Tasks.sentence_similarity,
            model='damo/nlp_masts_sentence-similarity_clue_chinese-large',
            model_revision='v1.0.0'
        )
    return _similarity_pipeline

# Initialize pipeline instances
asr_pipeline = get_asr_pipeline()
similarity_pipeline = get_similarity_pipeline()