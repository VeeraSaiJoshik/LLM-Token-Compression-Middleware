from enum import Enum

from sentence_transformers import SentenceTransformer
from optimizers.prompt_compressor import compress_prompt_conservative, compress_prompt_aggressive
from optimizers.toon_converter import convert_prompt_to_toon
from optimizers.whitespace_optimizer import optimize_whitespace
from optimizers.vector_prompt_optimizer import vector_prompt_optimizer, vector_prompt_optimizer_sync
from time import perf_counter

class CompressionAlgorithms(Enum): 
    vector = (1, 'v_comp_v1')
    compression_conservative = (2, 'compression_conservative')
    compression_aggresive = (3, "compression_aggresive")
    toon = (4, "toon")
    whitespace = (5, "whitespace")

def runCompressionAlgorithm(prompt, compression_flags: list[str]):
    compression_flags: list[CompressionAlgorithms] = [
        alg for alg in CompressionAlgorithms
        if alg.value[1] in compression_flags
    ]
    compression_time = []

    model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L12-v2")

    for compression in sorted(compression_flags, key=lambda val: val.value[0]):
        start_time = perf_counter()
        if compression == CompressionAlgorithms.compression_aggresive: 
            prompt = compress_prompt_aggressive(prompt)
        elif compression == CompressionAlgorithms.compression_conservative:
            prompt = compress_prompt_conservative(prompt)
        elif compression == CompressionAlgorithms.toon:
            prompt = convert_prompt_to_toon(prompt)
        elif compression == CompressionAlgorithms.whitespace:
            prompt = optimize_whitespace(prompt)
        else:
            prompt = vector_prompt_optimizer_sync(prompt, model)
        end_time = perf_counter()

        compression_time.append((compression.value[1], str(end_time - start_time)))
    
    return prompt, compression_time