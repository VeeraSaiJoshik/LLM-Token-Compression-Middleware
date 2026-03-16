import math
from typing import List
import click
from sentence_transformers import SentenceTransformer, util
import asyncio
import matplotlib.pyplot as plt
from tqdm import tqdm 
import numpy as np
import time
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

from rich.console import Console
from rich.table import Table

async def tag_pos(prompt: str):
    tokenized_words = word_tokenize(prompt)
    tagged_words = pos_tag(tokenized_words)

    return tagged_words

async def chunk_word_optimizations(tagged_words: tuple[str, str]):
    NOUNS = ["NN", "NNS", "NNP", "NNPS"]
    ADJECTIVES = ["JJ", "JJR", "JJS"]
    EXTRANEOUS = ["POS"]

    proper_nouns = []
    words = []
    is_proper_noun = False

    buffer = []
    i = 0
    for word, word_type in tagged_words:
        if word_type in NOUNS or word_type in ADJECTIVES or word_type in EXTRANEOUS: 
            if word_type in ["NNP", "NNPS"]: is_proper_noun = True
            buffer.append(word)
        else : 
            if buffer != []:
                if not is_proper_noun :
                    words.append(" ".join(buffer))
                    i += 1
                else : 
                    proper_nouns.append((
                        proper_nouns[-1][0] + 0.001 
                            if len(proper_nouns) != 0 and int(proper_nouns[-1][0]) == i -1 else i - 1 + 0.001, 
                        " ".join(buffer)
                    ))
                    is_proper_noun = False
                buffer = []
            else : 
                words.append(word)
                i += 1
    if buffer != []:
        if not is_proper_noun :
            words.append(" ".join(buffer))
            i += 1
        else : 
            proper_nouns.append((
                proper_nouns[-1][0] + 0.001 
                    if len(proper_nouns) != 0 and int(proper_nouns[-1][0]) == i -1 else i - 1 + 0.001, 
                " ".join(buffer)
            ))
        buffer = []

    chunked_words_list = []

    for i in range(len(words)):
        before = " ".join(words[0: i])
        after = " ".join(words[i + 1:])
        if before and after:
            chunked_words_list.append(before + " " + after)
        elif before:
            chunked_words_list.append(before)
        elif after:
            chunked_words_list.append(after)
    
    print("here are the proper nouns", proper_nouns)

    return chunked_words_list, proper_nouns, words

async def get_sentence_transformation(prompt: str, model: SentenceTransformer):
    return await asyncio.to_thread(model.encode, prompt, convert_to_tensor=True)

def remove_outliers(data, threshold=3.5):
    if not data:
        return data

    median = sorted(data)[len(data) // 2]
    abs_devs = [abs(x - median) for x in data]
    mad = sorted(abs_devs)[len(abs_devs) // 2]

    if mad == 0:
        return data

    def modified_z_score(x):
        return 0.6745 * (x - median) / mad

    return [x for x in data if abs(modified_z_score(x)) <= threshold]

def get_outliers(data, threshold=3.5):
    if not data:
        return data

    median = sorted(data)[len(data) // 2]
    abs_devs = [abs(x - median) for x in data]
    mad = sorted(abs_devs)[len(abs_devs) // 2]

    if mad == 0:
        return data

    def modified_z_score(x):
        return 0.6745 * (x - median) / mad

    return [x for x in data if abs(modified_z_score(x)) > threshold]

def find_maximas(data):
    if not data or len(data) < 3:
        return []
    
    maximas = []
    for i in range(1, len(data) - 1):
        if data[i] > data[i - 1] and data[i] > data[i + 1]:
            maximas.append(data[i])

    return maximas

async def near_neighbor_vector_optimization(
    prompt: str,
    sigma: float = 2.0,
) -> List[float]:
    model: SentenceTransformer = SentenceTransformer("all-MiniLM-L6-v2")
    words = prompt.split()
    n = len(words)

    tasks = [get_sentence_transformation(word, model) for word in words]
    encoding_results = await asyncio.gather(*tasks)  # length == n

    scores: List[float] = []

    for i in range(n):
        weighted_neighborhood_score = 0.0
        weight_sum = 0.0
        for j in range(n):
            if j == i:
                continue

            distance = abs(j - i)
            weight = math.exp(-(distance ** 2) / (2 * sigma ** 2))
            sim = util.cos_sim(encoding_results[i], encoding_results[j]).item()

            weighted_neighborhood_score += weight * sim
            weight_sum += weight

        if weight_sum > 0:
            weighted_neighborhood_score /= weight_sum

        scores.append(weighted_neighborhood_score)

    return scores


async def vector_prompt_optimizer(prompt: str, model: SentenceTransformer, log: bool = False):
    pos_tag = await tag_pos(prompt)
    chunked, proper_nouns, words = await chunk_word_optimizations(pos_tag)

    tasks = [get_sentence_transformation(prompt, model)] + [
        get_sentence_transformation(source, model) for source in chunked
    ]
    encoding_results = await asyncio.gather(*tasks)
    
    similarity = []
    for i in tqdm(range(1, len(encoding_results))):
        sim = util.pytorch_cos_sim(encoding_results[0], encoding_results[i])
        similarity.append(sim.item())
    
    # Create word-similarity pairs to track which words correspond to each similarity score
    word_sim_pairs = [(words[i], similarity[i]) for i in range(len(similarity))]
    
    # Filter outliers while keeping track of words
    filtered_pairs = [(word, sim) for word, sim in word_sim_pairs if sim in remove_outliers(similarity)]
    if log: print(filtered_pairs)
    filtered_pairs.sort(key=lambda x: x[1])  # Sort by similarity score
    
    sim_norm = [pair[1] for pair in filtered_pairs]
    corresponding_words = [pair[0] for pair in filtered_pairs]
    
    word_removal_window = len(similarity) - len(sim_norm)
    print("initial world removal", word_removal_window, proper_nouns)
    sim_der = [sim_norm[i + 1] - sim_norm[i] for i in range(len(sim_norm) - 1)]
    sim_der_outliers = get_outliers(sim_der)
    if len(sim_der_outliers) == 0 :
        sim_der_outliers = find_maximas(sim_der)

    if log:
        print(sim_der_outliers)
        plt.figure()
        plt.plot(list(range(len(sim_der))), sim_der)
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.title("Neighbor Weights")
        plt.grid(True)

        plt.figure()
        plt.plot(list(range(len(sim_norm))), sim_norm)
        plt.xticks(range(len(corresponding_words)), corresponding_words, rotation=45, ha='right')
        plt.xlabel("Words")
        plt.ylabel("Similarity Score")
        plt.title("Word Embedding Ranking")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    outlier_indexes = list(filter( lambda i: i >= 2, [sim_der.index(outlier) + 1 for outlier in sim_der_outliers]))
    if len(outlier_indexes) != 0 :
        if log: print(outlier_indexes, "index")
        word_removal_window = len(words) - (outlier_indexes[0] + 1 + word_removal_window)
    
    existing_indexes = sorted([(i, similarity[i]) for i in range(len(similarity))], key=lambda x: x[1])[0:len(similarity) - word_removal_window]
    
    optimized_words = []
    for i, _ in sorted(existing_indexes + proper_nouns, key=lambda x: x[0]) :
        optimized_words.append(words[i] if type(_) == float else _)
    print(sorted(existing_indexes + proper_nouns, key=lambda x: x[0]), optimized_words)
    
    final_string = " ".join(optimized_words)
    if log: print(1 - len(final_string)/len(prompt))
    print(final_string)

    return final_string

def vector_prompt_optimizer_sync(prompt: str, model: SentenceTransformer, log: bool = False):
    return asyncio.run(vector_prompt_optimizer(prompt, model, log))

console = Console()

@click.group()
def cli():
    """LLM Cost Optimization Testing Tool"""
    pass

@cli.command()
@click.argument('prompt')
def single_prompt(prompt):
    model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L12-v2")
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('averaged_perceptron_tagger_eng')
    
    start = time.time()
    
    asyncio.run(
        vector_prompt_optimizer(prompt, model, True)
    )
    end = time.time()
    print("Total time", end-start)

@cli.command()
def time_graph():
    model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L12-v2")
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('averaged_perceptron_tagger_eng')
    
    times = []
    sample_prompts = [
        "Explain the concept of recursion in simple terms.",
        "Generate three creative names for a futuristic electric scooter brand.",
        "Summarize the main themes of any popular fantasy novel without giving spoilers.",
        "Write a friendly reminder email asking a coworker to share the project files before noon tomorrow.",
        "Describe how photosynthesis works, focusing on why it is essential for sustaining life on Earth.",
        "Create a short motivational message for students preparing for their final exams next week.",
        "Give a detailed comparison between solar power and wind power, including cost, efficiency, and environmental impact considerations.",
        "Provide step-by-step instructions for organizing a small community event, covering planning, promotion, budgeting, and volunteer coordination.",
        "Write a brief story about a traveler who discovers an unexpected message hidden inside an ancient artifact during a desert expedition.",
        "Explain how machine learning models are trained, including data preprocessing, feature selection, training algorithms, and evaluating performance metrics."
    ]
    results_prompts = []
    for prompt in tqdm(sample_prompts) :
        start = time.time()
        results_prompts.append(asyncio.run(vector_prompt_optimizer(prompt, model, False)))
        end = time.time()

        times.append(end - start)

    table = Table(title="prompt compression")
    table.add_column("Before")
    table.add_column("After")

    for before, after in zip(sample_prompts, results_prompts):
        table.add_row(before, after)
    
    console.print(table)
    
    plt.figure()
    plt.plot(
        list([len(prompt.split(" ")) for prompt in sample_prompts]),
        times
    )
    plt.xlabel("Prompt Word Length")
    plt.ylabel("Processing Time")
    plt.title("Time Graph")

    plt.figure()
    plt.plot(
        list([len(prompt.split(" ")) for prompt in sample_prompts]),
        list(sorted([(len(after) - len(before))/(len(before))for before, after in zip(sample_prompts, results_prompts)]))
    )
    plt.xlabel("Prompt Word Length")
    plt.ylabel("Optimization")
    plt.title("Time Graph")
    plt.show()
    print("Total time", end-start)

if __name__ == "__main__":
    cli()