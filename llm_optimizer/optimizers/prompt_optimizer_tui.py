import time
import nltk
from sentence_transformers import SentenceTransformer
import asyncio
from llm_optimizer.optimizers import vector_prompt_optimizer
import click
from rich.console import Console
from rich.table import Table
from matplotlib import pyplot as plt
from tqdm import tqdm

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