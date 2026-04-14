import math
from typing import List
from sentence_transformers import SentenceTransformer, util
import asyncio
import matplotlib.pyplot as plt
from tqdm import tqdm 
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag

async def tag_pos(prompt: str):
    tokenized_words = word_tokenize(prompt)
    tagged_words = pos_tag(tokenized_words)

    return tagged_words

async def chunk_word_optimizations(prompt: str):
    chunked_sentences = sent_tokenize(prompt)

    for sentence in chunked_sentences:
        tagged_words = await tag_pos(sentence)
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
        
        print(words)
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
        print(chunked_words_list)
        #return chunked_words_list, proper_nouns, words

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
    model: SentenceTransformer,
    sigma: float = 2.0,
) -> List[float]:
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
    chunked, proper_nouns, words = await chunk_word_optimizations(prompt)

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

dummy_text = """Overview

AARP is the nation's largest nonprofit, nonpartisan organization dedicated to empowering people 50 and older to choose how they live as they age. With a nationwide presence, AARP strengthens communities and advocates for what matters most to the more than 100 million Americans 50-plus and their families health and financial security, and personal fulfillment. AARP also works for individuals in the marketplace by sparking new solutions and allowing carefully chosen, high-quality products and services to carry the AARP name. As a trusted source for news and information, AARP produces the nation's largest-circulation publications, AARP The Magazine and the AARP Bulletin.

Ready to dive into the world of video production and make a meaningful impact? As an Editorial and Video Production Intern at AARP Studios, you’ll support broadcast and video initiatives while helping shape storytelling that brings people together and drives engagement across our channels. You’ll gain hands-on experience in video capture, writing, editing, and content strategy, all while learning how AARP leverages its diverse media platforms. If you’re creative, curious, and eager to build your career in video production, this role offers an excellent opportunity to grow both your skills and your confidence. This paid internship begins in Summer 2026 and may run until September.

Responsibilities

Support the creation of video content for AARP.Org and social media channels, including livestreaming, short and long-form video
Assist producers and editors with research, scripting, video editing, fact-checking, logging footage, and social media support
Help brainstorm and execute creative ideas that foster social connection and engagement across programs
Assist with quality assurance of content, links, video, and post copy prior to posting and publishing

Qualifications

Must be enrolled in a degree program at an accredited college/university, rising undergraduate juniors or seniors, graduate students, or post-doctoral students, and remain academically enrolled throughout the internship, or must have previously graduated from college and enrolled in a continuing education program
Must be a self-starter with the ability to stay organized, manage multiple tasks, and meet deadlines
Must be extremely detail-oriented
Basic experience with Adobe Premiere and Photoshop
Basic understanding of TikTok, LinkedIn, Facebook, Instagram, and YouTube

AARP will not sponsor an employment visa for this position at this time.

Additional Requirements

Regular and reliable job attendance
Exhibit respect and understanding of others to maintain professional relationships
In office/open office environment with the ability to work effectively surrounded by moderate noise

Hybrid Work Environment

AARP observes Mondays and Fridays as remote workdays, except for essential functions. Remote work can only be done within the United States and its territories.

Compensation And Benefits

The hourly range is $18 for undergraduates, $21 for graduate students, and $28 for Ph.D. candidates. Internships are non-exempt positions and are not eligible for employee benefits.

Equal Employment Opportunity

AARP is an equal opportunity employer committed to hiring a diverse workforce and sustaining an inclusive culture. AARP does not discriminate on the basis of race, ethnicity, religion, sex, color, national origin, age, sexual orientation, gender identity or expression, mental or physical disability, genetic information, veteran status, or on any other basis prohibited by applicable law.

Seniority level
Not Applicable
Employment type
Internship
Job function
Marketing, Public Relations, and Writing/Editing
Industries
Non-profit Organizations"""

if __name__ == "__main__":
    asyncio.run(chunk_word_optimizations(dummy_text))