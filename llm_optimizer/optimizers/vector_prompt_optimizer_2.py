import math
import re
from typing import List
from sentence_transformers import SentenceTransformer, util
import asyncio
import matplotlib.pyplot as plt
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import nltk
from torch import Tensor
import uuid

# Create gree

class SentenceNode: 
    def __init__(self, sentence: str, embedding: Tensor):
        self.sentence = sentence
        self.embedding = embedding
        self.connections: List[SemanticConnection] = []
        self.uuid = uuid.uuid4()

class Connection: 
    def __init__(self, node1: SentenceNode, node2: SentenceNode, similarity: float):
        self.node1 = node1
        self.node2 = node2
        self.similarity = similarity

class SemanticForest: 
    def __init__(self):
        self.nodes: List[SentenceNode] = []
        self.connections: List[Connection] = []

    def connectionEstablished(self, node1: SentenceNode, node2: SentenceNode) -> bool:
        for connection in self.connections: 
            if connection.node1.uuid == node1.uuid and connection.node2.uuid == node2.uuid:
                return True
        return False
    
    def connectDenseForrest(self):
        for i in range(len(self.nodes)):
            for j in range(i + 1, len(self.nodes)):
                node1 = self.nodes[i]
                node2 = self.nodes[j]
                if not self.connectionEstablished(node1, node2):
                    similarity = util.cos_sim(node1.embedding, node2.embedding).item()
                    if similarity >= 0:
                        connection = Connection(node1, node2, similarity)
                        self.connections.append(connection)
    
    
    def clusterNodes(self, nodes: List[SentenceNode]): 
        cluster = SentenceCluster(nodes, self.connections)
        self.nodes.append(cluster)
        for node in nodes.copy() : 
            self.nodes.remove(node)

            for connection in self.connections.copy():
                if connection.node1.uuid == node.uuid and connection.node2.uuid not in nodes:
                    self.connections.remove(connection)
                elif connection.node1.uuid == node.uuid:
                    connection.node1 = cluster
                elif connection.node2.uuid == node.uuid:
                    connection.node2 = cluster

class SentenceCluster(SentenceNode):
    def __init__(self, sentences: list[SentenceNode], connections: list[Connection]):
        # Create a representative sentence for the cluster
        cluster_sentence = " ".join([sentence.sentence for sentence in sentences])
        super().__init__(sentence=cluster_sentence, embedding=None)
        
        self.sum_connections = 0
        for connection in connections: 
            if connection.node1 in sentences or connection.node2 in sentences:
                self.sum_connections += connection.similarity

class SemanticConnection:
    def __init__(self, node: SentenceNode, similarity: float):
        self.node = node
        self.similarity = similarity
        
async def tag_pos(prompt: str):
    tokenized_words = word_tokenize(prompt)
    tagged_words = pos_tag(tokenized_words)

    return tagged_words

async def chunk_word_optimizations(prompt: str) -> List[str]:
    """
    Chunk text hierarchically:
      1. Paragraph breaks (two or more newlines)
      2. Single newlines within a paragraph
      3. Sentence-ending punctuation (. ! ?) followed by whitespace
    Returns a flat list of non-empty string chunks.
    """
    chunks: List[str] = []

    # 1. Split on paragraph breaks
    paragraphs = re.split(r'\n{2,}', prompt)

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # 2. Split on single newlines within the paragraph
        lines = para.splitlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue

            sentences = re.split(r'(?<![A-Z])(?<=[.!?])\s+', line)
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence:
                    chunks.append(sentence)

    return chunks

async def get_sentence_transformation(prompt: str, model: SentenceTransformer):
    return await asyncio.to_thread(model.encode, prompt, convert_to_tensor=True)

async def prompt_compression_algorithm():
    # Step 1: Chunk the prompt
    chunks = await chunk_word_optimizations(dummy_text)

    # Step 2: Get sentence embeddings for each chunk
    model = SentenceTransformer('all-MiniLM-L6-v2')
    chunk_embeddings = await asyncio.gather(*(get_sentence_transformation(chunk, model) for chunk in chunks))


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
    asyncio.run(chunk_word_optimizations("so I need you to be an agent that is able to scrape the web to find the optimal scraping algorithm and then build the codebase for this"))