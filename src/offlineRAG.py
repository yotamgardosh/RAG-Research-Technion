# -*- coding: utf-8 -*-

!pip install sentence-transformers
!pip install langchain
!pip install -U langchain-community
!pip install chromadb
!pip install langchain_cohere
!pip install tiktoken
!pip install nltk
!pip install langchain_groq
!pip install transformers
!pip install openai

import os
from dotenv import load_dotenv
import json


# Load environment variables from '.env' file
load_dotenv()

# my personal api key, dont share with the world
os.environ['GROQ_API_KEY'] = "********************************************"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "**************************************"

# retrival
from sentence_transformers import SentenceTransformer
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document


# splitting
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt')


# relevency
import re
import openai
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_groq import ChatGroq
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from google.colab import drive
from huggingface_hub import notebook_login
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI


def login():
    notebook_login()

def mount_drive():
    drive.mount('/content/drive')



# Function to extract metadata from the filename
def extract_metadata(file_name):
    print(file_name)
    # Assuming the structure is "category-subcategory-file_name.txt"
    if "-" in file_name:
        parts = file_name.rsplit("-", 2)  # Split the filename into category, subcategory, and file name
        category = parts[0].replace("_", " ")  # Replace underscores with spaces for category
        subcategory = parts[1].replace("_", " ")  # Replace underscores with spaces for subcategory
        title = parts[2].replace("_", " ").replace(".txt", "")  # Replace underscores with spaces and remove .txt
    else:
        category = "Unknown"
        subcategory = "Unknown"
        title = file_name.replace(".txt", "")  # Use the entire file name as the title if structure is different

    return title, category, subcategory

"""# Testing different spliting algorithms:
- simple split based on paragrpahs.
- more advanced using RecursiveCharacterTextSplitter
- HuggingFace - BERT split function
- HuggingFace - BERT split function + natural languge techniques
"""

def simple_paragraph_split(content):
    return [page for page in content.split('\n\n') if page.strip()]

def recursive_character_split(content, chunk_size=250, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return text_splitter.split_text(content)

def huggingface_token_split(content, max_token_size=256, overlap=50):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokens = tokenizer.tokenize(content)
    chunks = []
    for i in range(0, len(tokens), max_token_size - overlap):
        chunk = tokens[i:i + max_token_size]
        chunks.append(tokenizer.convert_tokens_to_string(chunk))
    return chunks

def huggingface_nltk_sentence_split(content, max_token_size=500, overlap=80):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    sentences = sent_tokenize(content)
    current_chunk = []
    current_length = 0
    chunks = []
    for sentence in sentences:
        token_count = len(tokenizer.tokenize(sentence))
        if current_length + token_count > max_token_size:
            chunks.append(tokenizer.convert_tokens_to_string(current_chunk))
            current_chunk = current_chunk[-overlap:]
            current_length = len(tokenizer.tokenize(' '.join(current_chunk)))
        current_chunk.append(sentence)
        current_length += token_count
    if current_chunk:
        chunks.append(tokenizer.convert_tokens_to_string(current_chunk))
    return chunks

"""algorithm to manuly remove irrelevant text (titles, TOC, etc')"""


# Precompiled regex patterns
NAVIGATION_PATTERN = re.compile(r'<.*?>')
VERSION_METADATA_PATTERN = re.compile(r'SEBOK v\.\s*\d+\.\d+\.\d+|released \d{1,2} \w+ \d{4}', re.IGNORECASE)
FIGURE_PATTERN = re.compile(r'Figure\s+\d+.*|^Page\s+\d+$', re.IGNORECASE)
REFERENCES_PATTERN = re.compile(r'^References\s*\nNone\.$', re.IGNORECASE)
DOTS_DASHES_PATTERN = re.compile(r'[\.\-]{5,}')
ENDS_WITH_NUMBER_PATTERN = re.compile(r'[\s\.\-]\d+\s$')
NUMBERING_PATTERN = re.compile(r"^\d+(\.\d+)*\s")
BIBLIOGRAPHIC_PATTERN = re.compile(
    r'(ISBN|pp\.|Ver\.|Technical Report|Handbook|Edition|CRC Press|Taylor & Francis|MITRE|INCOSE)', re.IGNORECASE
)
STOP_WORDS = {"the", "of", "and", "to", "in", "on", "with", "a", "is"}

def is_potentially_irrelevant(text, title_keywords=None):
    if title_keywords is None:
        title_keywords = ["chapter", "contents", "table of contents", "section", "index"]

    # Check for navigation hints and version metadata
    if NAVIGATION_PATTERN.search(text) or VERSION_METADATA_PATTERN.search(text):
        return True

    # Check for figure labels, page numbers, and empty references
    if FIGURE_PATTERN.search(text) or REFERENCES_PATTERN.search(text):
        return True

    # Check for a high number of dots or dashes
    if DOTS_DASHES_PATTERN.search(text):
        return True

    if sum(c == '.' for c in text) / len(text) > 0.2 or sum(c == '•' for c in text) / len(text) > 0.2:
        return True

    # Check if the line ends with a number
    if ENDS_WITH_NUMBER_PATTERN.search(text):
        return True

    # Check for bibliographic patterns
    if BIBLIOGRAPHIC_PATTERN.search(text):
        return True

    # Check for common title or TOC keywords
    if any(keyword.lower() in text.lower() for keyword in title_keywords):
        return True

    # Exclude very short text with numbering
    if NUMBERING_PATTERN.match(text):
        return True

    # Check for excessive capitalization
    words = [word for word in text.split() if word.lower() not in STOP_WORDS]
    capitalized_words = sum(1 for word in words if word.isupper())
    if len(words) > 0 and capitalized_words >= len(words) / 2:
        return True

    # Exclude chunks that are too short or too long
    if len(text.split()) < 15 or len(text) > 1000:
        return True

    # If none match, consider it relevant
    return False

def process_documents_with_split(directory_path, split_method):
    documents = []

    for file_name in os.listdir(directory_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(directory_path, file_name)

            # Extract metadata
            title, category, subcategory = extract_metadata(file_name)

            # Read the content of the file
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()

            # Use the specified split method
            split_content = split_method(content)

            # Filter out irrelevant chunks
            split_content = [chunk for chunk in split_content if not is_potentially_irrelevant(chunk)]

            # Wrap each split chunk as a Document with metadata
            documents += [Document(page_content=chunk, metadata={"title": title, "category": category, "sub_category": subcategory})
                          for chunk in split_content]

    print(f"Processed {len(documents)} documents using {split_method.__name__}.")
    return documents

doc_dir_path = "***************"
clues_dir_path = "******************"

simple_split = process_documents_with_split(clues_dir_path, simple_paragraph_split)
recursive_split = process_documents_with_split(doc_dir_path, recursive_character_split)
huggingface_split = process_documents_with_split(doc_dir_path, huggingface_token_split)
huggingface_nltk = process_documents_with_split(doc_dir_path, huggingface_nltk_sentence_split)


# Specify the output file path
output_file_path = '*********************************

# Open the file in write mode
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    for doc in recursive_split:
        output_file.write("-----------------------\n")
        output_file.write(doc.page_content + "\n")
        output_file.write("-----------------------\n")

print(f"All chunks have been saved to {output_file_path}")

"""# Create Documents object from split text files

method that extract just the question from the query.
- ment to prevent retrival of data that's similar to wrong answer
"""

def extract_question(query_string):
    start_marker = "Question: "
    end_marker_options = ["\nA.", "\nAnswer:"]

    # Find the start of the question
    start_index = query_string.find(start_marker)
    if start_index == -1:
        start_index = 0
    else:
        start_index += len(start_marker)

    # Find the earliest occurrence of any end marker
    end_indices = [query_string.find(em, start_index) for em in end_marker_options]
    end_indices = [idx for idx in end_indices if idx != -1]
    if end_indices:
        end_index = min(end_indices)
    else:
        end_index = len(query_string)

    question_text = query_string[start_index:end_index].strip()
    return question_text


"""# LLM as a Judge"""

# Define LLM-based relevance model
class GradeDocuments(BaseModel):
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

# Initialize LLM model for relevance grading
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeDocuments)

# Define prompt template
system_prompt = """You are a grader assessing relevance of a retrieved document to a user question.
If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
It does not need to be a stringent test. The goal is to filter out erroneous retrievals.
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
grade_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "Retrieved document: \n\n {document} \n\n User question: {question}")
])
retrieval_grader = grade_prompt | structured_llm_grader

# Helper function to check LLM-based relevance
def llm_relevance_check(doc, question):
    torch.cuda.empty_cache()
    response = retrieval_grader.invoke({"question": question, "document": doc.page_content})
    return response.binary_score == 'yes'


system_prompt = """You are a grader assessing relevance of a retrieved document to a user question.
If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
It does not need to be a stringent test. The goal is to filter out erroneous retrievals.
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

"""Lamma_3_3B"""


# Load the tokenizer and model
model_name = "meta-llama/Llama-3.2-3B-Instruct"  # Adjust based on the specific Llama 3 model you intend to use
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def check_relevance_llama3(document, query):
    document_content = document.page_content  # Ensure this is the correct attribute for your document's content
    prompt = f"{system_prompt}\nQuestion: {query}\nDocument: {document_content}"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    output = model.generate(**inputs, max_new_tokens=5, do_sample=False)
    answer = tokenizer.decode(output[0], skip_special_tokens=True).strip().lower()
    is_relevant = "yes" in answer
    return is_relevant

"""Bert"""

def check_relevance_bert(document, query):
    document_content = document.page_content  # Ensure this is the correct attribute for your document's content
    prompt_text = f"{system_prompt}\nQuestion: {query}\nDocument: {document_content}"
    inputs = tokenizer.encode_plus(prompt_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model(**inputs)

    # Obtain scores and decide relevance
    scores = torch.softmax(outputs.logits, dim=1)
    is_relevant = scores[:, 1].item() > 0.6  # Assuming the second label corresponds to "Yes"
    if is_relevant:
      print(f"Query: {query}\nDocument snippet: {document_content[:5]}... Score for Relevance: {scores[:, 1].item()}\n\n")

    return is_relevant

"""GPT"""


def activate_openAi():
    key = "*******************************************"
    client = OpenAI(
        api_key = key
    )

def check_relevance_with_gradual_scores(document, query):
    prompt = f"""
    You are a systems engineering assistant. Assess if the retrieved document helps answer the query.
    Options: Relevant, Partially Relevant, Irrelevant.

    Query: "{query}"
    Document: "{document.page_content}"
    """
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert assistant evaluating relevance."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    relevance = response.choices[0].message.content.strip().lower()
    return relevance in ["relevant", "partially relevant"]

def check_relevance_with_chatgpt(document, query):
    relevance_prompt_template = """
    You are a systems engineering assistant. Your goal is to assess if the retrieved document provides the specific information required to answer the query accurately. Focus on whether the document includes details that directly support or relate to the correct answer choice.
    Query: "{query}"
    Retrieved Document: "{retrieved_text}"
    Based on the information in the retrieved document, can it help answer the query correctly? Answer 'true' if the document provides relevant and answer-specific details, or 'false' if it does not. Briefly explain your choice, focusing on whether the document directly addresses concepts related to the correct answer choice.
    """
    document_content = document.page_content  # Ensure this aligns with your document's attribute structure
    prompt = relevance_prompt_template.format(query=query, retrieved_text=document_content)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # or "gpt-3.5-turbo" for faster, less expensive evaluations
        messages=[
            {"role": "system", "content": "You are an expert assistant evaluating relevance."},
            {"role": "user", "content": prompt}
        ],
        temperature=0  # Use 0 for deterministic output
    )

    # Extract the assistant's response
    relevance_answer = response.choices[0].message.content.strip()
    is_relevant = "yes" in relevance_answer.lower()

    return is_relevant

"""# Retrive k document per query for all queries in the benchmark
- saves retrived fils in the RAG-eval foramt in provided output dir
"""

print("Starting process_queries_and_save_retrievals")
embeddings = SentenceTransformerEmbeddings(model_name='all-mpnet-base-v2')
print("Embeddings initialized")
vector_store = Chroma.from_documents(recursive_split, embedding=embeddings)
print("Vector store created")


def process_queries_and_save_retrievals(
    documents,
    name,
    query_file_path,
    output_dir="*******************************************************",
    model_name='all-mpnet-base-v2',
    with_relevancy=False,
    verbose=True
):
    with open(query_file_path, 'r') as f:
        queries = [json.loads(line)["query"] for line in f]

    retrieved_data = []
    retrieved_cache = {}  # Cache to store retrieved documents for each unique question
    i = 0

    for query in queries:
        i += 1
        print(f"Processing query {i}/{len(queries)}\n{'='*50}")
        print(f"Query: {query}")

        # Extract only the question part for retrieval
        question = extract_question(query)

        # Check if we already retrieved documents for this question
        if question in retrieved_cache:
            relevant_docs = retrieved_cache[question]
        else:
            # Retrieve top-k documents for the query in this case k =4
            retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={'k': 4}  # Number of documents to retrieve
            )
            retrieved_docs = retriever.invoke(query)

            # Filter out duplicate documents based on page_content
            unique_docs = []
            seen_texts = set()
            for doc in retrieved_docs:
                if doc.page_content not in seen_texts:
                    unique_docs.append(doc)
                    seen_texts.add(doc.page_content)

            # Apply relevance check
            if with_relevancy:
                print("Applying relevance checks...")
                relevant_docs = [doc for doc in unique_docs if check_relevance_with_gradual_scores(doc, query)]
            else:
                relevant_docs = unique_docs

            # Print retrieved documents (optional verbose mode)
            if verbose:
                print(f"Retrieved Documents for Query {i} (with_relevancy={with_relevancy}):")
                for idx, doc in enumerate(relevant_docs):
                    print(f"  Document {idx+1}: {doc.page_content[:200]}...")  # Print a preview of the content

            # Store retrieved documents in the cache
            retrieved_cache[question] = relevant_docs

        # Format the retrieved documents into the required format for RAG evaluation
        framing_context = "Imagine you are a Systems Engineering expert. "
        cot_reasoning_context = (
            "Based on the retrieved information, evaluate the following carefully. "
            "Analyze each piece of information step-by-step to answer the question:"
        )
        cot_reasoning_context_empty_RAG = (
            "Think thoroughly about the question before answering."
        )

        # Check if relevant_docs is empty
        if not relevant_docs:
            # Handle case with no relevant documents
            retrieved_data.append({
                "query": query,
                "ctxs": [
                    {
                        "retrieval text": (
                            "Instruction: " + framing_context + cot_reasoning_context_empty_RAG + "\n"
                        )
                    }
                ]
            })
        else:
            # Handle case with relevant documents
            retrieved_data.append({
                "query": query,
                "ctxs": [
                    {
                        "retrieval text": (
                            "Instruction: " + framing_context + cot_reasoning_context + "\n" + "Context:\n" + doc.page_content + "\n"
                            if idx == 0
                            else cot_reasoning_context + "\n" + doc.page_content + "\n"
                        )
                    }
                    for idx, doc in enumerate(relevant_docs)
                ]
            })

        # Print summary of processed query
        if verbose:
            print(f"Finished processing Query {i}/{len(queries)}")
            print(f"Retrieved {len(relevant_docs)} relevant documents.")
            print("="*50)

    # Use the 'name' parameter for a concise filename
    output_file_path = os.path.join(output_dir, f"{name}_retrieved_documents.jsonl")

    # Save the retrieved data in JSONL format for RAG evaluation
    with open(output_file_path, 'w') as f:
        for item in retrieved_data:
            f.write(json.dumps(item) + '\n')

    print(f"Retrieved documents saved successfully to {output_file_path}!")




def add_framing_to_empty_ctxs(file_path):
    # Define the framing and reasoning contexts
    framing_context = "Imagine you are a Systems Engineering expert. "
    cot_reasoning_context_empty_RAG = (
        "Think thoroughly about the question before answering"
    )

    updated_data = []

    # Read the existing JSONL file
    with open(file_path, 'r') as f:
        for line in f:
            entry = json.loads(line)

            # Check if the ctxs is empty
            if not entry.get("ctxs"):
                # Add a placeholder with framing and reasoning contexts
                entry["ctxs"] = [{
                    "retrieval text": framing_context + cot_reasoning_context_empty_RAG
                }]

            updated_data.append(entry)

    # Save the updated JSONL data back to the file
    with open(file_path, 'w') as f:
        for item in updated_data:
            f.write(json.dumps(item) + '\n')

    print(f"Updated entries with empty ctxs in {file_path}.")

