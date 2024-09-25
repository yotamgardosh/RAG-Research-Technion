
from openpyxl import load_workbook
from openpyxl.formatting.rule import CellIsRule
from openpyxl.styles import PatternFill
import pandas as pd
import os
from langchain import LLMChain, PromptTemplate
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from langchain.llms import LlamaCpp
from langchain.schema import SystemMessage, HumanMessage
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from langchain.llms import LlamaCpp
from langchain.schema import SystemMessage, HumanMessage

class Model:
    def __init__(self, model_name, shorter_name, file_path, n_ctx=16000, n_threads=32, n_gpu_layers=0, max_tokens=30, temperature=0.2, top_p=0.6):

        """
        Initialize a new Model instance with configuration for language model processing.

        Args:
            model_name (str): The name of the model as recognized by the hosting service (e.g., Hugging Face Hub).
            shorter_name (str): A shorter or more convenient name for the model.
            file_path (str): Path where model files will be stored or accessed.
            n_ctx (int): The number of context tokens the model supports.
            n_threads (int): Number of threads to use for parallel processing.
            n_gpu_layers (int): Number of GPU layers to utilize (if any).
            max_tokens (int): Maximum number of tokens to generate in a single request.
            temperature (float): Controls randomness in generation. Lower values lead to more deterministic outputs.
            top_p (float): Top-p sampling probability threshold for nucleus sampling.
        """
        self.model_name = model_name
        self.shorter_name = shorter_name
        self.file_path = file_path
        self.n_ctx = n_ctx,
        self.n_threads = n_threads,
        self.n_gpu_layers = n_gpu_layers,
        self.max_tokens = max_tokens,  # Limit the maximum number of tokens to generate
        self.temperature = temperature,  # Lower temperature for more deterministic output
        self.top_p = top_p
        print("------------------------------------------------------------------------")
        print(f"Initialized model '{self.shorter_name}' with the following settings:")
        print(f"Context length: {self.n_ctx}")
        print(f"Threads: {self.n_threads}")
        print(f"GPU layers: {self.n_gpu_layers}")
        print(f"Max tokens: {self.max_tokens}")
        print(f"Temperature: {self.temperature}")
        print(f"Top-p sampling: {self.top_p}")
        print("\nAvailable Methods:")
        print("  - create_model_path(): Download model files to a specified path.")
        print("  - create_llm(): Create and configure a language model for use.")
        print("  - __str__(): Return a string representation of the model's settings.")
        print("------------------------------------------------------------------------")
        print("\n\n")


    def __str__(self):
        """
        Returns a formatted string representation of the model settings.

        Returns:
            str: A description of the model.
        """
        return f"Model Name: {self.model_name}\nShort Name: {self.shorter_name}\nPath: {self.file_path}"
    def create_model_path(self):
        """
        Downloads the model files using Hugging Face Hub based on the model's configuration.

        Returns:
            str: The path to the downloaded model files.
        """
        return hf_hub_download(self.model_name, filename=self.file_path)

    def create_llm(self):
        """
        Creates and returns a configured language model instance using the provided model path.

        Returns:
            LlamaCpp: A language model instance configured with specified settings.
        """
        model_path = self.create_model_path()  # Ensure we download the model first
        return LlamaCpp(
            model_path=model_path,
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            n_gpu_layers=self.n_gpu_layers,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p
        )


