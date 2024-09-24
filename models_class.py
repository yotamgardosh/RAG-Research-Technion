class Model:
    def __init__(self, model_name, shorter_name, file_path
                 ,n_ctx = 16000,n_threads = 32,n_gpu_layers = 0,max_tokens = 30,temperature = 0.2,top_p = 0.6):
        from langchain import LLMChain, PromptTemplate
        from huggingface_hub import hf_hub_download
        from llama_cpp import Llama
        from langchain.llms import LlamaCpp
        from langchain.schema import SystemMessage, HumanMessage
        self.model_name = model_name
        self.shorter_name = shorter_name
        self.file_path = file_path
        self.n_ctx = n_ctx,
        self.n_threads = n_threads,
        self.n_gpu_layers = n_gpu_layers,
        self.max_tokens = max_tokens,  # Limit the maximum number of tokens to generate
        self.temperature = temperature,  # Lower temperature for more deterministic output
        self.top_p = top_p

    def __str__(self):
        return f"Model Name: {self.model_name}\nShort Name: {self.shorter_name}\nPath: {self.file_path}"

    def create_model_path(self):
        return hf_hub_download(self.model_name, filename=self.file_path)

    def create_llm(self,model_path):
        return LlamaCpp(
            model_path=model_path,
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            n_gpu_layers=self.n_gpu_layers,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p
        )



