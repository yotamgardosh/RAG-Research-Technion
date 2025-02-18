# -*- coding: utf-8 -*-

# Commented out IPython magic to ensure Python compatibility.
!git clone https://github.com/RulinShao/rag-evaluation-harness.git
# %cd rag-evaluation-harness
!conda create -n lm-eval python=3.11
!pip install -e .
!mkdir -p ************/tasks/sysengbench
!touch **********************demo_sysengbench.yaml


"""Create YAML:"""

# Define the path where the YAML file will be written
# prev used - You are given a multiple-choice question. Your task is to read the provided context, think about whether it helps answer the question, and then select the best answer. Respond with only the letter corresponding to the correct answer (A, B, C, or D).
#  Read the question below and select the best answer from the options provided. Respond with only the letter corresponding to the correct answer (A, B, C, or D).
rag_yaml_file_path = "***********/partial_sysengbench_prompt_eng.yaml"
YAML_SysEngBench_string = '''
task: partial_sysengbench_prompt_eng
dataset_path: arrow
dataset_kwargs:
  data_files:
    test: **********/test/data-00000-of-00001.arrow

description: ""

training_split: null
validation_split: null
test_split: test
fewshot_split: null

output_type: multiple_choice
doc_to_text: |

  Answer the following Question.

  Question: {{question.strip()}}
  A. {{choiceA}}
  B. {{choiceB}}
  C. {{choiceC}}
  D. {{choiceD}}
  Answer:
doc_to_choice: ["A", "B", "C", "D"]
doc_to_target: label
metric_list:
  - metric: acc
    aggregation: mean
    higher_is_better: true
  - metric: acc_norm
    aggregation: mean
    higher_is_better: true
'''

with open(rag_yaml_file_path, "w") as f:
    f.write(YAML_SysEngBench_string)


# **No RAG Evaluation Run:**

Models and Docs Constants:
"""

# models
Mistral_7B="mistralai/Mistral-7B-v0.1"
Ministral_8B = "mistralai/Ministral-8B-Instruct-2410"
Ministral_3B = "ministral/Ministral-3b-instruct"
Orca_2_7B="microsoft/Orca-2-7b"
Llama_3_3B = "meta-llama/Llama-3.2-3B"
Llama_2_7B = "meta-llama/Llama-2-7b-chat-hf"
Llama_2_13B = "meta-llama/Llama-2-13b-chat-hf"
Gemma_2_2B = "google/gemma-2-2b-it"
Phi_3_4B = "microsoft/Phi-3-mini-4k-instruct"

# tasks
TASK="demo_sysengbench"
PARTIAL_TASK = "partial_sysengbench"
PARTIAL_TASK_PROMPT_ENG = "partial_sysengbench_prompt_eng"
INCLUDE_TASK_PATH = "***********"
# the retrived docs here use the most simple data split
# RETRIEVED_DOCS="***************.jsonl"




################ NO RAG ################

"""Query input extraction"""

# Saves inputs based on given task to contect so we can download and move to drive - should be automatied in the feature.

!lm_eval --model hf \
    --model_args pretrained=$Llama_3_3B \
    --include_path INCLUDE_TASK_PATH \
    --tasks $PARTIAL_TASK \
    --save_inputs_only \
    --inputs_save_dir /content

"""No RAG controll lm_hrness run Mistral_7B:"""

!lm_eval --model hf \
    --model_args pretrained=$Mistral_7B \
    --include_path $PARTIAL_TASK \
    --tasks $PARTIAL_TASK_PROMPT_ENG \
    --output "{mistral_7b_no_RAG_results}"\
    --log_samples

"""No RAG controll lm_hrness run Orca_2_7B:"""

!lm_eval --model hf \
    --model_args pretrained=$Orca_2_7B \
    --include_path $INCLUDE_TASK_PATH \
    --tasks $PARTIAL_TASK_PROMPT_ENG \
    --output "{orca_2_7b_no_RAG_results}"\
    --log_samples

"""No RAG controll lm_hrness run Llama_2_13B:"""

!lm_eval --model hf \
    --model_args pretrained=$Llama_2_13B \
    --include_path $INCLUDE_TASK_PATH \
    --tasks $PARTIAL_TASK_PROMPT_ENG \
    --output "{llama_2_13b_no_RAG}" \
    --log_samples

"""No RAG controll lm_harness rus Gemma 2 2B:"""

!lm_eval --model hf \
    --model_args pretrained=$Gemma_2_2B \
    --include_path $INCLUDE_TASK_PATH \
    --tasks $PARTIAL_TASK_PROMPT_ENG \
    --output "{Gemma_2_2B_no_RAG}" \
    --log_samples

"""No RAG controll lm_harness rus Phi 3 3.8B:"""

!lm_eval --model hf \
    --model_args pretrained=$Phi_3_4B \
    --include_path $INCLUDE_TASK_PATH \
    --tasks $PARTIAL_TASK_PROMPT_ENG \
    --output "{phi_3_4B_no_RAG}" \
    --log_samples

"""No RAG controll lm_harness rus Ministral 3B:"""

!lm_eval --model hf \
    --model_args pretrained=$Ministral_3B \
    --include_path $PARTIAL_TASK \
    --tasks $PARTIAL_TASK_PROMPT_ENG \
    --output "{ministral_3B_no_RAG}"\
    --log_samples



!lm_eval --model hf \
    --model_args pretrained=$Llama_3_3B \
    --include_path $INCLUDE_TASK_PATH \
    --tasks $PARTIAL_TASK_PROMPT_ENG \
    --output  "{llama_3_3b_no_RAG_results}"\
    --log_samples

"""No RAG controll lm_hrness run Llama_2_7B:"""

!lm_eval --model hf \
    --model_args pretrained=$Llama_2_7B \
    --include_path $INCLUDE_TASK_PATH \
    --tasks $PARTIAL_TASK_PROMPT_ENG \
    --output "{llama_2_7b_no_RAG_results}"\
    --log_samples

"""---



---

# **RAG Evaluating:**

---



---

RAG lm_harness run Mistral_7B::
"""

# Mistral as example:
!lm_eval --model hf \
    --model_args pretrained=$Mistral_7B \
    --include_path $INCLUDE_TASK_PATH \
    --tasks $OUR_TASK \
    --retrieval_file "{RETRIEVED_DOCS}" \
    --concat_k 2 \ # decides how many retrieved docs are added to system prompt.
                   # this case it added the top 2 retrieved docs for this query
    --output "{***********}" \
    --log_samples

!lm_eval --model hf \
    --model_args pretrained=$Mistral_7B \
    --include_path $INCLUDE_TASK_PATH \
    --tasks $PARTIAL_TASK_PROMPT_ENG \
    --retrieval_file "{RETRIEVED_DOCS}" \
    --concat_k 2 \
    --output "{mistral_7b_RAG_results}" \
    --log_samples



"""RAG lm_harness run Llama_2_13B:"""

!lm_eval --model hf \
    --model_args pretrained=$Llama_2_13B \
    --include_path $INCLUDE_TASK_PATH \
    --tasks $PARTIAL_TASK_PROMPT_ENG \
    --retrieval_file "{RETRIEVED_DOCS}" \
    --concat_k 1 \
    --output "{llama_2_13b_RAG_results}" \
    --log_samples

"""RAG lm_harness run Orca_2:"""


!lm_eval --model hf \
    --model_args pretrained=$Orca_2_7B \
    --include_path $INCLUDE_TASK_PATH \
    --tasks $PARTIAL_TASK_PROMPT_ENG \
    --retrieval_file "{RETRIEVED_DOCS}" \
    --concat_k 1 \
    --output "{orca_2_7b_RAG_results}" \
    --log_samples


"""RAG lm_harness run Gemma 2 2B:"""
!lm_eval --model hf \
    --model_args pretrained=$Gemma_2_2B \
    --include_path $INCLUDE_TASK_PATH \
    --tasks $PARTIAL_TASK_PROMPT_ENG \
    --retrieval_file "{RETRIEVED_DOCS}" \
    --concat_k 1 \
    --output "{Gemma_2_2B_RAG_results}" \
    --log_samples


"""RAG lm_harness run Phi 3 3.8B:"""
!lm_eval --model hf \
    --model_args pretrained=$Phi_3_4B \
    --include_path $INCLUDE_TASK_PATH \
    --tasks $PARTIAL_TASK_PROMPT_ENG \
    --retrieval_file "{RETRIEVED_DOCS}" \
    --concat_k 1 \
    --output "{phi_3_4B_RAG_results}" \
    --log_samples


"""RAG lm_harness run Llama_2_7B:"""

!lm_eval --model hf \
    --model_args pretrained=$Llama_2_7B \
    --include_path $INCLUDE_TASK_PATH \
    --tasks $PARTIAL_TASK_PROMPT_ENG \
    --retrieval_file "{RETRIEVED_DOCS}" \
    --concat_k 1 \
    --output "{llama_2_7b_RAG_results}" \
    --log_samples


"""RAG lm_harness run Llama_3_3B:"""
!lm_eval --model hf \
    --model_args pretrained=$Llama_3_3B \
    --include_path $INCLUDE_TASK_PATH \
    --tasks $PARTIAL_TASK_PROMPT_ENG \
    --retrieval_file "{RETRIEVED_DOCS}" \
    --concat_k 1 \
    --output "{llama_3_3b_RAG_results_partial_1st_path}" \
    --log_samples


"""RAG lm_harness rus Ministral_3B:"""
!lm_eval --model hf \
    --model_args pretrained=$Ministral_3B \
    --include_path $INCLUDE_TASK_PATH \
    --tasks $PARTIAL_TASK_PROMPT_ENG \
    --retrieval_file "{RETRIEVED_DOCS}" \
    --concat_k 3 \
    --output "{ministral_3B_RAG_results}" \
    --log_samples

