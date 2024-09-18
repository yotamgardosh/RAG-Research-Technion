# import os
# import json

class SystemMessage:
    def __init__(self, content):
        """
        A class to represent a system message for the chat model.

        Args:
        - content (str): The system's instructions or context.
        """
        self.content = content


class HumanMessage:
    def __init__(self, content):
        """
        A class to represent a human message for the chat model.

        Args:
        - content (str): The user's input or query.
        """
        self.content = content


# Define the function
def generate_mcq_responses(df, folder_directory, instructions, file_output_name='llm_output.json', chat_model=None):
    """
    Generates responses for multiple-choice questions using a chat model and saves the results to a JSON file.

    Args:
    - df (DataFrame): A DataFrame containing the questions and choices.
    - folder_directory (str): The directory path where the JSON output file will be saved.
    - instructions (str): Instructions to guide the chat model's responses.
    - file_output_name (str, optional): The name of the output JSON file. Defaults to 'llm_output.json'.
    - chat_model (function, optional): The chat model function to generate responses. It should accept a list of messages and return a response object with a `content` attribute.
    """
    results_dict = {}
    file_path = os.path.join(folder_directory, file_output_name)

    # Loop through each row in the DataFrame
    for index, row in df.iterrows():
        print("\nQuestion #" + str(row['Q#']))
        prompt = (
            f"Question: {row['Question']}\nChoices:\nA. {row['Choice A']}\nB. {row['Choice B']}\n"
            f"C. {row['Choice C']}\nD. {row['Choice D']}"
        )
        print(prompt)

        # Use the chat model to get the response
        if chat_model:
            result = chat_model([SystemMessage(content=instructions), HumanMessage(content=prompt)])
            results_dict[row['Q#']] = result.content
        else:
            print("Chat model is not provided.")
            return

    # Write the results_dict to a JSON file in the specified directory
    with open(file_path, 'w') as file:
        json.dump(results_dict, file)

    print(f"Results saved to {file_path}")


# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch


def load_model_and_tokenizer(model_name):
    """
    Loads the model and tokenizer from the Hugging Face Hub.

    Args:
    - model_name (str): The Hugging Face model name or path.

    Returns:
    - model: The loaded model.
    - tokenizer: The loaded tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=True,  # Enable 8-bit quantization for the model
        device_map='auto'  # Automatically use available devices (CPU/GPU)
    )
    return model, tokenizer

# example -
# # Load the chosen model and tokenizer (example: Llama-2-7B-Chat)
# model_name = "TheBloke/Llama-2-7B-Chat-GGUF"
# model, tokenizer = load_model_and_tokenizer(model_name)


def chat_model(messages, model, tokenizer):
    """
    A chat model method that takes a list of message objects and generates a response using a local language model.

    Args:
    - messages (list): A list of message objects (SystemMessage, HumanMessage) with a `content` attribute.
    - model: The language model to use for generating responses.
    - tokenizer: The tokenizer to use with the language model.

    Returns:
    - response: An object with a `content` attribute containing the generated response.
    """
    # Concatenate the content of all message objects to create the input prompt
    prompt = "\n".join([message.content for message in messages])

    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors='pt').to('cuda' if torch.cuda.is_available() else 'cpu')

    # Generate the response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=200,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,
            do_sample=True,
            top_p=0.95,
            temperature=0.7
        )

    # Decode the generated response
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Return a response object with the `content` attribute
    class Response:
        content = response_text.strip()

    return Response()
