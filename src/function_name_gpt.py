from models import MODEL_IDENTIFIERS
from langchain_community.llms import LlamaCpp

from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate


class CodeAnalyzer(BaseModel):
    function_name: str = Field(description="Suggested name for the function")
    function_description: str = Field(description="Description of the function")


class FunctionNameGPT:
    """
    FunctionNameGPT facilitates querying a local Large Language Model (LLM) to suggest function names based on
    decompiler output. This functionality is particularly useful in the context of reverse engineering, where accurate
    and meaningful function names can significantly enhance the readability and understanding of disassembled code.
    Decompiler output from tools like Ghidra, Binary Ninja, or IDA Pro can be fed into this class to

    generate suggestions. The parameters used for querying the LLM are empirically determined
    to offer a balanced trade-off between the quality of the suggestions and the analysis time required.
    """

    def __init__(self, config):
        """
        Initializes the FunctionNameGPT instance with specific configurations for querying the LLM model.

        The configuration includes selecting the appropriate model from MODEL_IDENTIFIERS, setting
        a context limit, and defining generation parameters aimed at optimizing the name suggestion process.

        Parameters:
        - config (dict): A configuration dictionary to be passed to the LLM_Agent.
        """
        # Overrides specific configuration settings for FunctionNameGPT usage
        # Model identifier for the LLM
        # config["model_identifier"] = MODEL_IDENTIFIERS["deepseek-coder-6.7b-instruct"]

        # Define generation kwargs with empirically determined values for optimal performance
        config["generation_kwargs"] = {
            # Limit model output to prevent overly verbose responses
            "max_tokens": 5000,
            # Token indicating the end of the model's output
            "stop": ["</s>"],
            # Minimum probability threshold for token generation
            "min_p": 0.1,
            # Sampling temperature for diversity
            "temperature": 0.1,
            # Penalty for repeated token generation to encourage diversity
            "repeat_penalty": 1,
        }

        # Number of tokens for the context window
        self.n_context = config["n_context"]
        # Use memory mapping for model loading
        self.use_mmap = config["use_mmap"]
        # Number of CPU threads to utilize
        self.n_threads = config["n_threads"]
        # Number of model layers to offload to GPU
        self.n_gpu_layers = config["n_gpu_layers"]
        # Model identifier for downloading
        # self.model_identifier = config["model_identifier"]
        # Seed for model initialization
        self.seed = config["seed"]
        # Verbosity level
        self.verbose = config["verbose"]
        # Generation kwargs to define model inference options
        self.generation_kwargs = config["generation_kwargs"]
        self.use_mlock = config["use_mlock"]
        self.f16_kv = config["f16_kv"]
        self.rope_freq_base = config["rope_freq_base"]
        self.rope_freq_scale = config["rope_freq_scale"]
        # Downloading the model and getting its local path
        # self.model_path = self.get_model_path(self.model_identifier)
        # self.model_path = "/Users/junan/.cache/lm-studio/models/TheBloke/deepseek-coder-6.7B-instruct-GGUF/deepseek-coder-6.7b-instruct.Q4_K_M.gguf"
        self.model_path = "/Users/junan/Desktop/text-generation-webui/models/deepseek-coder-6.7b-instruct.Q4_K_M.gguf"
        self.n_batch = config["n_batch"]

        self.max_tokens = config["max_tokens"]
        # Token indicating the end of the model's output
        self.stop = config["stop"]
        # Minimum probability threshold for token generation
        # self.min_p = config["min_p"]
        # Sampling temperature for diversity
        self.temperature = config["temperature"]
        # Penalty for repeated token generation to encourage diversity
        self.repeat_penalty = config["repeat_penalty"]
        # Instantiating the Llama model with specified configuration
        self.llm = LlamaCpp(
            model_path=self.model_path,
            use_mmap=self.use_mmap,
            n_ctx=self.n_context,
            n_batch=self.n_batch,
            n_threads=self.n_threads,
            n_gpu_layers=self.n_gpu_layers,
            rope_freq_base=self.rope_freq_base,
            rope_freq_scale=self.rope_freq_scale,
            f16_kv=self.f16_kv,
            seed=self.seed,
            verbose=self.verbose,
            use_mlock=self.use_mlock,
            max_tokens=self.max_tokens,
            stop=self.stop,
            # min_p=self.min_p,
            temperature=self.temperature,
            repeat_penalty=self.repeat_penalty,
            callbacks=[StreamingStdOutCallbackHandler()],
        )

    def build_function_name_few_shot(self, parser):

        format_instructions = parser.get_format_instructions()
        # And a query intented to prompt a language model to populate the data structure.
        examples = [
            {
                "Instruction": """                
                ### Instruction:
                
                Analyze the following code's operations, logic, and any identifiable patterns to suggest a suitable function name, do not return the original function name.
                                
                Only return the suggested name and description of the following code function, strictly a JSON object.
                                
                Do not include any unnecessary information or symbols beyond the JSON output.
                
                JSON framework must be constructed with double-quotes. Double quotes within strings must be escaped with backslash, single quotes within strings will not be escaped.
                
                {format_instructions}
                
                Code:
                        
                def xxx():
                    print("hello world")
                    
                ### Response:
                """,
                "Response": '{{"function_name": "hello_world()", "function_description": "This function prints the string hello world."}}',
            },
            {
                "Instruction": """                
                ### Instruction:
                
                Analyze the following code's operations, logic, and any identifiable patterns to suggest a suitable function name, do not return the original function name.
                                
                Only return the suggested name and description of the following code function, strictly a JSON object.
                                
                Do not include any unnecessary information or symbols beyond the JSON output.
                
                JSON framework must be constructed with double-quotes. Double quotes within strings must be escaped with backslash, single quotes within strings will not be escaped.
                
                {format_instructions}
                
                Code:
                        
                def unknown(n):
                    unknown_list = [0, 1]  # Initial Fibonacci sequence with first two terms
                    while len(fib_sequence) < n:
                        next_term = unknown_list[-1] + unknown_list[-2]
                        unknown_list.append(next_term)
                    return unknown[:n]
                    
                ### Response:
                """,
                "Response": '{{"function_name": "fibonacci(n)", "function_description": "This function returns the fibonacci number at the nth sequence."}}',
            },
        ]
        example_prompt = PromptTemplate(
            input_variables=["Instruction", "Response"],
            template="{Instruction}\n{Response}",
        )

        fewshotprompt = FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
            suffix="""
                {input} 
                """,
            input_variables=["input"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        return fewshotprompt

    def build_function_name_prompt(self, code):
        """
        Constructs a custom prompt tailored for querying the LLM to suggest function names based on decompiler output.

        Parameters:
        - code (str): The decompiler output for a given function.

        Returns:
        - str: A formatted prompt for the LLM including the instruction and the decompiled code.
        """
        # Constructing a detailed prompt to guide the LLM in generating a suitable function name
        user_prompt = (
            f"<s>[INST]Given the following decompiler output for a function, "
            f"analyze its operations, logic, and any identifiable patterns to suggest a suitable function name. "
            f"Your response should strictly be the function name suggestion and up to 20 characters. "
            f"Discard all explanations or content, only the suggested name.[/INST] add_two_values</s> "
            f"[INST]Here's the code:\n {code}[/INST]"
        )
        return self.agent.build_prompt(user_prompt)

    def build_code_description_prompt(self, code):
        """
        Constructs a custom prompt tailored for querying the LLM to suggest function names based on decompiler output.

        Parameters:
        - code (str): The decompiler output for a given function.

        Returns:
        - str: A formatted prompt for the LLM including the instruction and the decompiled code.
        """
        # Constructing a detailed prompt to guide the LLM in generating a suitable function name
        user_prompt = (
            f"<s>[INST]Given the following decompiler output for a function, "
            f"analyze its operations, logic, and any identifiable patterns to suggest a suitable function description. "
            f"Your response should strictly be the function description. "
            f"[INST]Here's the code:\n {code}[/INST]"
        )
        return self.agent.build_prompt(user_prompt)

    def query_gpt_for_function_name_suggestion(self, code):
        """
        Directly queries the GPT model for a function name suggestion based on the provided decompiler output.

        Parameters:
        - code (str): The decompiler output for a given function.

        Returns:
        - The raw output from the LLM model as a response to the query.
        """
        # Passes the custom prompt to the LLM_Agent and returns the raw response
        return self.agent.generate_response(self.build_function_name_prompt(code))

    def query_gpt_for_function_description_suggestion(self, code):
        """
        Directly queries the GPT model for a function name suggestion based on the provided decompiler output.

        Parameters:
        - code (str): The decompiler output for a given function.

        Returns:
        - The raw output from the LLM model as a response to the query.
        """
        # Passes the custom prompt to the LLM_Agent and returns the raw response
        return self.agent.generate_response(self.build_code_description_prompt(code))

    def get_function_name_suggestion(self, code):
        """
        Attempts to get a function name suggestion from the LLM. If the suggestion process fails
        (e.g., due to the code being too long), it raises an exception.

        Parameters:
        - code (str): The decompiler output for the function.

        Returns:
        - str: The suggested function name or the original name if suggestion fails.
        """
        try:
            # Attempts to query the LLM for a name suggestion and filter the output
            suggested_name = self.query_gpt_for_function_name_suggestion(code)
            return self.filter_output(suggested_name)
        except:
            # Raise an error
            raise ValueError(
                "Failed to query the LLM for a name suggestion. The input code may exceed the maximum token limit supported by the LLM."
            )

    def get_function_description_suggestion(self, code):
        """
        Attempts to get a function name suggestion from the LLM. If the suggestion process fails
        (e.g., due to the code being too long), it raises an exception.

        Parameters:
        - code (str): The decompiler output for the function.

        Returns:
        - str: The suggested function name or the original name if suggestion fails.
        """
        try:
            # Attempts to query the LLM for a name suggestion and filter the output
            suggested_name = self.query_gpt_for_function_description_suggestion(code)
            return self.filter_output(suggested_name)
        except:
            # Raise an error
            raise ValueError(
                "Failed to query the LLM for a name suggestion. The input code may exceed the maximum token limit supported by the LLM."
            )

    @staticmethod
    def filter_output(output):
        """
        Cleans the model's response by removing any additional explanations and normalizing the function name format.
        Specifically, it ensures function names containing underscores are correctly formatted without
        escape characters.

        Parameters:
        - output (str): The raw model output containing the function name suggestion.

        Returns:
        - str: The filtered and normalized function name.
        """
        # Process the model's output to extract and normalize the function name
        filtered_output = output.strip().split("\n")[0].strip().replace("\\_", "_")
        return filtered_output
