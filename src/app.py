import streamlit as st
import json

# from langchain.vectorstores import Chroma
# from langchain import PromptTemplate
# from langchain.chains import RetrievalQA
# from langchain.embeddings import GPT4AllEmbeddings
# from langchain.chat_models import ChatOpenAI
import time

# from langchain.schema import HumanMessage, SystemMessage
import os

# from dotenv import load_dotenv, find_dotenv
from utilities import parse_c_file, parse_py_file, filter_output
from agent import LLM_Agent
from models import MODEL_IDENTIFIERS
from config import load_user_config
from function_name_gpt import FunctionNameGPT

# embeddings = GPT4AllEmbeddings()

# questions = [
#     "What are the technical skills of this candidate?",
#     "Please extract all the relevant hyperlinks, contact information and email address from this document.",
#     "List all of this person's past work or internship experiences",
#     "Which universities are mentioned in the CV?",
# ]

# answers = []

# load_dotenv(find_dotenv(), override=True)

# llm = ChatOpenAI(
#     model_name="gpt-3.5-turbo",
#     temperature=0.1,
#     openai_api_key=os.environ["OPENAI_API_KEY"],
# )


# loading PDF, DOCX and TXT files as LangChain Documents
def load_document(file):
    import os

    name, extension = os.path.splitext(file)

    if extension == ".c":
        functions = parse_c_file(file)
    elif extension == ".py":
        functions = parse_py_file(file)
    else:
        print("Document format is not supported!")
        return None

    return functions


# def chunk_data(data, chunk_size=256, chunk_overlap=0):
#     from langchain.text_splitter import RecursiveCharacterTextSplitter

#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size, chunk_overlap=chunk_overlap
#     )

#     chunks = text_splitter.split_documents(data)

#     return chunks


# def generate_context(answers):
#     relevant_context = ""

#     for doc in answers:
#         relevant_context += doc + "\n"
#     return relevant_context


# def add_embedding(texts):
#     if "vs" in st.session_state:
#         vector_store = st.session_state.vs
#         vector_store._collection.delete()
#     vector_store = Chroma.from_documents(
#         documents=texts, embedding=embeddings, collection_name="resume"
#     )
#     # print(vector_store._collection)

#     return vector_store


# def query_document(question, vector_store, k):
#     retriever = vector_store.as_retriever(search_kwargs={"k": k})

#     template = """
#         Use the following pieces of context to answer the question at the end.
#         If you are unsure of the answer, just state that you don't know. Do not try to make up the answer.
#         Please do not include duplicates.
#         {context}

#         Question: {question}
#         Answer:
#         """

#     prompt = PromptTemplate(
#         template=template,
#         input_variables=["context", "question"],
#         retriever=retriever,
#     )

#     chain_type_kwargs = {"prompt": prompt}

#     qa_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=retriever,
#         return_source_documents=True,
#         chain_type_kwargs=chain_type_kwargs,
#     )

#     response = qa_chain({"query": question})

#     return response


# def assess_info(question, context):
#     template = f"""
#         Use the following pieces of context to answer the question at the end.
#         If you are unsure of the answer, just state that you don't know. Do not try to make up the answer.
#         {context}
#         """
#     messages = [
#         SystemMessage(content=template),
#         HumanMessage(content=question),
#     ]

#     response = llm(messages)

#     print(response)
#     return response.content


# def screen_doc(k):
#     vector_store = st.session_state.vs
#     global answers
#     with st.spinner("Generating Results..."):
#         for q in questions:
#             start_time = time.time()
#             response = query_document(q, vector_store, k)
#             end_time = time.time()
#             print(f"Query: {q} answered. Time taken: {end_time - start_time}")
#             answers.append(response["result"])
#     st.success("Documents Retrieved!")

#     return answers


# # clear the chat history from streamlit session state
# def clear_history():
#     if "history" in st.session_state:
#         del st.session_state["history"]


if __name__ == "__main__":
    import os

    st.subheader("Code Analysis Application 🤖")

    # file uploader widget
    uploaded_file = st.file_uploader("Upload a file:", type=["c", "py"])

    # Initiate Local LLM
    config = load_user_config("example_config.toml")
    gpt = FunctionNameGPT(config)

    # chunk_overlap = st.number_input(
    #     "Chunk overlap:",
    #     min_value=0,
    #     max_value=500,
    #     value=20,
    #     on_change=clear_history,
    # )

    # k number input widget
    # k = st.number_input(
    #     "k", min_value=1, max_value=20, value=3, on_change=clear_history
    # )

    # add_data = st.button("Add Data", on_click=clear_history)

    if uploaded_file:
        # Read and embed file
        file_name = os.path.join("../data", uploaded_file.name)

        functions = load_document(file_name)
        # chunks = chunk_data(data, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        # st.write(f"Chunk size: {chunk_size}, Chunks: {len(chunks)}")

        # vector_store = add_embedding(chunks)

        # st.session_state.vs = vector_store

        # generated_doc = screen_doc(k)

        # st.session_state.context = generated_doc

        st.subheader("Functions and Suggested Function Names:")
        col1, col2, col3 = st.columns(3)

        for function in functions:
            col1.text_area("Function", value=function, height=50)

            prompt = f"""
            
            ### Instruction:
            
            Analyze its operations, logic, and any identifiable patterns to suggest a suitable function name, do not return the original function name. \n
            
            Only return the suggested name and description of the following code function, strictly in JSON format. \n
            
            Do not include any unnecessary information beyond the JSON output. \n
            
            Code:
            \n
            {function}
            
            ### Response:
            """
            fewshotprompt = gpt.build_function_name_few_shot(prompt)
            llm_output = gpt.llm.invoke(fewshotprompt)
            print(llm_output)
            print(type(llm_output))
            json_output = filter_output(str(llm_output))
            print(json_output)
            if json_output:

                col2.text_area(
                    "Suggested Function Name",
                    value=json_output["name"],
                    height=50,
                )

                col3.text_area(
                    "Suggested Function Description",
                    value=json_output["description"],
                    height=50,
                )
            else:

                col2.text_area("Suggested Function Name", value=json_output, height=50)

                col3.text_area(
                    "Suggested Function Description",
                    value=json_output,
                    height=50,
                )

        st.success(f"Function Name Suggestions Generated! \n Source: {file_name}!")
