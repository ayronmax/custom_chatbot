import streamlit as st

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import MessagesPlaceholder

from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.llms import HuggingFaceHub, LlamaCpp

from dotenv import load_dotenv

load_dotenv()

# Configurações do Streamlit
st.set_page_config(page_title="Seu assistente virtual 🤖", page_icon="🤖")
st.title("Seu assistente virtual 🤖")

with st.sidebar:
    model_class = st.selectbox(
        "Seelcione a Classe do Modelo",
        ("Hugging Face LlamaCpp", "Hugging Face Hub", "Ollama", "Groq")
    )
    
    if model_class == "Hugging Face LlamaCpp":
        model = st.selectbox(
            "Seelcione o Modelo",
            (
                "/home/alexinaldo/Downloads/DuckDB-NSQL-7B-v0.1-q8_0.gguf"
            )
        )

    if model_class == "Hugging Face Hub":
        model = st.selectbox(
            "Seelcione o Modelo",
            (
                "meta-llama/Meta-Llama-3-8B-Instruct"
            )
        )

    if model_class == "Ollama":
        model = st.selectbox(
            "Seelcione o Modelo",
            ("phi3")
        )

    if model_class == "Groq":
        model = st.selectbox(
            "Seelcione o Modelo",
            ("mixtral-8x7b-32768")
        )

    if model:
        temperature = st.slider(    
            "Temperatura",
            min_value=0.0,
            max_value=1.0,
            value=0.1,  # Valor inicial
            step=0.1,  # Incremento de 0.1
            format="%.1f"  # Formato com uma casa decimal
        )

def model_hf_llamacpp(model=model, temperature=temperature):
    llm = LlamaCpp(
        model_path=model,
        n_ctx=2048,              # Tamanho do contexto (pode aumentar dependendo da RAM disponível)
        temperature=temperature,
        n_threads=2,             # Número de threads para processamento
        n_batch=512,             # Tamanho do batch para inferência
        top_k=40,                # Limita a amostragem às k tokens mais prováveis
        top_p=0.95,              # Nucleus sampling - probabilidade cumulativa máxima
        repeat_penalty=1.1,      # Penalidade para repetição de tokens
        last_n_tokens_size=64,   # Número de tokens passados para considerar na repeat_penalty
        seed=-1,                 # Seed para reprodutibilidade (-1 para aleatório)
        stop=["</s>", "Human:", "Assistant:"],  # Tokens de parada
        f16_kv=True,            # Usar precisão float16 para key/value cache
        verbose=True            # Mostrar informações detalhadas durante a execução
    )
    return llm

def model_hf_hub(model=model, temperature=temperature):
    llm = HuggingFaceHub(
        repo_id=model,
        model_kwargs={
            "temperature": temperature,
            "return_full_text": False,
            "max_new_tokens": 512,
            "repetition_penalty": 1.3,
            "stop": ["<|eot_id|>"],
            # demais parâmetros que desejar
        }
    )
    return llm

def model_ollama(model="phi3", temperature=0.1):
    llm = ChatOllama(
        model=model,
        temperature=temperature,
    )
    return llm

def model_groq(model=model, temperature=temperature): 
    llm = ChatGroq(
        model=model,
        temperature=temperature,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        # other params...
    )
    return llm

def model_response(user_query, chat_history, model_class):
    ## Carregamento da LLM
    if model_class == "Hugging Face LlamaCpp":
        llm = model_hf_llamacpp()
    elif model_class == "Hugging Face Hub":
        llm = model_hf_hub()
    elif model_class == "Ollama":
        llm = model_ollama()
    elif model_class == "Groq":
        llm = model_groq()

    ## Definição dos prompts
    system_prompt = """
    You are a helpful assistant and are answering general questions. Respond in {language}.
    """
    # corresponde à variável do idioma em nosso template
    language = "portuguese"

    # Adequando à pipeline
    if model_class.startswith("hf"):
        user_prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    else:
        user_prompt = "{input}"

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", user_prompt)
    ])

    ## Criação da Chain
    chain = prompt_template | llm | StrOutputParser()

    ## Retorno da resposta / Stream
    return chain.stream({
        "chat_history": chat_history,
        "input": user_query,
        "language": language
    })


if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Olá, sou o seu assistente virtual! Como posso ajudar você?"),
    ]

for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

user_query = st.chat_input("Digite sua mensagem aqui...")

if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        resp = st.write_stream(model_response(user_query, st.session_state.chat_history, model_class))
        print(st.session_state.chat_history)

    st.session_state.chat_history.append(AIMessage(content=resp))