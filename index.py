import os
import gradio as gr
import re

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_openai import OpenAI
from difflib import SequenceMatcher

def clean_format_docs(name, pdf_docs, subject):
  text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )

  chunks = []

  for doc in pdf_docs:
      current_page = doc.metadata.get("page", None)

      # Filter based on filename + page range
      if name == 'Maturidade-Emocional-Frederico-Mattos':
          if not (19 < current_page < 215):
              continue
      elif name == 'Como-ser-Adulto-nos-Relacionamentos-David-Richo':
          if not (22 < current_page < 323):
              continue
      # For all others, include all pages

      # Clean text
      cleaned_text = doc.page_content.replace("\n", " ")

      # Split this page into chunks
      page_chunks = text_splitter.create_documents(
          [cleaned_text],
          metadatas=[{
              "name": name,
              "subject": subject,
              "page": current_page
          }]
      )

      chunks.extend(page_chunks)

  return chunks

def load_pdfs_from_folder(folder_path, subject):
  all_texts = []
  for filename in os.listdir(folder_path):
      if filename.endswith(".pdf"):
          file_path = os.path.join(folder_path, filename)
          name = os.path.splitext(filename)[0]
          pdf_loader = PyPDFLoader(file_path, mode="page")
          pdf_docs = pdf_loader.load()
          texts = clean_format_docs(name, pdf_docs, subject)
          all_texts.extend(texts)
  return all_texts

def chunked(iterable, size):
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]

load_dotenv()

MAX_BATCH_SIZE = 5000
FRIENDSHIP_FOLDER = 'content/friendship'
LOVE_FOLDER = 'content/love'
DB_PERSIST_PATH = './db'
EMBEDDING_MODEL = 'text-embedding-3-small'

embedding = OpenAIEmbeddings(model=EMBEDDING_MODEL)

friendship_texts = load_pdfs_from_folder(FRIENDSHIP_FOLDER, 'friendship')
love_texts = load_pdfs_from_folder(LOVE_FOLDER, 'love')

all_docs = friendship_texts + love_texts

if os.path.exists(DB_PERSIST_PATH):
    vector_store = Chroma(
        persist_directory=DB_PERSIST_PATH,
        embedding_function=embedding
    )
else:
    vector_store = Chroma.from_documents(
        all_docs,
        embedding=embedding,
        persist_directory=DB_PERSIST_PATH
    )

existing_docs = vector_store.get()["metadatas"]

existing_keys = set(
    (meta["name"], meta["subject"], meta["page"]) for meta in existing_docs
)

new_docs = [
    doc for doc in all_docs
    if (doc.metadata.get("name"), doc.metadata.get("subject"), doc.metadata.get("page")) not in existing_keys
]

if new_docs:
    for batch in chunked(new_docs, MAX_BATCH_SIZE):
        vector_store.add_documents(batch)
    vector_store.persist()

class LLMWrapper:
    def __init__(self, model="openai"):
        if model == "openai":
            self.llm = OpenAI(model="gpt-4o-mini", temperature=0.7)
        else:
            raise ValueError("Modelo não suportado")

    def generate(self, prompt):
        return self.llm.invoke(prompt)

llm = LLMWrapper()

session_state = {
    "phase": "select_type",
    "relationship": None,
    "duration": "",
    "type_relationship": "",
    "num_questions": None,
    "turn": "A",
    "names": {"A": "", "B": ""},
    "ages": {"A": "", "B": ""},
    "genders": {"A": "", "B": ""},
    "jobs": {"A": "", "B": ""},
    "responses": {"A": [], "B": []},
    "questions": [],
    "current_q": 0
}

def generate_questions(common_interests, relationship, n=5):
    subject = "love" if relationship.lower() == "amor" else "friendship"
    results = vector_store.similarity_search(common_interests, k=20, filter={"subject": subject})
    context = ''
    for res in results:
        context += f"* {res.page_content}\n"
    a_name = session_state["names"]["A"]
    b_name = session_state["names"]["B"]
    a_age = session_state["ages"]["A"]
    b_age = session_state["ages"]["B"]
    a_gender = session_state["genders"]["A"]
    b_gender = session_state["genders"]["B"]
    a_job = session_state["jobs"]["A"]
    b_job = session_state["jobs"]["B"]
    duration = session_state["duration"]
    type_relationship = session_state["type_relationship"]



    prompt = f"""És um especialista em relações humanas. Cria exatamente {n} perguntas únicas, divertidas, emocionalmente envolventes e introspectivas para duas pessoas com um vínculo de {relationship.lower()}.

Contexto:
- Pessoa A: {a_name}, {a_age} anos, {a_gender}, trabalho: {a_job}
- Pessoa B: {b_name}, {b_age} anos, {b_gender}, trabalho: {b_job}
- Duração do vínculo: {duration}
- Tipo de vínculo: {type_relationship}

Contexto Adicional: {context}

Instruções:
- As perguntas devem ser relevantes para que {a_name} e {b_name} se compreendam melhor.
- As perguntas devem estar numeradas com algarismos e colocadas cada uma numa nova linha:
  1. [Texto da pergunta]
  2. [Texto da pergunta]
  ...
- Assegura que sejam perguntas criativas e sem repetir temas.
- Escreve em português de Portugal.
"""
    output = llm.generate(prompt)
    print('prompt')
    print(prompt)
    print('output')
    print(output)

    # Extract clean questions
    questions = re.findall(r"^\d+\.\s+(.*)", output, re.MULTILINE)
    return questions[:n]

def evaluate_connection():
    a_name = session_state["names"]["A"]
    b_name = session_state["names"]["B"]
    a_resps = session_state["responses"]["A"]
    b_resps = session_state["responses"]["B"]
    questions = session_state["questions"]

    n = min(len(questions), len(a_resps), len(b_resps))

    formatted_data = ""
    for i in range(n):
        formatted_data += (
            f"{i+1}. {questions[i]}\n"
            f"- {a_name}: {a_resps[i]}\n"
            f"- {b_name}: {b_resps[i]}\n"
        )

    prompt = f"""
Baseado nas respostas abaixo às perguntas entre duas pessoas ({a_name} e {b_name}), avalia o nível de ligação emocional entre elas numa escala de 0 a 100%.

Critérios:
- Grau de abertura emocional
- Similaridade de valores ou visões
- Interesse genuíno um pelo outro

Responde apenas com o número inteiro da pontuação e depois uma curta explicação em 1-2 frases.

Respostas:
{formatted_data}
    """

    output = llm.generate(prompt)

    print('prompt')
    print(prompt)
    print('output')
    print(output)

    return f"Pontuação da ligação entre {a_name} e {b_name}: {output}"

def chatbot(user_input, chat_history):
    global session_state
    response = ""
    phase = session_state["phase"]

    # Relationship type
    if phase == "select_type":
        relation = user_input.lower().strip()
        if relation in ["amizade", "amor"]:
            session_state["relationship"] = relation
            session_state["phase"] = "num_questions"
            response = "Quantas perguntas queres gerar? (1–10)"
        else:
            response = "Por favor escreve 'amizade' ou 'amor'."

    # Number of questions
    elif phase == "num_questions":
        if user_input.isdigit() and 1 <= int(user_input) <= 10:
            session_state["num_questions"] = int(user_input)
            session_state["phase"] = "name_A"
            response = "Qual é o nome da primeira pessoa?"
        else:
            response = "Introduza um número entre 1 e 10."

    elif phase == "name_A":
        session_state["names"]["A"] = user_input
        session_state["phase"] = "age_A"
        response = f"Qual é a idade do(a) {user_input}?"

    elif phase == "age_A":
        session_state["ages"]["A"] = user_input
        session_state["phase"] = "gender_A"
        response = f"Qual é o género do(a) {session_state['names']['A']}?"

    elif phase == "gender_A":
        session_state["genders"]["A"] = user_input
        session_state["phase"] = "job_A"
        response = f"Qual é o trabalho do(a) {session_state['names']['A']}?"

    elif phase == "job_A":
        session_state["jobs"]["A"] = user_input
        session_state["phase"] = "name_B"
        response = "Qual é o nome da segunda pessoa?"

    elif phase == "name_B":
        session_state["names"]["B"] = user_input
        session_state["phase"] = "age_B"
        response = f"Qual é a idade do(a) {user_input}?"

    elif phase == "age_B":
        session_state["ages"]["B"] = user_input
        session_state["phase"] = "gender_B"
        response = f"Qual é o género do(a) {session_state['names']['B']}?"

    elif phase == "gender_B":
        session_state["genders"]["B"] = user_input
        session_state["phase"] = "job_B"
        response = f"Qual é o trabalho do(a) {session_state['names']['B']}?"

    elif phase == "job_B":
        session_state["jobs"]["B"] = user_input
        session_state["phase"] = "duration"
        response = "Qual a duração atual do vínculo? (ex: 1 mês, 6 meses, 1 ano, 5 anos ou mais)"

    elif phase == "duration":
        session_state["duration"] = user_input
        session_state["phase"] = "type_relationship"
        response = "E que tipo de vínculo procuram? (ex: amor duradouro, amor casual, amizade colorida...)"

    elif phase == "type_relationship":
        session_state["type_relationship"] = user_input
        session_state["phase"] = "common_interests"
        response = "Ótimo! Antes de gerarmos as perguntas, digam-me: que interesses ou gostos em comum vocês têm?"

    elif phase == "common_interests":
        common_interests = user_input
        session_state["phase"] = "questions"
        session_state["questions"] = generate_questions(common_interests, session_state["relationship"], session_state["num_questions"])
        first_question = session_state["questions"][0]
        response = f"Perfeito, vamos começar!\n\n**Questão 1:** {first_question}\n\n{session_state['names']['A']}, qual é a tua resposta?"

    elif phase == "questions":
        current_turn = session_state["turn"]
        session_state["responses"][current_turn].append(user_input)

        if current_turn == "A":
            session_state["turn"] = "B"
            response = f"{session_state['names']['B']}, qual é a tua resposta à mesma pergunta?"
        else:
            session_state["turn"] = "A"
            session_state["current_q"] += 1
            if session_state["current_q"] >= session_state["num_questions"]:
                session_state["phase"] = "done"
                result = evaluate_connection()
                return chat_history + [
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": result}
                ], ""
            else:
                next_q = session_state["questions"][session_state["current_q"]]
                response = f"**Questão {session_state['current_q']+1}:** {next_q}\n\n{session_state['names']['A']}, a tua resposta?"

    elif phase == "done":
        response = "A conversa terminou. Reinicie para tentar novamente."

    return chat_history + [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": response}
    ], ""

def reset():
    global session_state
    session_state = {
        "phase": "select_type",
        "relationship": None,
        "num_questions": None,
        "turn": "A",
        "names": {"A": "", "B": ""},
        "ages": {"A": "", "B": ""},
        "genders": {"A": "", "B": ""},
        "jobs": {"A": "", "B": ""},
        "responses": {"A": [], "B": []},
        "questions": [],
        "current_q": 0
    }
    return [], ""

with gr.Blocks() as demo:
    gr.Markdown("## Baralho de Questões: Geração de Perguntas Personalizadas")
    chatbot_ui = gr.Chatbot(type="messages")
    msg = gr.Textbox(placeholder="Começa por escrever se o vínculo é 'amizade' ou 'amor'.", label="Mensagem aqui..")
    clear_btn = gr.Button("Reiniciar")

    msg.submit(chatbot, [msg, chatbot_ui], [chatbot_ui, msg])
    clear_btn.click(reset, [], [chatbot_ui, msg])

share_mode = os.getenv("SHARE", "false").lower() == "true"

demo.launch(server_name="0.0.0.0", share=share_mode)