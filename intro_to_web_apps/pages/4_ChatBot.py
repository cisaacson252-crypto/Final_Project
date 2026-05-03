import warnings
warnings.filterwarnings('ignore')

import json
import os
import tempfile
import traceback
import importlib.util
from abc import ABC, abstractmethod
from pathlib import Path
from sqlite3 import connect
from typing import Any, Dict, List, Optional, Type, Union

import openai
import pandas as pd
import streamlit as st
import tiktoken
from pydantic import BaseModel

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent.parent.parent
DB_PATH = BASE / "data" / "structured_data.db"
CHROMA_PATH = BASE / "data" / "VectorDB"
CHROMA_COLLECTION = "corpus_collection"

# ── Load supabase ─────────────────────────────────────────────────────────────
_client_path = Path(__file__).parent.parent / "backend" / "supabase.py"
_spec = importlib.util.spec_from_file_location("supabase", _client_path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
get_posts_as_context = _module.get_posts_as_context

# ══════════════════════════════════════════════════════════════════════════════
# TOKEN COUNTER
# ══════════════════════════════════════════════════════════════════════════════

class OpenAITokenCounter:
    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.model_ = model
        try:
            self.enc_ = tiktoken.encoding_for_model(model)
        except Exception:
            self.enc_ = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        if not isinstance(text, str):
            text = str(text)
        return len(self.enc_.encode(text))

# ══════════════════════════════════════════════════════════════════════════════
# LLM
# ══════════════════════════════════════════════════════════════════════════════

OPENAI_TOKEN_LIMITS = {
    "gpt-5": 400000, "gpt-5-mini": 400000, "gpt-5-nano": 400000,
    "gpt-4": 32768, "gpt-4-turbo": 131072,
    "gpt-4o": 128000, "gpt-4o-mini": 128000,
}
SAFETY_LIM = 1000
DEFAULTS = {"model": "gpt-4o-mini", "temperature": 1}

class OpenAILLM:
    def __init__(self, api_key: str, model_args: Dict[str, Any] = DEFAULTS, **kwargs) -> None:
        self.client_ = openai.OpenAI(api_key=api_key)
        self.model_args_ = model_args
        if "model" not in model_args:
            raise ValueError("model must be specified in model_args.")
        self.token_counter_ = OpenAITokenCounter(model=model_args["model"])

    def _check_token_limit(self, text: str) -> None:
        if not isinstance(text, str):
            text = str(text)
        count = self.token_counter_.count_tokens(text)
        limit = OPENAI_TOKEN_LIMITS.get(self.model_args_["model"], 128000)
        if count + SAFETY_LIM > limit:
            raise ValueError(f"Text too long. Tokens: {count}, limit: {limit}")

    def _build_message(self, prompt, system_prompt=None):
        if system_prompt is None:
            system_prompt = "You are a helpful AI assistant."
        if isinstance(prompt, str):
            return [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
        return prompt

    def query(self, prompt, system_prompt=None) -> str:
        self._check_token_limit(prompt if isinstance(prompt, str) else str(prompt))
        message = self._build_message(prompt, system_prompt)
        response = self.client_.chat.completions.create(messages=message, **self.model_args_)
        return response.choices[0].message.content

    def structured_query(self, response_format: Type[BaseModel], prompt: str, system_prompt=None) -> BaseModel:
        self._check_token_limit(prompt)
        message = self._build_message(prompt, system_prompt)
        response = self.client_.beta.chat.completions.parse(
            messages=message, response_format=response_format, **self.model_args_
        )
        return response.choices[0].message.parsed

# ══════════════════════════════════════════════════════════════════════════════
# VECTOR DB (ChromaDB)
# ══════════════════════════════════════════════════════════════════════════════

import chromadb
from sentence_transformers import SentenceTransformer

class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_ = SentenceTransformer(model_name)

    def embed(self, text):
        if isinstance(text, str):
            text = [[text]]
        elif isinstance(text, list) and isinstance(text[0], str):
            text = [text]
        docs = []
        for doc in text:
            docs.append(self.model_.encode(doc).tolist())
        return docs

class ChromaDBVectorDB:
    def __init__(self, dbpath: str, embedder, distance_measure: str = "cosine") -> None:
        self.dbpath_ = dbpath
        self.embedder_ = embedder
        self.distance_measure_ = distance_measure
        self.client_ = None
        self.collection_ = None

    def initialize_db(self) -> None:
        if not os.path.exists(self.dbpath_):
            os.makedirs(self.dbpath_)
        self.client_ = chromadb.PersistentClient(path=self.dbpath_)

    def initialize_collection(self, collection_name: str) -> None:
        if not self.client_:
            raise ValueError("Call initialize_db first.")
        self.collection_ = self.client_.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": self.distance_measure_}
        )

    def retrieve(self, question: str, k: int = 5):
        if not self.collection_:
            raise ValueError("No collection selected.")
        query_embedding = self.embedder_.embed(question)[0][0]
        results = self.collection_.query(query_embeddings=[query_embedding], n_results=k)
        return results["documents"][0], results["distances"][0]

# ══════════════════════════════════════════════════════════════════════════════
# AGENTS
# ══════════════════════════════════════════════════════════════════════════════

class ChromaAgent:
    def __init__(self, llm, vectordb, forum_context: str = "") -> None:
        self.llm_ = llm
        self.vectordb_ = vectordb
        self.forum_context_ = forum_context
        if self.vectordb_.collection_ is None:
            raise ValueError("No collection attached to the vector database.")

    def query(self, prompt: str, k: int = 5, max_distance: float = 0.65, show_citations: bool = False):
        self.docs_, self.distances_ = self.vectordb_.retrieve(question=prompt, k=k)
        
        if not self.docs_ or not self.distances_:
            # No ChromaDB results — fall back to forum context only
            if self.forum_context_:
                self.prompt_ = (
                    "You are an intelligent AI assistant. Use the following community forum posts to answer the question.\n\n"
                    f"### Community Forum Posts ###\n{self.forum_context_}\n\n"
                    f"### Question ###\n{prompt}"
                )
                self.response_ = self.llm_.query(self.prompt_)
                return self.response_
            return None

        mindist = min(self.distances_)
        if mindist > max_distance:
            # ChromaDB results not relevant — try forum context
            if self.forum_context_:
                self.prompt_ = (
                    "You are an intelligent AI assistant. Use the following community forum posts to answer the question.\n\n"
                    f"### Community Forum Posts ###\n{self.forum_context_}\n\n"
                    f"### Question ###\n{prompt}"
                )
                self.response_ = self.llm_.query(self.prompt_)
                return self.response_
            return None

        system_prompt = (
            "You are an intelligent AI assistant answering a question via retrieval augmented generation. "
            "Use the provided context to answer accurately and concisely. "
            "NEVER reference the context you received, simply use it to answer the question.\n\n"
        )
        context_str = "\n\n".join(f"- {doc}" for doc in self.docs_)

        forum_section = ""
        if self.forum_context_:
            forum_section = f"\n\n### Community Forum Posts ###\n{self.forum_context_}"

        self.prompt_ = f"{system_prompt}### Context ###\n{context_str}{forum_section}\n\n### Question ###\n{prompt}"
        self.response_ = self.llm_.query(self.prompt_)
        return self.response_

class SQLResponse(BaseModel):
    sql_query: str
    explanation: str

class SQLiteAgent:
    def __init__(self, llm, database_url: str, db_desc: Optional[str] = None, include_detail: bool = True, **kwargs) -> None:
        self.llm_ = llm
        self.db_desc_ = db_desc
        self.include_detail_ = include_detail
        try:
            self.engine_ = connect(database_url, check_same_thread=False)
        except Exception as e:
            raise ValueError(f"Could not connect to database: {e}")
        self._build_schema()

    def _build_schema(self) -> None:
        schema: Dict[str, Any] = {}
        for ix in ['table', 'view']:
            tables = list(pd.read_sql(f"SELECT tbl_name FROM sqlite_master WHERE type = '{ix}'", self.engine_)['tbl_name'])
            if tables:
                td = {}
                for t in tables:
                    temp = pd.read_sql(f"SELECT * FROM {t} LIMIT 1", self.engine_)
                    cols = list(temp.columns)
                    types = temp.dtypes.apply(lambda x: str(x)).to_list()
                    cd = {}
                    for i, c in enumerate(cols):
                        if self.include_detail_:
                            if any(x in types[i] for x in ['int', 'float', 'date']):
                                unq = pd.read_sql(f"SELECT MIN([{c}]) AS mn, MAX([{c}]) AS mx FROM {t} LIMIT 1", self.engine_)
                                cd[c] = {'datatype': types[i], 'min': str(unq['mn'].values[0]), 'max': str(unq['mx'].values[0])}
                            else:
                                unq = pd.read_sql(f"SELECT DISTINCT [{c}] AS vls FROM {t}", self.engine_)
                                cd[c] = {'datatype': types[i], 'example values': unq['vls'].to_list()[:20]}
                        else:
                            cd[c] = {'datatype': types[i]}
                    td[t] = {'columns': cd}
                schema[ix + 's'] = td
        self.schema_json_ = json.dumps(schema)

    def query(self, prompt: str, view_sql: bool = False, retries: int = 0):
        system_prompt = ("You are an expert in converting real world questions into SQL queries.\n"
                         "Your job is to take the question below and use the provided database architecture to convert the question into a SQLite query.\n\n"
                         "You are to respond both a SQL query required to answer the provided question and a short explanation of the query.\n"
                         "When you generate the SQL query, make sure to return it with proper syntax for SQLite and make it legible.\n"
                         'Example Question: "I want to know every type of car that was sold"\n'
                         'Example Response: "SELECT DISTINCT car_type FROM Car_Sales"\n\n'
                         'Example Explanation: "The Car_Sales table has information on the total units of sales."')
        if self.db_desc_:
            base_query = (f"Given the following SQLite database description and architecture, please answer the following question:\n\n"
                     f"Database Description:\n{self.db_desc_}\n\nDatabase Architecture:\n{self.schema_json_}\n\nQuestion:\n{prompt}")
        else:
            base_query = (f"Given the following SQLite database architecture, please answer the following question:\n\n"
                     f"Database Architecture:\n{self.schema_json_}\n\nQuestion:\n{prompt}")

        attempts = 0
        error_context = ""
        last_sql = ""

        while attempts <= retries:
            try:
                if attempts > 0 and last_sql:
                    retry_system_prompt = system_prompt + "\n\nYour previous SQL query failed. Please fix the issues and try again."
                    retry_query = base_query + f"\n\nPrevious failed SQL:\n{last_sql}\n\nError:\n{error_context}\n\nArchitecture:\n{self.schema_json_}"
                    self.response_ = self.llm_.structured_query(response_format=SQLResponse, prompt=retry_query, system_prompt=retry_system_prompt)
                else:
                    self.response_ = self.llm_.structured_query(response_format=SQLResponse, prompt=base_query, system_prompt=system_prompt)

                last_sql = self.response_.sql_query
                answer = pd.read_sql(self.response_.sql_query, self.engine_)

                if answer.empty:
                    return "The query ran successfully but returned no results. Try rephrasing or broadening your question."

                final_prompt = (f"Given the following question, supporting data, and SQL generated to answer the question, "
                                f"Please provide a concise answer.\n\nQuestion:\n{prompt}\n\nSupporting Data:\n{answer.to_json()}\n\nSQL Used:\n{self.response_.sql_query}")
                result = self.llm_.query(final_prompt)
                self.response_ = result
                return result if result else "I ran the query but couldn't generate a response. Please try again."

            except Exception as e:
                error_context = f"{str(e)}\n\n{traceback.format_exc()}"
                attempts += 1
                if attempts > retries:
                    return f"SQL Agent failed after {retries+1} attempt(s): {str(e)}"

        return None

class ConductorResponse(BaseModel):
    agent_integer: int
    explanation: str

class MultiAgent:
    def __init__(self, llm, agent_names, agents, agent_descriptions, agent_query_kwargs=None) -> None:
        self.llm_ = llm
        self.agent_names_ = agent_names
        self.agents_ = agents
        self.agent_descriptions_ = agent_descriptions
        self.agent_types_ = [type(agent).__name__ for agent in agents]
        self.agent_query_kwargs_ = agent_query_kwargs if agent_query_kwargs else [{} for _ in agent_names]

        self.system_prompt_ = ("You are responsible for determining the most appropriate agent for a given question "
                               "based on the category descriptions. Each agent will be given a corresponding integer, "
                               "and your response include the integer of the agent that best matches the question and a short explanation why you selected that agent.\n\n"
                               "Example:\nAgent 0 (Text_Agent): Qualitative questions that requires text mining related to: sports history.\n"
                               "Agent 1 (Database_Agent): Quantitative questions that could be answered with a SQL query related to: sports statistics.\n"
                               "Question: How many points did Kobe Bryant score over his career?\nAnswer: 1\n"
                               "Explanation: This is a statistics question, matching agent 1.")

        self.prompt_ = "Use the following category descriptions to answer the following question:\n\n"
        type_tags = {'ChromaAgent': 'Qualitative questions that requires text mining related to: ',
                     'SQLiteAgent': 'Quantitative questions that could be answered with a SQL query related to: '}
        for i, d in enumerate(self.agent_descriptions_):
            desc = d.strip()
            if desc[-1] != '.':
                desc += '.'
            self.prompt_ += f"Category {i} ({self.agent_names_[i]}):\n{type_tags[self.agent_types_[i]]}{desc}\n"
        self.prompt_ += '\nQuestion:\n'

    def query(self, prompt: str, show_logic: bool = False):
        query = self.prompt_ + prompt
        response = self.llm_.structured_query(response_format=ConductorResponse, prompt=query, system_prompt=self.system_prompt_)
        agent_int = response.agent_integer
        if show_logic:
            print(f"Agent selected: {self.agent_names_[agent_int]}\nExplanation:\n{response.explanation}\n")
        return self.agents_[agent_int].query(prompt, **self.agent_query_kwargs_[agent_int])

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

if 'api_key' not in st.session_state:
    st.session_state['api_key'] = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'user_launched_convo' not in st.session_state:
    st.session_state['user_launched_convo'] = False
if 'llm' not in st.session_state:
    st.session_state['llm'] = None
if 'agent' not in st.session_state:
    st.session_state['agent'] = None
if 'trace_log' not in st.session_state:
    st.session_state['trace_log'] = []

# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .block-container { padding: 0 2rem 2rem 2rem !important; }
    .page-title { font-family: 'Bebas Neue', sans-serif; font-size: clamp(36px, 5vw, 64px); letter-spacing: 3px; color: #F0EDE8; margin: 32px 0 4px 0; line-height: 1; }
    .page-title .red { color: #E8302A; }
    .section-label { font-family: 'DM Mono', monospace; font-size: 10px; letter-spacing: 3px; text-transform: uppercase; color: #E8302A; margin-bottom: 8px; }
    .red-divider { width: 40px; height: 2px; background: #E8302A; margin: 8px 0 20px 0; }
    .trace-card { background: #141414; border: 1px solid #1e1e1e; border-left: 3px solid #E8302A; padding: 16px; margin-bottom: 12px; }
    .trace-q { font-family: 'DM Mono', monospace; font-size: 11px; color: #E8302A; margin-bottom: 6px; }
    .trace-agent { font-family: 'DM Mono', monospace; font-size: 10px; color: #555; margin-bottom: 4px; }
    .trace-body { font-size: 12px; color: #888; line-height: 1.6; }
    .stButton > button { background: #E8302A !important; color: white !important; border: none !important; border-radius: 0 !important; font-family: 'DM Mono', monospace !important; font-size: 11px !important; letter-spacing: 2px !important; text-transform: uppercase !important; padding: 10px 24px !important; }
    .stButton > button:hover { background: #c4241e !important; }
    [data-testid="stChatMessage"] { background: #141414 !important; border: 1px solid #1e1e1e !important; margin-bottom: 8px !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="page-title">TRAYCED <span class="red">AI</span></div>', unsafe_allow_html=True)
st.markdown('<div class="red-divider"></div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="section-label">// Configuration</div>', unsafe_allow_html=True)
    openaikey = st.text_input("OpenAI API Key", placeholder="sk-...", type="password")

chat_col, trace_col = st.columns([3, 2])

with chat_col:
    st.markdown('<div class="section-label">// Chat</div>', unsafe_allow_html=True)

    viable = False
    if not st.session_state['user_launched_convo']:
        if st.button("Begin Conversation"):
            viable = True
            if not openaikey:
                viable = False
                st.error("Enter an OpenAI API key in the sidebar.", icon="🚨")

    if viable or st.session_state['user_launched_convo']:
        if not st.session_state['user_launched_convo']:
            with st.spinner("Initializing agents..."):
                try:
                    llm = OpenAILLM(api_key=openaikey, model_args=DEFAULTS)

                    # Load forum context from Supabase
                    try:
                        forum_context = get_posts_as_context(limit=50)
                    except Exception:
                        forum_context = ""

                    embedder = SentenceTransformerEmbedder()
                    vdb = ChromaDBVectorDB(dbpath=str(CHROMA_PATH), embedder=embedder, distance_measure="cosine")
                    vdb.initialize_db()
                    vdb.initialize_collection(CHROMA_COLLECTION)
                    rag_agent = ChromaAgent(llm=llm, vectordb=vdb, forum_context=forum_context)
                    rag_desc = "general questions about cars, builds, modifications, parts, community advice, and user build posts from the REDLINE forum"
                    rag_kwargs = {"k": 5, "max_distance": 0.75, "show_citations": False}

                    db_desc = """
                    This database contains automotive make/model/year data for 18 major manufacturers spanning 1980-2024. 
                    The only table to query is called all_brands_combined. 
                    It has 43,394 records with these columns: make_id, make_name, model_id, model_name, and year. 
                    Use ONLY the all_brands_combined table. Do not query any other table.
                    """.rstrip()
                    sql_agent = SQLiteAgent(llm=llm, database_url=str(DB_PATH), db_desc=db_desc, include_detail=False)
                    sql_desc = "make, model, and production year of vehicles from 18 major automotive brands spanning 1980-2024 across 43,394 entries"
                    sql_kwargs = {"view_sql": False, "retries": 1}

                    multi_agent = MultiAgent(
                        llm=llm,
                        agent_names=["Rag Agent", "SQL Agent"],
                        agents=[rag_agent, sql_agent],
                        agent_descriptions=[rag_desc, sql_desc],
                        agent_query_kwargs=[rag_kwargs, sql_kwargs]
                    )

                    st.session_state['agent'] = multi_agent
                    st.session_state['llm'] = llm
                    st.session_state['user_launched_convo'] = True
                    st.rerun()

                except Exception as e:
                    st.error(f"Failed to initialize: {e}", icon="🚨")

        if st.session_state['user_launched_convo']:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])

            if user_response := st.chat_input("Ask about car builds, specs, or data..."):
                st.session_state.messages.append({"role": "user", "content": user_response})
                with st.chat_message("user"):
                    st.write(user_response)

                with st.spinner("Thinking..."):
                    agent = st.session_state['agent']
                    trace_info = {"agent_used": "Unknown", "explanation": "", "sql": None, "citations": None}
                    try:
                        conductor_prompt = agent.prompt_ + user_response
                        conductor_resp = agent.llm_.structured_query(
                            response_format=ConductorResponse,
                            prompt=conductor_prompt,
                            system_prompt=agent.system_prompt_
                        )
                        selected_idx = conductor_resp.agent_integer
                        trace_info['agent_used'] = agent.agent_names_[selected_idx]
                        trace_info['explanation'] = conductor_resp.explanation

                        selected_agent = agent.agents_[selected_idx]
                        kwargs = agent.agent_query_kwargs_[selected_idx]
                        response = selected_agent.query(user_response, **kwargs)

                        if hasattr(selected_agent, 'response_') and hasattr(selected_agent.response_, 'sql_query'):
                            trace_info['sql'] = selected_agent.response_.sql_query
                        if hasattr(selected_agent, 'docs_') and hasattr(selected_agent, 'distances_'):
                            trace_info['citations'] = list(zip(selected_agent.docs_, selected_agent.distances_))

                    except Exception as e:
                        response = f"Agent error: {e}"

                if response is not None:
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    with st.chat_message("assistant"):
                        st.write(response)
                    st.session_state.trace_log.append({"question": user_response, **trace_info})
                else:
                    st.warning(f"Agent returned no response. Trace: {trace_info}")

with trace_col:
    st.markdown('<div class="section-label">// Trayceability</div>', unsafe_allow_html=True)
    st.markdown('<div class="red-divider"></div>', unsafe_allow_html=True)

    if not st.session_state['trace_log']:
        st.markdown('<p style="font-family: DM Mono, monospace; font-size: 11px; color: #444;">No queries yet. Start a conversation to see agent logic here.</p>', unsafe_allow_html=True)
    else:
        for entry in reversed(st.session_state['trace_log']):
            agent_color = "#E8302A" if "Rag" in entry['agent_used'] else "#4fc3f7"

            sql_block = ""
            if entry.get('sql'):
                sql_block = f"<div style='margin-top:10px;'><span style='font-family:DM Mono,monospace;font-size:9px;color:#555;letter-spacing:2px;text-transform:uppercase;'>SQL Used</span><pre style='background:#0D0D0D;padding:10px;font-size:10px;color:#aaa;margin-top:4px;overflow-x:auto;white-space:pre-wrap;'>{entry['sql']}</pre></div>"

            citations_block = ""
            if entry.get('citations'):
                cites = "".join([
                    f"<div style='margin-bottom:8px;'><span style='color:#E8302A;font-family:DM Mono,monospace;font-size:9px;'>dist: {round(d,3)}</span><div style='font-size:11px;color:#777;margin-top:2px;line-height:1.5;'>{doc[:140]}...</div></div>"
                    for doc, d in entry['citations']
                ])
                citations_block = f"<div style='margin-top:10px;'><span style='font-family:DM Mono,monospace;font-size:9px;color:#555;letter-spacing:2px;text-transform:uppercase;'>Citations</span><div style='margin-top:6px;'>{cites}</div></div>"

            st.markdown(f"""
            <div class="trace-card">
                <div class="trace-q">Q: {entry['question']}</div>
                <div class="trace-agent">Agent: <span style="color:{agent_color};">{entry['agent_used']}</span></div>
                <div class="trace-body">{entry['explanation']}</div>
                {sql_block}
                {citations_block}
            </div>
            """, unsafe_allow_html=True)

        if st.button("Clear Trace Log"):
            st.session_state['trace_log'] = []
            st.rerun()