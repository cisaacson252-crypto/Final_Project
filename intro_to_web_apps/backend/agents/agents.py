from abc import ABC, abstractmethod
import json
import os
import tempfile
import traceback
from sqlite3 import connect
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from pydantic import BaseModel

from intro_to_web_apps.backend.agents.llms import BaseLLM
from intro_to_web_apps.backend.rag.vector_databases import BaseVectorDB

# ==============================================================
# Strategy Patterns
# ==============================================================

class BaseRAGAgent(ABC):
    """
    Abstract base class for Retrieval-Augmented Generation (RAG) agents.
    
    RAG agents combine vector search with language models to provide
    context-aware responses based on a knowledge base.
    """
    @abstractmethod
    def __init__(self, llm: BaseLLM, vectordb: BaseVectorDB) -> None:
        """
        Initialize a RAG agent.
        
        Args:
            llm: The language model to use for generating responses
            vectordb: The vector database to use for retrieving context
        """
        pass
    
    @abstractmethod
    def query(self, prompt: str, k: int = 5, max_distance: float = 0.65, 
              show_citations: bool = False) -> Optional[str]:
        """
        Query the RAG agent with a prompt.
        
        Args:
            prompt: The user's question or prompt
            k: Number of documents to retrieve from the vector database
            max_distance: Maximum semantic distance threshold for retrieved documents
            show_citations: Whether to display the retrieved documents as citations
            
        Returns:
            The generated response or None if confidence is too low
        """
        pass

class BaseStructuredDataAgent(ABC):
    """
    Abstract base class for structured data agents.
    
    Structured data agents interact with databases or structured data sources
    to answer questions by generating and executing queries.
    """
    @abstractmethod
    def __init__(self, llm: BaseLLM, database_url: str, 
                 db_desc: Optional[str] = None, include_detail: bool = True, **kwargs) -> None:
        """
        Initialize a structured data agent.
        
        Args:
            llm: The language model to use for generating responses
            database_url: URL or path to the database
            db_desc: Optional description of the database
            include_detail: Whether to include detailed schema information
            **kwargs: Additional keyword arguments
        """
        pass
    
    @abstractmethod
    def query(self, prompt: str, view_sql: bool = False, retries: int = 0) -> Optional[str]:
        """
        Query the structured data agent with a prompt.
        
        Args:
            prompt: The user's question or prompt
            view_sql: Whether to display the generated SQL query
            retries: Number of retry attempts if the query fails
            
        Returns:
            The generated response or None if the query fails
        """
        pass
    
class BaseMultiAgent(ABC):
    """
    Abstract base class for multi-agent systems.
    
    Multi-agent systems coordinate multiple specialized agents to handle
    different types of questions or tasks.
    """
    @abstractmethod
    def __init__(self, llm: BaseLLM, 
                 agent_names: List[str],
                 agents: List[Union[BaseRAGAgent, BaseStructuredDataAgent]], 
                 agent_descriptions: List[str],
                 agent_query_kwargs: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize a multi-agent system.
        
        Args:
            llm: The language model to use for agent selection
            agent_names: Names of the available agents
            agents: List of agent instances
            agent_descriptions: Descriptions of each agent's capabilities
            agent_query_kwargs: Optional keyword arguments for each agent's query method
        """
        pass
    
    @abstractmethod
    def query(self, prompt: str, show_logic: bool = False) -> Optional[str]:
        """
        Query the multi-agent system with a prompt.
        
        Args:
            prompt: The user's question or prompt
            show_logic: Whether to display the agent selection logic
            
        Returns:
            The generated response from the selected agent
        """
        pass

# ==============================================================
# ChromaDB RAG Agent
# ==============================================================

class ChromaAgent(BaseRAGAgent):
    """
    RAG agent implementation using ChromaDB as the vector database.
    
    This agent retrieves relevant documents from a ChromaDB collection
    and uses an LLM to generate responses based on the retrieved context.
    """
    def __init__(self, llm: BaseLLM, vectordb: BaseVectorDB) -> None:
        """
        Initialize a ChromaDB RAG agent.
        
        Args:
            llm: The language model to use for generating responses
            vectordb: The ChromaDB vector database to use for retrieving context
            
        Raises:
            ValueError: If no collection is attached to the vector database
        """
        self.llm_ = llm
        self.vectordb_ = vectordb

        # Check if collection is attached
        if self.vectordb_.collection_ is None:
            raise ValueError("No collection attached to the vector database.")
        
    def query(self, prompt: str, k: int = 5, max_distance: float = 0.65, 
              show_citations: bool = False) -> Optional[str]:
        """
        Query the ChromaDB RAG agent with a prompt.
        
        Args:
            prompt: The user's question or prompt
            k: Number of documents to retrieve from the vector database
            max_distance: Maximum semantic distance threshold for retrieved documents
            show_citations: Whether to display the retrieved documents as citations
            
        Returns:
            The generated response or None if confidence is too low
            
        Note:
            Higher distance values indicate weaker matches. If the minimum
            distance exceeds max_distance, the function returns None.
        """
        # Retrieve top k results
        self.docs_, self.distances_ = self.vectordb_.retrieve(question=prompt, k=k)

        if show_citations:
            citations = [f"Citation {i+1} (Distance: {round(self.distances_[i], 2)}):\n {doc}" 
                         for i, doc in enumerate(self.docs_)]
            citations_str = "\n\n".join(citations)
            print(f"Citations:\n{citations_str}\n")

        # Check if retrieval results are below confidence threshold
        mindist = min(self.distances_)
        if mindist > max_distance:
            msg = (
                "⚠️ I'm sorry, but I don't have high confidence in answering this question. "
                "The retrieved context does not seem relevant enough.\n\n"
                f"🔹 Minimum distance found: {round(mindist, 2)} (higher values indicate weaker matches)\n"
                "🛠️ Try increasing `max_distance` or refining your query."
            )
            print(msg)
            return None

        # Construct system role and context
        system_prompt = (
            "You are an intelligent AI assistant answering a question generated from retrieval augmented generation. "
            "Use the provided context to answer "
            "the question accurately and concisely. If the context is insufficient, state that "
            "you don't have enough information rather than making assumptions.\n"
            "NEVER reference the context you received, simply use it to answer the question.\n\n"
        )

        # Format retrieved documents into context
        context_str = "\n\n".join(f"- {doc}" for doc in self.docs_)

        # Final formatted prompt
        self.prompt_ = f"{system_prompt}### Context ###\n{context_str}\n\n### Question ###\n{prompt}"

        # Query the LLM
        self.response_ = self.llm_.query(self.prompt_)
        return self.response_
    

# ==============================================================
# SQLite Structured Data Agent
# ==============================================================

class SQLResponse(BaseModel):
    """
    Pydantic model for structured SQL responses from the LLM.
    
    Attributes:
        sql_query: The generated SQL query
        explanation: An explanation of what the query does
    """
    sql_query: str
    explanation: str

class SQLiteAgent(BaseStructuredDataAgent):
    """
    Structured data agent for SQLite databases.
    
    This agent analyzes database schema, generates SQL queries based on
    natural language questions, and interprets the results.
    """
    def __init__(self, llm: BaseLLM, database_url: Union[str, URL], 
                 db_desc: Optional[str] = None, include_detail: bool = True, **kwargs) -> None:
        """
        Initialize a SQLite agent.
        
        Args:
            llm: The language model to use for generating responses
            database_url: URL or path to the SQLite database
            db_desc: Optional description of the database
            include_detail: Whether to include detailed schema information
            **kwargs: Additional keyword arguments
            
        Raises:
            ValueError: If connection to the database fails
        """
        self.llm_ = llm
        self.db_desc_ = db_desc
        self.include_detail_ = include_detail
        try:
            self.engine_ = connect(database_url)
        except Exception as e:
            raise ValueError(f"Could not connect to database: {e}")

        self._build_schema()

    def _build_schema(self) -> None:
        """
        Build a schema representation of the SQLite database.
        
        This method queries the database to extract table and view information,
        column names, data types, and optionally min/max values or example values.
        The schema is stored as a JSON string in self.schema_json_.
        """
        schema: Dict[str, Dict[str, Any]] = {}
        
        # Iterate to understand database
        itm = ['table','view']
        for ix in itm:
            tables = list(pd.read_sql(f"SELECT tbl_name FROM sqlite_master WHERE type = '{ix}'", self.engine_)['tbl_name'])
            if len(tables) != 0:
                td = {}
                for t in tables:
                    temp = pd.read_sql(f"SELECT * FROM {t} LIMIT 1", self.engine_)
                    cols = list(temp.columns)
                    types = temp.dtypes.apply(lambda x: str(x)).to_list()
                    cd = {}
                    for i, c in enumerate(cols):
                        if self.include_detail_ == True:
                            if 'int' in types[i] or 'float' in types[i] or 'date' in types[i]:
                                unq = pd.read_sql(f"SELECT MIN([{c}]) AS mn, MAX([{c}]) AS mx FROM {t} LIMIT 1", self.engine_)
                                mn = unq['mn'].values[0]
                                mx = unq['mx'].values[0]
                                cd[c] = {'datatype':types[i],
                                        'min':str(mn),
                                        'max':str(mx)}
                            else:
                                unq = pd.read_sql(f"SELECT DISTINCT [{c}] AS vls FROM {t}", self.engine_)
                                cd[c] = {'datatype':types[i],
                                         'example values':unq['vls'].to_list()[:20]}
                        else:
                            cd[c] = {'datatype':types[i]}
                    td[t] = {'columns':cd}
                schema[ix+'s'] = td
        # Dump to json to package in prompt
        self.schema_json_ = json.dumps(schema)

    def query(self, prompt: str, view_sql: bool = False, retries: int = 0) -> Optional[str]:
        """
        Query the SQLite database using natural language.
        
        Args:
            prompt: The user's question or prompt
            view_sql: Whether to display the generated SQL query
            retries: Number of retry attempts if the query fails
            
        Returns:
            The generated response or None if all query attempts fail
            
        Note:
            If a query fails and retries > 0, the agent will attempt to
            generate a new SQL query with error feedback from the previous attempt.
        """
        system_prompt = ("You are an expert in converting real world questions into SQL queries.\n"
                         "Your job is to take the question below and use the provided database architecture to convert the question into a SQLite query.\n\n"
                         "You are to respond both a SQL query required to answer the provided question and a short explanation of the query.\n"
                         "When you generate the SQL query, make sure to return it with proper syntax for SQLite and make it legible.\n"
                         'Example Question: "I want to know every type of car that was sold"\n'
                         'Example Response: "SELECT DISTINCT car_type FROM Car_Sales"\n\n'
                         'Example Explanation: "The Car_Sales table has information on the total units of sales, including the brands of the car."') 
        
        if self.db_desc_ is not None:
            query = ("Given the following SQLite database description and architecture, please answer the following question:\n\n"
                     f"Database Description:\n{self.db_desc_}\n\nDatabase Architecture:\n{self.schema_json_}\n\n"
                     f"Question:\n{prompt}")
        else:
            query = ("Given the following SQLite database architecture, please answer the following question:\n\n"
                     f"Database Architecture:\n{self.schema_json_}\n\n"
                     f"Question:\n{prompt}")
        
        attempts = 0
        error_context = ""
        
        while attempts <= retries:
            # If this is a retry, add error context to the query
            if attempts > 0:
                retry_system_prompt = system_prompt + "\n\nYour previous SQL query failed. Please fix the issues and try again."
                retry_query = query + f"\n\nPrevious failed SQL query:\n{self.response_.sql_query}\n\nError details:\n{error_context}\n\nDatabase Architecture:\n{self.schema_json_}"
                self.response_ = self.llm_.structured_query(response_format=SQLResponse, prompt=retry_query, system_prompt=retry_system_prompt)
            else:
                self.response_ = self.llm_.structured_query(response_format=SQLResponse, prompt=query, system_prompt=system_prompt)

            if view_sql == True:
                print(f"Executed SQL:\n{self.response_.sql_query}\n\nSQL Explanation:\n{self.response_.explanation}\n")
            try:
                answer = pd.read_sql(self.response_.sql_query, self.engine_)
                prompt = ("Given the following question, supporting data, and SQL generated to answer the question, "
                          "Please provide a concise answer to the question.\n\n"
                         f"Question:\n{prompt}\n\nSupporting Data:\n{answer.to_json()}\n\nSQL Used:\n{self.response_.sql_query}")
                self.response_ = self.llm_.query(prompt)             
                return self.response_
            except Exception as e:
                error_trace = traceback.format_exc()
                error_context = f"{str(e)}\n\n{error_trace}"
                print(error_trace)
                print("The generated SQL failed to properly query the database!\n\n", self.response_.sql_query)
                
                # If we've exhausted our retries, return None
                if attempts >= retries:
                    return None
            
            attempts += 1
        
        return None

        # Create a connection to the SQLite database
        conn = connect(dbpath)

        # For each sheet in the Excel file, upload the data to the SQLite database
        def clean_table_name(name: str) -> str:
            """Clean table name to be SQLite compatible."""
            # Replace any non-alphanumeric characters (except underscores) with underscores
            import re
            cleaned = re.sub(r'[^\w\s]', '_', name)
            # Replace spaces with underscores and ensure no double underscores
            cleaned = re.sub(r'\s+', '_', cleaned)
            cleaned = re.sub(r'_+', '_', cleaned)
            # Remove leading/trailing underscores
            cleaned = cleaned.strip('_')
            return cleaned

        if 'xls' in ftype.lower():
            for sheet in pd.read_excel(database_url, sheet_name=None):
                data = pd.read_excel(database_url, sheet_name=sheet)
                table_name = clean_table_name(sheet)
                data.to_sql(table_name, conn, index=False, if_exists='replace')
        else:
            data = pd.read_csv(database_url)
            tab = database_url[database_url.rfind('\\')+1:-4]
            table_name = clean_table_name(tab)
            data.to_sql(table_name, conn, index=False, if_exists='replace')

        conn.commit()
        conn.close()     

        # Convert classes
        newmodel = SQLiteAgent(self.llm_, dbpath, db_desc=self.db_desc_, include_detail=self.include_detail_)
        self.__class__ = SQLiteAgent
        self.__dict__ = newmodel.__dict__

    def query(self, prompt: str, view_sql: bool = False, retries: int = 0) -> Optional[str]:
        """
        This method is not used as the instance is transformed into a SQLiteAgent.
        
        The actual query method used will be the one from SQLiteAgent.
        """
        pass


# ==============================================================
# Vanilla Multi-Agent
# ==============================================================

class ConductorResponse(BaseModel):
    """
    Pydantic model for structured agent selection responses from the LLM.
    
    Attributes:
        agent_integer: The index of the selected agent
        explanation: An explanation of why this agent was selected
    """
    agent_integer: int
    explanation: str

class MultiAgent(BaseMultiAgent):
    """
    Multi-agent system that selects the most appropriate agent for a given query.
    
    This implementation uses an LLM to determine which specialized agent
    should handle a particular question based on agent descriptions.
    """
    def __init__(self, llm: BaseLLM, 
                 agent_names: List[str],
                 agents: List[Union[BaseRAGAgent, BaseStructuredDataAgent]], 
                 agent_descriptions: List[str],
                 agent_query_kwargs: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize a multi-agent system.
        
        Args:
            llm: The language model to use for agent selection
            agent_names: Names of the available agents
            agents: List of agent instances
            agent_descriptions: Descriptions of each agent's capabilities
                Agent descriptions should follow: "can answer questions related to:" 
                    ex: "employee demographics, locations and departments of work, salaries, names, and hr contacts"
            agent_query_kwargs: Optional keyword arguments for each agent's query method
                a. ChromaAgent: {"k": 5, "max_distance": 0.5, "show_citations": False} 
                b. SQLiteAgent: {'show_sql':False}
                c. ExcelAgent: {'show_sql':False}
            
        Raises:
            AssertionError: If the input lists don't have the same length
        """
        if agent_query_kwargs is None or agent_query_kwargs == {}:
            assert len(agent_names) == len(agents) == len(agent_descriptions), "All lists must be the same length"
        else:
            assert len(agent_names) == len(agents) == len(agent_descriptions) == len(agent_query_kwargs), "All lists must be the same length"

        self.llm_ = llm
        self.agent_names_ = agent_names
        self.agents_ = agents
        self.agent_descriptions_ = agent_descriptions
        self.agent_types_ = [type(agent).__name__ for agent in agents]
        if agent_query_kwargs is None or agent_query_kwargs == {}:
            self.agent_query_kwargs_ = [{} for i in range(len(agent_names))]
        else:
            self.agent_query_kwargs_ = agent_query_kwargs
        
        self.system_prompt_ = ("You are responsible for determining the most appropriate agent for a given question "
                              "based on the category descriptions. Each agent will be given a corresponding integer, "
                              "and your response include the integer of the agent that best matches the question and a short explanation why you selected that agent.\n\n"
                              "Example:\nAgent 0 (Text_Agent): Qualitative questions that requires text mining related to: sports history, sport philosophy, and sports rules.\n"
                              "Agent 1 (Database_Agent): Quantitative questions that could be answered with a SQL query related to: sports statistics and athelete data.\n"
                              "Question: How many points did Kobe Bryant score over his career?\nAnswer: 1\n"
                              "Explanation: The question is about statistics (points scored) and is therefore a quantitative question, matching the agent 1 (Database_Agent).")

        self.prompt_ = "Use the following category descriptions to answer the following question:\n\n"

        type_tags = {'ChromaAgent':'Qualitative questions that requires text mining related to: ',
                     'SQLiteAgent':'Quantitative questions that could be answered with a SQL query related to: '}
        
        for i, d in enumerate(self.agent_descriptions_):
            desc = d.strip()
            if desc[-1] != '.':
                desc += '.'
            self.prompt_ += f"Category {str(i)} ({self.agent_names_[i]}):\n{type_tags[self.agent_types_[i]]}{desc}\n"
        self.prompt_ += '\nQuestion:\n'

    def query(self, prompt: str, show_logic: bool = False) -> Optional[str]:
        """
        Query the multi-agent system with a prompt.
        
        This method:
        1. Uses the LLM to select the most appropriate agent
        2. Forwards the query to the selected agent
        3. Returns the response from that agent
        
        Args:
            prompt: The user's question or prompt
            show_logic: Whether to display the agent selection logic
            
        Returns:
            The generated response from the selected agent
        """
        # Retrieve category
        query = self.prompt_ + prompt
        response = self.llm_.structured_query(response_format=ConductorResponse, prompt=query, system_prompt=self.system_prompt_)
        agent_int = response.agent_integer
        explanation = response.explanation
        
        if show_logic == True:
            print(f"Agent selected: {self.agent_names_[agent_int]}\nExplanation:\n{explanation}\n")

        # Pass question to proper agent
        response = self.agents_[agent_int].query(prompt, **self.agent_query_kwargs_[agent_int])
        return response
