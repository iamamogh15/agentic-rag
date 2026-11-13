# %% [markdown]
# # Agentic RAG over financial documents (LangChain / LangGraph)
# Install:
# uv add torch sentence-transformers langchain langchain-classic langchain-groq \
#        langchain-huggingface langchain-community langchain-text-splitters \
#        langgraph rank_bm25 python-dotenv pydantic

# %% Imports
import csv
import time
import unicodedata
from pathlib import Path
from typing import List, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.tools import StructuredTool
from langchain_core.messages import ToolMessage

from langchain_community.document_loaders import TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.prebuilt import create_react_agent

# %% Step 1: Configuring the models
load_dotenv()  # GROQ_API_KEY and HUGGINGFACEHUB_API_TOKEN in .env

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0, max_tokens=2048, max_retries=6)
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# %% Step 2: Loading the corpus and building per-document indexes
DATA_DIR = Path("data")

SOURCES = {
    "ten_k": "acme_10k_excerpt.md",
    "earnings_call": "acme_earnings_call_q3.md",
    "risk_policy": "firm_risk_policy.md",
}

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)


def load_file(path: Path):
    return TextLoader(str(path), encoding="utf-8").load()


def load_indices() -> dict[str, InMemoryVectorStore]:
    indices = {}
    for key, filename in SOURCES.items():
        docs = load_file(DATA_DIR / filename)
        chunks = splitter.split_documents(docs)
        for c in chunks:
            c.metadata["source_key"] = key
        indices[key] = InMemoryVectorStore.from_documents(chunks, embeddings)
    return indices


def load_all_chunks() -> list:
    all_chunks = []
    for key, filename in SOURCES.items():
        chunks = splitter.split_documents(load_file(DATA_DIR / filename))
        for c in chunks:
            c.metadata["source_key"] = key
        all_chunks.extend(chunks)
    return all_chunks


def build_combined_store() -> InMemoryVectorStore:
    return InMemoryVectorStore.from_documents(load_all_chunks(), embeddings)

# %% Helpers: question, logging, report writer
COMPARISON_QUESTION = (
    "Does Acme Robotics' most recent quick ratio meet the firm's minimum "
    "quick-ratio requirement under its liquidity risk policy?"
)

REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)
_comparison_log: list[dict] = []


def _log_comparison(title: str, response: str, nodes: list | None = None) -> None:
    _comparison_log.append({"title": title, "response": response, "nodes": nodes or []})
    print(f"\n=== {title} ===\n{response}\n")


def report_step(filename: str, title: str, description: str, entries: list[dict]) -> None:
    lines = [title, "=" * len(title), "", description, ""]
    for e in entries:
        lines.append(f"{e.get('label', 'Question')}: {e['question']}")
        lines.append("")
        lines.append(f"Response:\n{e['response']}")
        if e.get("note"):
            lines.append(f"\nNote: {e['note']}")
        if e.get("nodes"):
            lines.append("\nRetrieved chunks:")
            for i, d in enumerate(e["nodes"], 1):
                src = d.metadata.get("source_key", d.metadata.get("source", "?"))
                lines.append(f"  [{i}] ({src}) {d.page_content[:300]}...")
        lines.append("\n" + "-" * 60 + "\n")
    (REPORT_DIR / filename).write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {REPORT_DIR / filename}")

# %% Step 3: Naive RAG baseline
RAG_PROMPT = ChatPromptTemplate.from_template(
    "Answer the question using only the context below. "
    "If the answer is not in the context, say so.\n\n"
    "Context:\n{context}\n\nQuestion: {question}"
)


def naive_query(question: str, retriever) -> str:
    docs = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs)
    chain = RAG_PROMPT | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": question})


def step_3_naive_rag() -> None:
    combined = build_combined_store()
    retriever = combined.as_retriever(search_kwargs={"k": 3})

    single_fact = "What was Acme's full-year 2024 total revenue?"
    single_answer = naive_query(single_fact, retriever)
    comparative = naive_query(COMPARISON_QUESTION, retriever)
    _log_comparison("Step 3 — Naive RAG (single combined index, top-k=3)", comparative)

    report_step(
        "Step_3_naive_RAG_baseline.txt",
        "Step 3 — Naive RAG baseline",
        "One combined index over all three documents, a single top-k retrieval "
        "pass, no routing or self-checking.",
        [
            {"question": single_fact, "response": single_answer},
            {"question": COMPARISON_QUESTION, "response": comparative},
        ],
    )

# %% Step 4: Routing
TOOL_SPECS = {
    "ten_k": {
        "name": "annual_report_10k",
        "description": (
            "Acme Robotics' FY2024 Form 10-K. Audited annual figures: full-year "
            "revenue, margins, balance-sheet items, year-end current and quick "
            "ratios, credit-facility covenants, and business risk factors."
        ),
    },
    "earnings_call": {
        "name": "q3_earnings_call",
        "description": (
            "Acme Robotics' Q3 FY2024 earnings-call transcript. Quarterly figures "
            "and management commentary: Q3 revenue, Q3 quick ratio, inventory "
            "build, guidance updates, and analyst Q&A."
        ),
    },
    "risk_policy": {
        "name": "firm_risk_policy",
        "description": (
            "Birchwood Asset Management's internal liquidity/counterparty risk "
            "policy. Defines required thresholds (current ratio, quick ratio, "
            "net-debt/EBITDA), covenant-awareness rules, and concentration caps."
        ),
    },
}

ToolName = Literal["annual_report_10k", "q3_earnings_call", "firm_risk_policy"]


def build_query_engine_tools(indices: dict[str, InMemoryVectorStore]) -> dict[str, dict]:
    """Expose each document index as a described, callable tool."""
    return {
        spec["name"]: {
            "description": spec["description"],
            "retriever": indices[key].as_retriever(search_kwargs={"k": 3}),
        }
        for key, spec in TOOL_SPECS.items()
    }


class RouteChoice(BaseModel):
    tool_name: ToolName = Field(description="The single best tool for the question.")


ROUTER_PROMPT = ChatPromptTemplate.from_template(
    "Pick exactly ONE tool that best answers the question.\n\n"
    "Tools:\n{tool_list}\n\nQuestion: {question}"
)


def router_query(question: str, tools: dict[str, dict]) -> str:
    tool_list = "\n".join(f"- {n}: {t['description']}" for n, t in tools.items())
    selector = ROUTER_PROMPT | llm.with_structured_output(RouteChoice, method="json_schema")
    choice = selector.invoke({"tool_list": tool_list, "question": question})
    print(f"[router] {choice.tool_name} <- {question}")
    return naive_query(question, tools[choice.tool_name]["retriever"])


def step_4_router(indices: dict[str, InMemoryVectorStore]) -> None:
    tools = build_query_engine_tools(indices)

    routing_questions = [
        "What is the firm's minimum required quick ratio for a new long position?",
        "What was Acme's Q3 quick ratio?",
        "What credit-facility covenants does Acme disclose in its 10-K?",
    ]
    entries = [{"question": q, "response": router_query(q, tools)} for q in routing_questions]
    comparative = router_query(COMPARISON_QUESTION, tools)
    entries.append({"question": COMPARISON_QUESTION, "response": comparative})
    _log_comparison("Step 4 — Router (picks ONE source via structured output)", comparative)

    report_step(
        "Step_4_router_query_engine.txt",
        "Step 4 — Router query engine",
        "An LLM selector picks exactly one per-document tool per query.",
        entries,
    )

# %% Step 5: Query decomposition (sub-questions)
class SubQuestion(BaseModel):
    tool_name: ToolName
    sub_question: str = Field(description="A self-contained question for this tool.")


class SubQuestionPlan(BaseModel):
    sub_questions: List[SubQuestion]


DECOMPOSE_PROMPT = ChatPromptTemplate.from_template(
    "Break the user question into the smallest set of sub-questions. "
    "Each sub-question must be answerable by ONE tool and be self-contained.\n\n"
    "Tools:\n{tool_list}\n\nQuestion: {question}"
)

SYNTHESIS_PROMPT = ChatPromptTemplate.from_template(
    "Answer the original question using only the sub-question answers below.\n\n"
    "Sub-question answers:\n{qa_pairs}\n\nOriginal question: {question}"
)


def subquestion_query(question: str, tools: dict[str, dict]) -> str:
    tool_list = "\n".join(f"- {n}: {t['description']}" for n, t in tools.items())
    planner = DECOMPOSE_PROMPT | llm.with_structured_output(SubQuestionPlan, method="json_schema")
    plan = planner.invoke({"tool_list": tool_list, "question": question})

    qa_pairs = []
    for sq in plan.sub_questions:
        print(f"[sub-question] {sq.tool_name} <- {sq.sub_question}")
        answer = naive_query(sq.sub_question, tools[sq.tool_name]["retriever"])
        qa_pairs.append(f"Q: {sq.sub_question}\nA: {answer}")

    chain = SYNTHESIS_PROMPT | llm | StrOutputParser()
    return chain.invoke({"qa_pairs": "\n\n".join(qa_pairs), "question": question})


def step_5_subquestion(indices: dict[str, InMemoryVectorStore]) -> None:
    tools = build_query_engine_tools(indices)

    spanning = (
        "How did Acme's quick ratio change from year-end 2024 to Q3, and does the "
        "Q3 level breach the firm's risk policy?"
    )
    spanning_answer = subquestion_query(spanning, tools)
    comparative = subquestion_query(COMPARISON_QUESTION, tools)
    _log_comparison("Step 5 — Sub-question decomposition (fans out to all sources)", comparative)

    report_step(
        "Step_5_subquestion_decomposition.txt",
        "Step 5 — Sub-question decomposition",
        "An LLM splits a cross-source question into per-tool "
        "sub-questions, answers each, then synthesises.",
        [
            {"question": spanning, "response": spanning_answer},
            {"question": COMPARISON_QUESTION, "response": comparative},
        ],
    )

# %% Step 6: Self-correcting agent
AGENT_SYSTEM_PROMPT = (
    "You are a buy-side compliance analyst. Answer using ONLY the document tools "
    "provided. First work out what evidence the question actually requires: for a "
    "policy or compliance check that usually means both the relevant figure from "
    "the issuer's filings and the matching threshold from the firm's risk policy, "
    "but a simpler factual question may need only a single figure. Before "
    "concluding, make sure you have gathered every piece of evidence the question "
    "requires; if one tool call does not give you everything, call another. If a "
    "required figure cannot be found, say so explicitly rather than estimating. "
    "Cite the source document and reporting period for every figure you use."
)


def _make_agent_tool(name: str, spec: dict) -> StructuredTool:
    retriever = spec["retriever"]

    def _run(query: str):
        docs = retriever.invoke(query)
        context = "\n\n".join(d.page_content for d in docs)
        answer = (RAG_PROMPT | llm | StrOutputParser()).invoke(
            {"context": context, "question": query}
        )
        return answer, docs  # (content shown to agent, artifact we collect)

    return StructuredTool.from_function(
        func=_run,
        name=name,
        description=spec["description"],
        response_format="content_and_artifact",
    )


def build_agent_tools(tools: dict[str, dict]) -> list[StructuredTool]:
    return [_make_agent_tool(name, spec) for name, spec in tools.items()]


def _run_agent_sync(agent, task: str):
    """Run the agent and collect the chunks it retrieved. Returns (response, nodes)."""
    result = agent.invoke(
        {"messages": [("user", task)]},
        config={"recursion_limit": 25},
    )
    nodes = []
    for msg in result["messages"]:
        if isinstance(msg, ToolMessage) and msg.artifact:
            nodes.extend(msg.artifact)
    response = result["messages"][-1].content
    return response, nodes


def step_6_agent(indices: dict[str, InMemoryVectorStore]) -> None:
    agent = create_react_agent(
        llm,
        build_agent_tools(build_query_engine_tools(indices)),
        prompt=AGENT_SYSTEM_PROMPT,
    )

    task = (
        "Can we open a new long position in Acme Robotics under the firm's liquidity "
        "policy? Check the quick-ratio rule against Acme's most recent disclosed "
        "quick ratio."
    )
    task_response, task_nodes = _run_agent_sync(agent, task)
    comparative, comp_nodes = _run_agent_sync(agent, COMPARISON_QUESTION)
    _log_comparison(
        "Step 6 — Self-correcting agent (re-queries tools until sufficient)",
        comparative,
        nodes=comp_nodes,
    )

    report_step(
        "Step_6_self_correcting_agent.txt",
        "Step 6 — Self-correcting agent",
        "A LangGraph ReAct agent calls the document tools, judges whether it has both the "
        "figure and the threshold, and re-queries when it does not.",
        [
            {"label": "Task", "question": task, "response": task_response, "nodes": task_nodes},
            {"question": COMPARISON_QUESTION, "response": comparative, "nodes": comp_nodes},
        ],
    )

# %% Step 7: Hybrid search (dense + BM25)
def build_hybrid_retriever() -> EnsembleRetriever:
    """Dense + sparse (BM25) retrievers over one combined corpus, fused with RRF."""
    all_chunks = load_all_chunks()

    dense_store = InMemoryVectorStore.from_documents(all_chunks, embeddings)
    vector_retriever = dense_store.as_retriever(search_kwargs={"k": 5})
    bm25_retriever = BM25Retriever.from_documents(all_chunks, k=5)

    return EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.5, 0.5],
    )


def step_7_hybrid() -> None:
    hybrid_retriever = build_hybrid_retriever()

    comparative = naive_query(COMPARISON_QUESTION, hybrid_retriever)
    _log_comparison("Step 7 — Hybrid dense + sparse retrieval (fused recall)", comparative)

    report_step(
        "Step_7_hybrid_retrieval.txt",
        "Step 7 — Hybrid dense + sparse retrieval",
        "A dense vector retriever and a BM25 sparse retriever over the same chunks, "
        "fused with reciprocal rank fusion.",
        [{"question": COMPARISON_QUESTION, "response": comparative}],
    )

# %% Step 8: Reranking
def build_reranking_retriever(store: InMemoryVectorStore, top_k: int = 6, top_n: int = 2):
    base_retriever = store.as_retriever(search_kwargs={"k": top_k})
    cross_encoder = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    reranker = CrossEncoderReranker(model=cross_encoder, top_n=top_n)
    return ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=base_retriever,
    )


def step_8_rerank(indices: dict[str, InMemoryVectorStore]) -> None:
    reranked_retriever = build_reranking_retriever(indices["ten_k"], top_k=6, top_n=2)

    q = "What was Acme's gross margin in fiscal 2024 and why did it expand?"
    report_step(
        "Step_8_reranking.txt",
        "Step 8 — Reranking",
        "Retrieve a broad top-6 from the 10-K, then a cross-encoder reranker "
        "re-scores and keeps the best 2 chunks.",
        [{"question": q, "response": naive_query(q, reranked_retriever)}],
    )

# %% Step 9: Hybrid search + reranking
def step_9_hybrid_rerank() -> None:
    cross_encoder = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    hybrid_rerank_retriever = ContextualCompressionRetriever(
        base_compressor=CrossEncoderReranker(model=cross_encoder, top_n=3),
        base_retriever=build_hybrid_retriever(),
    )

    comparative = naive_query(COMPARISON_QUESTION, hybrid_rerank_retriever)
    _log_comparison("Step 9 — Hybrid + rerank (recall-then-precision)", comparative)

    report_step(
        "Step_9_hybrid_rerank.txt",
        "Step 9 — Hybrid search + reranking",
        "The Step 7 hybrid retriever for recall, with a Step 8 cross-encoder reranker "
        "attached to trim the fused pool to the best chunks for precision.",
        [{"question": COMPARISON_QUESTION, "response": comparative}],
    )

# %% Step 10: Evaluating the answer (faithfulness / relevancy triad)
class FaithfulnessGrade(BaseModel):
    passing: bool = Field(description="True if every claim in the answer is supported by the context.")
    reasoning: str


class RelevancyGrade(BaseModel):
    score: float = Field(ge=0, le=1, description="Score from 0 to 1.")
    reasoning: str


FAITHFULNESS_PROMPT = ChatPromptTemplate.from_template(
    "Check whether the answer is fully supported by the context. "
    "Pass only if every claim in the answer is backed by the context.\n\n"
    "Context:\n{context}\n\nAnswer:\n{answer}"
)
ANSWER_REL_PROMPT = ChatPromptTemplate.from_template(
    "Score from 0 to 1 how well the answer addresses the question "
    "(1 = fully answers it, 0 = irrelevant).\n\n"
    "Question: {question}\n\nAnswer:\n{answer}"
)
CONTEXT_REL_PROMPT = ChatPromptTemplate.from_template(
    "Score from 0 to 1 how relevant the retrieved context is to the question "
    "(1 = contains what is needed, 0 = unrelated).\n\n"
    "Question: {question}\n\nContext:\n{context}"
)


def step_10_evaluate(indices: dict[str, InMemoryVectorStore]) -> None:
    retriever = indices["ten_k"].as_retriever(search_kwargs={"k": 3})
    q = "What was Acme's net income per diluted share in fiscal 2024?"

    docs = retriever.invoke(q)
    context = "\n\n".join(d.page_content for d in docs)
    response = (RAG_PROMPT | llm | StrOutputParser()).invoke(
        {"context": context, "question": q}
    )

    f_result = (
        FAITHFULNESS_PROMPT | llm.with_structured_output(FaithfulnessGrade, method="json_schema")
    ).invoke({"context": context, "answer": response})
    ar_result = (
        ANSWER_REL_PROMPT | llm.with_structured_output(RelevancyGrade, method="json_schema")
    ).invoke({"question": q, "answer": response})
    cr_result = (
        CONTEXT_REL_PROMPT | llm.with_structured_output(RelevancyGrade, method="json_schema")
    ).invoke({"question": q, "context": context})

    report_step(
        "Step_10_evaluation.txt",
        "Step 10 — RAG evaluation",
        "The evaluation triad: a faithfulness check that the answer is grounded "
        "in the retrieved context, an answer relevancy score against the question, "
        "and a context relevancy score for the retrieved chunks against the question.",
        [
            {
                "question": q,
                "response": response,
                "note": (
                    f"Faithful to sources? {f_result.passing} | "
                    f"Answer relevancy (0-1): {ar_result.score} | "
                    f"Context relevancy (0-1): {cr_result.score}"
                ),
            }
        ],
    )

# %% Step 11: Test-set evaluation across all methods
PASS_WORDS = ["exceed", "above", "pass", "satisf", "not breach", "not block", "meets", "meet ", "≥", ">="]

TEST_SET = [
    {"q": "What was Acme's full-year 2024 total revenue?", "must": [["612.4"]]},
    {"q": "What was Acme's net income per diluted share in fiscal 2024?", "must": [["1.92"]]},
    {"q": "What was Acme's gross margin in fiscal 2024?", "must": [["43.1"]]},
    {"q": "What was Acme's quick ratio at the end of Q3?", "must": [["1.18"]]},
    {"q": "What is the firm's minimum required quick ratio for a new long position?", "must": [["1.00", "1.0"]]},
    {"q": "Does Acme's Q3 quick ratio meet the firm's minimum quick-ratio requirement?",
     "must": [["1.18"], ["1.00", "1.0"], PASS_WORDS]},
    {"q": "Does Acme's year-end 2024 quick ratio meet the firm's minimum quick-ratio requirement?",
     "must": [["1.23"], ["1.00", "1.0"], PASS_WORDS]},
    {"q": "Does Acme's year-end 2024 current ratio meet the firm's current ratio threshold?",
     "must": [["1.61"], ["1.50", "1.5"], PASS_WORDS]},
    {"q": "What net-debt-to-EBITDA limit does the firm's policy set, and what covenant limit does Acme's credit facility have?",
     "must": [["3.5"], ["3.0"]]},
    {"q": "Acme's top three customers are what share of revenue, and what does the firm's policy say about issuers above its concentration threshold?",
     "must": [["41"], ["40"], ["3%", "3 %", "3 percent"]]},
]


def is_correct(answer: str, must: list[list[str]]) -> bool:
    a = unicodedata.normalize("NFKC", str(answer)).lower()
    return all(any(opt.lower() in a for opt in group) for group in must)


def build_methods(indices):
    naive_ret = build_combined_store().as_retriever(search_kwargs={"k": 3})
    tools = build_query_engine_tools(indices)
    hybrid_ret = build_hybrid_retriever()
    cross_encoder = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    rerank_ret = ContextualCompressionRetriever(
        base_compressor=CrossEncoderReranker(model=cross_encoder, top_n=3),
        base_retriever=build_hybrid_retriever(),
    )
    agent = create_react_agent(llm, build_agent_tools(tools), prompt=AGENT_SYSTEM_PROMPT)

    return {
        "3_naive": lambda q: naive_query(q, naive_ret),
        "4_router": lambda q: router_query(q, tools),
        "5_subquestion": lambda q: subquestion_query(q, tools),
        "6_agent": lambda q: _run_agent_sync(agent, q)[0],
        "7_hybrid": lambda q: naive_query(q, hybrid_ret),
        "9_hybrid_rerank": lambda q: naive_query(q, rerank_ret),
    }


def run_evaluation(indices, pause: float = 3.0):
    methods = build_methods(indices)
    rows = []
    for name, fn in methods.items():
        for i, t in enumerate(TEST_SET, 1):
            try:
                ans = fn(t["q"])
            except Exception as e:
                ans = f"ERROR: {e}"
            rows.append({"method": name, "q_no": i, "correct": is_correct(ans, t["must"]), "answer": str(ans)})
            time.sleep(pause)

    print(f"\n{'Method':<18}{'Score':>8}")
    for name in methods:
        ok = sum(r["correct"] for r in rows if r["method"] == name)
        print(f"{name:<18}{ok:>5}/{len(TEST_SET)}")

    with open(REPORT_DIR / "evaluation_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["method", "q_no", "correct", "answer"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {REPORT_DIR / 'evaluation_results.csv'}")
    return rows

# %% Run: build indexes and run the steps one by one
indices = load_indices()

# %%
step_3_naive_rag()

# %%
step_4_router(indices)

# %%
step_5_subquestion(indices)

# %%
step_6_agent(indices)

# %%
step_7_hybrid()

# %%
step_8_rerank(indices)

# %%
step_9_hybrid_rerank()

# %%
step_10_evaluate(indices)

# %% Step 11 run (about 70+ LLM calls)
rows = run_evaluation(indices)
