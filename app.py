import streamlit as st
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langgraph.graph import StateGraph
from typing import TypedDict, Annotated
import pymongo
from datetime import datetime
import uuid

# ✅ Initialize Google Gemini API
genai.configure(api_key="AIzaSyCCEUdNbJcvaT76FTjoWdqY1q3eWwRtQO8")  # Replace with your API key


# ✅ MongoDB Setup
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["agile_workflow_db"]
conversations_collection = db["conversations"]

# Create the saved_notes_collection if it doesn't exist
if "saved_notes" not in db.list_collection_names():
    db.create_collection("saved_notes")
saved_notes_collection = db["saved_notes"]

def query_gemini(role: str, prompt: str) -> str:
    try:
        model = genai.GenerativeModel("gemini-1.5-flash-latest")
        response = model.generate_content(f"You are a {role}. {prompt}")
        return response.text if response else "[No meaningful response received.]"
    except Exception as e:
        return f"[Error querying Gemini API: {str(e)}]"

# ✅ URL Summarization Function
def summarize_url(url: str) -> str:
    summary_prompt = f"""
    Extract and summarize key insights from the following URL:
    {url}

    Provide a **concise summary** of its main points.
    """
    return query_gemini("Summarizer", summary_prompt)

# ✅ Define Workflow State Schema
class WorkflowState(TypedDict):
    feature_request: Annotated[str, "single"]
    product_vision: Annotated[str, "single"]
    retrieved_knowledge: Annotated[str, "single"]
    backlog_priorities: Annotated[str, "single"]
    technical_feasibility: Annotated[str, "single"]
    ux_design: Annotated[str, "single"]
    execution_plan: Annotated[str, "single"]
    retrospective_analysis: Annotated[str, "single"]
    okr_insights: Annotated[str, "single"]

# ✅ Load and Vectorize Knowledge Base
try:
    loader = TextLoader("agile_knowledge_base.txt")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(texts, embeddings)
except Exception as e:
    st.error(f"[Error loading knowledge base: {str(e)}]")
    vector_store = None

# ✅ Define Workflow Nodes
def product_manager(state: WorkflowState) -> WorkflowState:
    prompt = f"""
    You are a **Product Manager** following SVPG Agile principles.
    - Feature Request: "{state['feature_request']}"

    Define the **product vision**:
    - What customer problem does it solve?
    - What is the key value proposition?
    - Prioritize based on **market demand, business goals, and usability**.
    """
    return {"product_vision": query_gemini("Product Manager", prompt)}

def rag_retrieval_agent(state: WorkflowState) -> WorkflowState:
    if vector_store:
        docs = vector_store.similarity_search(state["feature_request"], k=2)
        if docs:
            retrieved_text = "\n".join([doc.page_content for doc in docs])
            return {"retrieved_knowledge": retrieved_text}
    return {"retrieved_knowledge": "[No relevant knowledge found in the database.]"}

def safe_product_owner(state: WorkflowState) -> WorkflowState:
    prompt = f"""
    You are a **SAFe Product Owner** managing an Agile Release Train.
    - Feature Request: "{state['feature_request']}"
    - Product Vision: "{state['product_vision']}"
    - Retrieved Knowledge: "{state['retrieved_knowledge']}"

    Generate an **Agile backlog**:
    - Define 3-5 high-priority **user stories**.
    - Prioritize tasks using SAFe PI Planning principles.
    - Ensure feasibility based on market demand & technical constraints.
    """
    return {"backlog_priorities": query_gemini("SAFe Product Owner", prompt)}

def tech_lead(state: WorkflowState) -> WorkflowState:
    prompt = f"""
    You are a **Tech Lead** assessing feasibility.
    - Feature Request: "{state['feature_request']}"
    - Product Vision: "{state['product_vision']}"
    - Retrieved Knowledge: "{state['retrieved_knowledge']}"
    - Backlog Priorities: "{state['backlog_priorities']}"

    Provide **technical recommendations**:
    - Preferred tech stack (e.g., Python, LangChain, FAISS, Streamlit)
    - Key **challenges & risks**
    - Suggestions for **scalability & performance**.
    """
    return {"technical_feasibility": query_gemini("Tech Lead", prompt)}

def ux_designer(state: WorkflowState) -> WorkflowState:
    prompt = f"""
    You are a **UX Designer** improving user experience.
    - Feature Request: "{state['feature_request']}"
    - Product Vision: "{state['product_vision']}"
    - Backlog Priorities: "{state['backlog_priorities']}"
    - Technical Constraints: "{state.get('technical_feasibility', 'No technical feasibility data available')}"

    Suggest the **UX approach**:
    - Wireframe structure for main UI
    - Intuitive navigation & accessibility features
    - Any technical limitations to consider.
    """
    return {"ux_design": query_gemini("UX Designer", prompt)}

def execution_planner(state: WorkflowState) -> WorkflowState:
    prompt = f"""
    You are an **Agile Project Manager** creating an execution plan.
    - Feature Request: "{state['feature_request']}"
    - Product Vision: "{state['product_vision']}"
    - Backlog Priorities: "{state['backlog_priorities']}"
    - Technical Feasibility: "{state.get('technical_feasibility', 'No technical feasibility data available')}"
    - UX Design: "{state.get('ux_design', 'No UX design data available')}"

    Develop an **execution plan**:
    - Define key **milestones** and **deliverables**.
    - Suggest a **timeline** for each milestone.
    - Identify **dependencies** between tasks.
    """
    return {"execution_plan": query_gemini("Agile Project Manager", prompt)}

def retrospective_analyst(state: WorkflowState) -> WorkflowState:
    prompt = f"""
    You are a **Retrospective Analyst** conducting a post-sprint review.
    - Feature Request: "{state['feature_request']}"
    - Product Vision: "{state['product_vision']}"
    - Backlog Priorities: "{state['backlog_priorities']}"
    - Technical Feasibility: "{state.get('technical_feasibility', 'No technical feasibility data available')}"
    - UX Design: "{state.get('ux_design', 'No UX design data available')}"
    - Execution Plan: "{state.get('execution_plan', 'No execution plan data available')}"

    Conduct a **retrospective analysis**:
    - Identify what **went well** during the sprint.
    - Identify what **could be improved**.
    - Suggest **actionable steps** for future sprints.
    """
    return {"retrospective_analysis": query_gemini("Retrospective Analyst", prompt)}

def okr_advisor(state: WorkflowState) -> WorkflowState:
    prompt = f"""
    You are an **OKR Advisor** aligning work with strategic goals.
    - Feature Request: "{state['feature_request']}"
    - Product Vision: "{state['product_vision']}"

    Provide **OKR insights**:
    - Suggest **Objectives** that this feature request supports.
    - Suggest **Key Results** to measure the success of this feature.
    - Ensure alignment with overall business strategy.
    """
    return {"okr_insights": query_gemini("OKR Advisor", prompt)}

# ✅ Ensure Execution Order
workflow = StateGraph(WorkflowState)

workflow.add_node("product_manager", product_manager)
workflow.add_node("rag_retrieval_agent", rag_retrieval_agent)
workflow.add_node("safe_product_owner", safe_product_owner)
workflow.add_node("tech_lead", tech_lead)
workflow.add_node("ux_designer", ux_designer)
workflow.add_node("execution_planner", execution_planner)
workflow.add_node("retrospective_analyst", retrospective_analyst)
workflow.add_node("okr_advisor", okr_advisor)

workflow.add_edge("product_manager", "rag_retrieval_agent")
workflow.add_edge("rag_retrieval_agent", "safe_product_owner")
workflow.add_edge("safe_product_owner", "tech_lead")
workflow.add_edge("tech_lead", "ux_designer")
workflow.add_edge("ux_designer", "execution_planner")
workflow.add_edge("execution_planner", "retrospective_analyst")
workflow.add_edge("retrospective_analyst", "okr_advisor")

workflow.set_entry_point("product_manager")

# ✅ Functions to Save and Load Conversation History
def save_conversation(session_id, feature_request, workflow_results):
    try:
        timestamp = datetime.now()
        document = {
            "session_id": session_id,
            "timestamp": timestamp,
            "feature_request": feature_request,
            "workflow_results": workflow_results,
        }
        conversations_collection.insert_one(document)
        print(f"Conversation saved successfully for session: {session_id}")
    except Exception as e:
        print(f"Error saving conversation: {e}")

def load_all_conversations():
    history = []
    try:
        documents = conversations_collection.find().sort("timestamp", pymongo.ASCENDING)
        for doc in documents:
            history_item = {
                "session_id": doc.get("session_id", "No session id found"),
                "feature_request": doc.get("feature_request", "No feature request found"),
                "workflow_results": doc.get("workflow_results", {}),
            }
            history.append(history_item)
        print(f"All conversations loaded successfully")
    except Exception as e:
        print(f"Error loading all conversations: {e}")
    return history

# ✅ New Functions for Saving and Loading Saved Notes
def save_notes(session_id, feature_request, workflow_results):
    try:
        timestamp = datetime.now()
        document = {
            "session_id": session_id,
            "timestamp": timestamp,
            "feature_request": feature_request,
            "workflow_results": workflow_results,
        }
        saved_notes_collection.insert_one(document)
        st.success("Notes saved successfully!")
    except Exception as e:
        st.error(f"Error saving notes: {e}")

def load_saved_notes():
    saved_notes = []
    try:
        documents = saved_notes_collection.find().sort("timestamp", pymongo.ASCENDING)
        for doc in documents:
            saved_notes_item = {
                "session_id": doc.get("session_id", "No session id found"),
                "feature_request": doc.get("feature_request", "No feature request found"),
                "workflow_results": doc.get("workflow_results", {}),
            }
            saved_notes.append(saved_notes_item)
        print(f"All saved notes loaded successfully")
    except Exception as e:
        print(f"Error loading saved notes: {e}")
    return saved_notes

# ✅ Streamlit UI
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Workflow", "History", "Saved Notes"])  # Added "Saved Notes"

# ✅ Session Management
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
print(f"Current Session ID: {st.session_state.session_id}")

# ...existing code...

# ... (previous code remains the same until the "Workflow" section)

if page == "Workflow":
    st.title("Agile AI Workflow")
    # ✅ Section 1: URL Summarization
    st.subheader("🔗 Summarize a Webpage")
    url_input = st.text_input("Enter a URL to summarize:")
    if st.button("Summarize"):
        if url_input.strip():
            summary_result = summarize_url(url_input)
            st.subheader("Summary Result")
            st.write(summary_result)
        else:
            st.error("URL cannot be empty.")

    # ✅ Section 2: AI Agile Workflow
    st.subheader("🚀 AI Agile Workflow")

    inbuilt_prompts = [
        "Generate a prioritized backlog based on SAFe PI Planning principles and business goals.",
        "Provide UX design recommendations to improve user experience for a new feature.",
        "Analyze retrieved Agile knowledge and suggest best practices for feature development.",
        "Generate an execution plan for delivering an Agile feature within a sprint cycle."
    ]

    selected_prompt = st.radio("Select a predefined prompt or enter your own:", inbuilt_prompts, index=None)
    feature_request = st.text_input("Enter Feature Request:", selected_prompt if selected_prompt else "")

    if st.button("Run Workflow"):
        if not feature_request.strip():
            st.error("Feature request cannot be empty.")
        else:
            graph = workflow.compile()
            initial_state = {"feature_request": feature_request}
            result = graph.invoke(initial_state)

            # ✅ Save Conversation to MongoDB (History)
            save_conversation(st.session_state.session_id, feature_request, result)

            st.subheader("Workflow Results")
            for key, value in result.items():
                st.write(f"**{key.replace('_', ' ').title()}**:")
                st.write(value)

            # ✅ Store workflow results in session state for saving notes
            st.session_state.workflow_results = result
            st.session_state.feature_request = feature_request

    # ✅ Save Notes Button (only show if workflow results are available)
    if "workflow_results" in st.session_state:
        if st.button("Save Notes"):
            save_notes(st.session_state.session_id, st.session_state.feature_request, st.session_state.workflow_results)
            st.success("Notes saved successfully!")

# ... (previous code remains the same until the "Saved Notes" section)

elif page == "Saved Notes":
    st.title("Saved Notes")
    saved_notes = load_saved_notes()
    if saved_notes:
        for item in saved_notes:
            st.write(f"**Session ID:** {item['session_id']}")
            st.write(f"**Feature Request:** {item['feature_request']}")
            for key, value in item["workflow_results"].items():
                st.write(f"**{key.replace('_', ' ').title()}**:")
                st.write(value)
            st.write("---")
    else:
        st.write("No saved notes found.")
