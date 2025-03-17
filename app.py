import streamlit as st
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langgraph.graph import StateGraph
from typing import TypedDict, Annotated

# ✅ Initialize Google Gemini API
genai.configure(api_key="AIzaSyCCEUdNbJcvaT76FTjoWdqY1q3eWwRtQO8")

def query_gemini(role: str, prompt: str) -> str:
    try:
        model = genai.GenerativeModel("gemini-1.5-flash-latest")
        response = model.generate_content(f"You are a {role}. {prompt}")
        return response.text if response else "[No meaningful response received.]"
    except Exception as e:
        return f"[Error querying Gemini API: {str(e)}]"

# ✅ Define Workflow State Schema
class WorkflowState(TypedDict):
    feature_request: Annotated[str, "single"]
    refined_requirements: Annotated[str, "single"]
    product_vision: Annotated[str, "single"]
    retrieved_knowledge: Annotated[str, "single"]
    backlog_priorities: Annotated[str, "single"]
    technical_feasibility: Annotated[str, "single"]
    ux_design: Annotated[str, "single"]

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

# ✅ Requirement Analysis with Interactive Follow-Up Questions
if 'questioning_stage' not in st.session_state:
    st.session_state.questioning_stage = 0
    st.session_state.refined_responses = []
    st.session_state.follow_up_question = ""
    st.session_state.feature_request = ""

if st.session_state.questioning_stage == 0:
    st.session_state.feature_request = st.text_input("Enter Feature Request:")
    if st.button("Start Requirement Analysis"):
        if st.session_state.feature_request.strip():
            st.session_state.follow_up_question = query_gemini("Business Analyst", f"Ask the first follow-up question for the feature request: {st.session_state.feature_request}")
            st.session_state.questioning_stage = 1
        else:
            st.error("Feature request cannot be empty.")

elif st.session_state.questioning_stage > 0:
    st.write("**Follow-Up Question:**", st.session_state.follow_up_question)
    user_response = st.text_input("Your Response:")
    if st.button("Next Question"):
        if user_response.strip():
            st.session_state.refined_responses.append(user_response)
            follow_up_prompt = f"Given the feature request: {st.session_state.feature_request} and responses: {st.session_state.refined_responses}, ask another follow-up question or finalize the refined requirements."
            next_question = query_gemini("Business Analyst", follow_up_prompt)
            
            if "finalized requirements" in next_question.lower():
                st.session_state.refined_requirements = next_question
                st.session_state.questioning_stage = -1  # Move to Agile processing
            else:
                st.session_state.follow_up_question = next_question
                st.session_state.questioning_stage += 1
        else:
            st.error("Response cannot be empty.")

# ✅ Proceed to Agile Workflow After Requirement Analysis
if st.session_state.questioning_stage == -1:
    st.success("Requirement analysis completed! Running Agile workflow...")
    
    # ✅ Define Workflow Nodes
    def product_manager(state: WorkflowState) -> WorkflowState:
        prompt = f"""
        You are a **Product Manager** following **SVPG Agile** principles.
        - Define the **product vision** for the refined requirements: "{state['refined_requirements']}".
        - Identify the **customer problem** it solves.
        - Prioritize based on **market demand, business goals, and usability**.
        """
        return {"product_vision": query_gemini("Product Manager", prompt)}
    
    def rag_retrieval_agent(state: WorkflowState) -> WorkflowState:
        if vector_store:
            docs = vector_store.similarity_search(state["refined_requirements"], k=2)
            if docs:
                return {"retrieved_knowledge": "\n".join([doc.page_content for doc in docs])}
        return {"retrieved_knowledge": "[No relevant knowledge found in the database.]"}
    
    def safe_product_owner(state: WorkflowState) -> WorkflowState:
        prompt = f"""
        You are a **SAFe Product Owner** managing an Agile Release Train (ART).
        - Translate the product vision into an **Agile backlog**.
        - Prioritize tasks based on **retrieved insights** and SAFe PI Planning principles.
        """
        return {"backlog_priorities": query_gemini("SAFe Product Owner", prompt)}
    
    def tech_lead(state: WorkflowState) -> WorkflowState:
        return {"technical_feasibility": query_gemini("Tech Lead", "You are a Tech Lead assessing technical feasibility. Provide recommendations.")}
    
    def ux_designer(state: WorkflowState) -> WorkflowState:
        return {"ux_design": query_gemini("UX Designer", "You are a UX Designer improving the user experience.")}
    
    workflow = StateGraph(WorkflowState)
    workflow.add_node("product_manager", product_manager)
    workflow.add_node("rag_retrieval_agent", rag_retrieval_agent)
    workflow.add_node("safe_product_owner", safe_product_owner)
    workflow.add_node("tech_lead", tech_lead)
    workflow.add_node("ux_designer", ux_designer)
    workflow.add_node("end", lambda state: state)
    
    workflow.set_entry_point("product_manager")
    workflow.add_edge("product_manager", "rag_retrieval_agent")
    workflow.add_edge("rag_retrieval_agent", "safe_product_owner")
    workflow.add_edge("safe_product_owner", "tech_lead")
    workflow.add_edge("safe_product_owner", "ux_designer")
    workflow.add_edge("tech_lead", "end")
    workflow.add_edge("ux_designer", "end")
    
    graph = workflow.compile()
    initial_state = {"feature_request": st.session_state.feature_request, "refined_requirements": st.session_state.refined_requirements}
    result = graph.invoke(initial_state)
    
    st.subheader("Workflow Results")
    for key, value in result.items():
        st.write(f"**{key.replace('_', ' ').title()}**:")
        st.write(value)
