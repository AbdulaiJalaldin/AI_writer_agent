from langgraph.graph import StateGraph,END
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import ChatPromptTemplate
from typing import Dict, TypedDict
import streamlit as st
import sqlite3
from datetime import datetime
import time

api_key = os.getenv("TAVILY_API_KEY")
groq_key = os.getenv("GROQ_API_KEY")
groq_key = st.secrets["GROQ_API_KEY"] 
api_key =st.secrets["TAVILY_API_KEY"]




model=ChatGroq(model="llama-3.3-70b-versatile",api_key=groq_key)
search_tool = TavilySearchResults(
    # Make sure to use the correct parameter name
    max_results=3
)


#########data base connection#########
def init_db():
    conn = sqlite3.connect('articles.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS articles
                 (id INTEGER PRIMARY KEY, topic TEXT, content TEXT, created_at TEXT)''')
    conn.commit()
    conn.close()

def save_article(topic, content):
    conn = sqlite3.connect('articles.db')
    c = conn.cursor()
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO articles (topic,content, created_at) VALUES (?, ?, ?)", (topic,content, created_at))
    article_id = c.lastrowid 
    conn.commit()
    conn.close()
    return article_id 

@st.cache_data(ttl=1)
def get_articles():
    conn = sqlite3.connect('articles.db')
    c = conn.cursor()
    c.execute("SELECT * FROM articles")
    articles = c.fetchall()
    conn.close()
    return articles

def get_article_by_id(article_id):
    conn = sqlite3.connect('articles.db')
    c = conn.cursor()
    c.execute("SELECT * FROM articles WHERE id=?", (article_id,))
    article = c.fetchone()
    conn.close()
    return article

def delete_article(article_id):
    conn = sqlite3.connect('articles.db')
    c = conn.cursor()
    c.execute("DELETE FROM articles WHERE id=?", (article_id,))
    conn.commit()
    conn.close()


class GraphState(TypedDict):
    input: str
    search_data: list  # Changed from search_results
    outline_data: str  # Changed from outline
    article_data: str  # Changed from article
def search_process(state: Dict) -> Dict:
    """Search node that returns results."""
    query = state["input"]
    results = search_tool.invoke(query)  # Remove the dictionary wrapper
    new_state = state.copy()
    new_state["search_data"] = results
    return new_state

def outline_process(state: Dict) -> Dict:
    """Outline node that processes search results."""
    # Modify how we handle the search results
    results_str = "\n\n".join(
        [f"Source {i+1}:\n{res}" 
         for i, res in enumerate(state["search_data"])]
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You're a senior editor. Create a structured outline for a news article using these sources. Include key points and mark sections."),
        ("user", f"Query: {state['input']}\n\nSources:\n{results_str}")
    ])
    
    response = model.invoke(prompt.format_messages())
    new_state = state.copy()
    new_state["outline_data"] = response.content
    return new_state

def article_process(state: Dict) -> Dict:
    """Article node that generates the final article."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You're a professional journalist. Write a comprehensive article using this outline. Cite sources appropriately."),
        ("user", f"Outline:\n{state['outline_data']}")
    ])
    
    response = model.invoke(prompt.format_messages())
    new_state = state.copy()
    new_state["article_data"] = response.content
    return new_state

# Create workflow
workflow = StateGraph(GraphState)

# Add nodes with different names from state keys
workflow.add_node("search_step", search_process)
workflow.add_node("outline_step", outline_process)
workflow.add_node("article_step", article_process)

# Configure the graph
workflow.set_entry_point("search_step")
workflow.add_edge("search_step", "outline_step")
workflow.add_edge("outline_step", "article_step")
workflow.add_edge("article_step", END)

# Compile the graph
app = workflow.compile()


    


def main():
    init_db()
    st.set_page_config(page_title="AI writer agent", page_icon="✍️", layout="wide")
    
    # Initialize session states
    if 'deleted_articles' not in st.session_state:
        st.session_state.deleted_articles = set()
    if 'current_article_id' not in st.session_state:
        st.session_state.current_article_id = None
    if 'show_new_article' not in st.session_state:
        st.session_state.show_new_article = True
    if 'new_article_generated' not in st.session_state:
        st.session_state.new_article_generated = False
    if 'generated_content' not in st.session_state:
        st.session_state.generated_content = None
    
    def start_new_article():
        st.session_state.show_new_article = True
        st.session_state.current_article_id = None
        st.session_state.new_article_generated = False
        st.session_state.generated_content = None
    
    # Sidebar
    st.sidebar.title("AI Writer")
    
    if st.sidebar.button("➕ New Article", use_container_width=True):
        start_new_article()
    
    st.sidebar.divider()
    
    st.sidebar.subheader("Previous Articles")
    
    def handle_delete(article_id):
        try:
            delete_article(article_id)
            st.session_state.deleted_articles.add(article_id)
            if st.session_state.current_article_id == article_id:
                start_new_article()
            return True
        except Exception as e:
            st.error(f"Error deleting article: {str(e)}")
            return False
    
    previous_articles = [
        article for article in get_articles() 
        if article[0] not in st.session_state.deleted_articles
    ]
    
    # Display articles in the sidebar
    if not previous_articles:
        st.sidebar.write("No articles yet")
    else:
        for article in previous_articles:
            col1, col2 = st.sidebar.columns([3, 1])
            with col1:
                if st.sidebar.button(f"{article[1]}", key=f"article_{article[0]}"):
                    st.session_state.current_article_id = article[0]
                    st.session_state.show_new_article = False
                    st.session_state.new_article_generated = False
                    st.rerun()
            with col2:
                if st.sidebar.button("🗑️", key=f"delete_{article[0]}"):
                    if handle_delete(article[0]):
                        st.cache_data.clear()
                        st.rerun()
    
    # Main content area
    st.title("AI Writer Agent")
    
    # Show selected article or new article interface
    if st.session_state.new_article_generated and st.session_state.generated_content:
        # Display the newly generated article
        topic, content = st.session_state.generated_content
        st.write(f"### {topic}")
        st.write(content)
    elif not st.session_state.show_new_article and st.session_state.current_article_id:
        # Display selected previous article
        article = get_article_by_id(st.session_state.current_article_id)
        if article:
            st.write(f"### {article[1]}")
            st.write(article[2])
    else:
        # Show new article interface
        st.markdown("Generate comprehensive articles with AI-powered research")
        query = st.text_input("Enter your article topic:", placeholder="Enter your article topic here")
        generate_button = st.button("Generate Article")
        if generate_button and query:
            with st.spinner('Starting research and writing process...'):
                initial_state = {"input": query}
                result = app.invoke(initial_state)
                
                # Get the generated content
                article_content = result.get("article_data", "No article generated")
                
                # Save to database and get the article id
                article_id = save_article(
                    topic=query,
                    content=article_content
                )
                
                # Update session state
                st.session_state.generated_content = (query, article_content)
                st.session_state.new_article_generated = True
                st.session_state.current_article_id = article_id
                st.session_state.show_new_article = False
                
                # Clear the cache for get_articles
                st.cache_data.clear()
                
                # Force a rerun to update the UI
                st.rerun()
            
            

if __name__ == "__main__":
  main()
