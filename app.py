import streamlit as st
import time
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings  
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough

## use cache
@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

@st.cache_resource(show_spinner=False)
def get_vectorStore():  # Fixed typo in function name
    return FAISS.load_local(
        "faiss_index",
        embeddings=get_embeddings(),
        allow_dangerous_deserialization=True
    )

@st.cache_resource(show_spinner=False)
def get_rag_chain():
    vectorstore = get_vectorStore()  # Fixed function name
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    prompt_template = """
        Analyze these clinical guidelines
        {context}

        Question: {question}

        Provide Structured Response:
        1. Diagnosis criteria
        2. Treatment protocol
        3. Acupuncture points
        4. Safety considerations
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    llm = ChatGroq(
        temperature=0.3,
        model_name="llama3-70b-8192",
        api_key=st.secrets["GROQ_API_KEY"]
    )

    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

## Custom CSS for the input field with arrow animation
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600&display=swap');

/* Heartbeat animation */
@keyframes heartbeat {
  0% { transform: scale(1); }
  14% { transform: scale(1.1); }
  28% { transform: scale(1); }
  42% { transform: scale(1.1); }
  70% { transform: scale(1); }
}

.title-container {
  text-align: center;
  margin: 1.5rem auto;
  padding: 1.2rem;
  position: relative;
  max-width: 800px;
}

.pulse-ring {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 95%;
  height: 100%;
  border: 2px solid #2c9c5a;
  border-radius: 12px;
  opacity: 0;
  animation: pulse 2s infinite;
}

@keyframes pulse {
  0% { transform: translate(-50%, -50%) scale(0.95); opacity: 0.7; }
  70% { transform: translate(-50%, -50%) scale(1.05); opacity: 0; }
  100% { transform: translate(-50%, -50%) scale(1.05); opacity: 0; }
}

.custom-title {
  font-family: 'Poppins', sans-serif;
  font-size: 2.8rem;
  font-weight: 600;
  line-height: 1.2;
  letter-spacing: 0.5px;
  margin: 0;
  display: inline-block;
}

.custom-title .emoji {
  font-size: 2.5rem;
  margin-right: 0.5rem;
  vertical-align: middle;
  animation: heartbeat 2s infinite;
  display: inline-block;
}

.custom-title .word-1 {
  color: #0d5f36;
  text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
}

.custom-title .word-2 {
  color: #1a8d50;
  text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
}

.custom-title .word-3 {
  color: #2cb567;
  text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
}

/* Response text styling */
.response-text {
    font-family: 'Poppins', sans-serif;
    font-size: 1.1rem;
    line-height: 1.6;
    max-width: 800px;
    margin: 0 auto;
    padding: 2rem 1rem;
    color: #ffffff;  /* Changed to white for contrast on black */
    font-weight: 500;
    position: relative;
    background-color: #000000;  /* Changed to black background */
    border-radius: 8px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.2);
}

/* Typing cursor animation */
.typing-cursor {
   display: inline-block;
   width: 8px;
   height: 1.2rem;
   background: #ffffff;  /* Changed to white for visibility on black */
   margin-left: 2px;
   animation: blink-caret .75s step-end infinite;
}

@keyframes blink-caret {
    from, to { opacity: 1 }
    50% { opacity: 0 }
}

.stream-container {
    min-height: 200px;
    padding: 1rem;
}
</style>

<div class="title-container">
  <div class="pulse-ring"></div>
  <h1 class="custom-title">
    <span class="emoji">🩺</span>
    <span class="word-1">Acupuncture</span> 
    <span class="word-2">Clinical</span> 
    <span class="word-3">Advisor</span>
  </h1>
</div>
""", unsafe_allow_html=True)

# JavaScript for enter key detection
st.components.v1.html("""
<script>
const container = window.parent.document.querySelector('.input-container');
const textInput = container.querySelector('input');

textInput.addEventListener('keypress', function(e) {
    if(e.key === 'Enter') {
        this.blur();  // Force Streamlit update
    }
});

// Observe mutations to handle dynamic elements
new MutationObserver(() => {
    const loading = window.parent.document.getElementById('loading-spinner');
    const arrow = window.parent.document.getElementById('submit-arrow');
    
    if(loading && !loading.style.display) {
        arrow.style.display = 'none';
        loading.style.display = 'block';
    }
}).observe(container, { childList: true, subtree: true });
</script>
""")


# Input container with animated elements
with st.container():
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    
    query = st.text_input(
        "Enter clinical query:",
        placeholder="e.g. Treatment protocol for chronic headache...",
        key="query_input",
        label_visibility="collapsed"
    )
    
    # Animated submit indicator
    if st.session_state.get('processing', False):
        st.markdown('<div class="loading-spinner" id="loading-spinner"></div>', unsafe_allow_html=True)
    else:
        st.markdown('<i class="submit-arrow fa-solid fa-arrow-up" id="submit-arrow"></i>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Font Awesome CSS (for the arrow icon)
st.markdown('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">', 
            unsafe_allow_html=True)

# Modify your response handling section
if query:
    if 'last_query' not in st.session_state or query != st.session_state.last_query:
        st.session_state.processing = True
        st.session_state.last_query = query
        
        try:
            with st.spinner("Analyzing clinical guidelines..."):
                chain = get_rag_chain()
                response = chain.invoke(query)
                
                # Store the full response
                st.session_state.full_response = response.content
                st.session_state.display_response = ""
                
                # Create a container for the streaming effect
                response_container = st.empty()
                response_container.markdown('<div class="stream-container">', unsafe_allow_html=True)

                # Split response into words for streaming effect
                words = st.session_state.full_response.split()
                for word in words + [""]:  # Add empty string for final cursor
                    st.session_state.display_response += word + " "
                    
                    # Create the streaming effect HTML
                    streaming_html = f"""
                    <div class="response-text">
                        {st.session_state.display_response}
                        <span class="typing-cursor"></span>
                    </div>
                    """
                    
                    response_container.markdown(streaming_html, unsafe_allow_html=True)
                    time.sleep(0.05)  # Adjust speed here

                # Show final response without cursor
                response_container.markdown(
                    f'<div class="response-text">{st.session_state.full_response}</div>',
                    unsafe_allow_html=True
                )
                
        except Exception as e:
            st.error(f"Clinical analysis failed: {str(e)}")
            st.exception(e)
        finally:
            st.session_state.processing = False