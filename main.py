import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load models
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

# Custom CSS for minimal, clean UI
st.markdown("""
    <style>
    /* Main container */
    .main {
        padding: 2rem 1rem;
    }
    
    /* Title styling */
    h1 {
        font-weight: 600;
        font-size: 2rem !important;
        margin-bottom: 0.5rem !important;
        color: #1f1f1f;
    }
    
    /* Subtitle styling */
    .subtitle {
        color: #666;
        font-size: 0.95rem;
        margin-bottom: 2rem;
    }
    
    /* Text area styling */
    .stTextArea textarea {
        border-radius: 8px;
        border: 1.5px solid #e0e0e0;
        padding: 0.75rem;
        font-size: 0.95rem;
        transition: border-color 0.2s ease;
    }
    
    .stTextArea textarea:focus {
        border-color: #4285f4;
        box-shadow: 0 0 0 1px #4285f4;
    }
    
    /* Button styling */
    .stButton button {
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 500;
        width: 100%;
        transition: all 0.2s ease;
    }
    
    /* Result card styling */
    .result-card {
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 1.5rem;
        text-align: center;
        animation: fadeIn 0.3s ease-in;
    }
    
    .spam-result {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        color: white;
        border: none;
    }
    
    .not-spam-result {
        background: linear-gradient(135deg, #51cf66 0%, #37b24d 100%);
        color: white;
        border: none;
    }
    
    .result-card h2 {
        font-size: 1.5rem;
        margin: 0;
        font-weight: 600;
    }
    
    .result-icon {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Keyboard hint */
    .keyboard-hint {
        color: #888;
        font-size: 0.85rem;
        margin-top: 0.5rem;
        text-align: center;
    }
    
    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# App title and subtitle
st.title("📧 Email/SMS Spam Classifier")
st.markdown('<p class="subtitle">Paste your message below to check if it\'s spam or legitimate</p>', unsafe_allow_html=True)

# Initialize session state
if 'result' not in st.session_state:
    st.session_state.result = None

# Input text area with key
input_sms = st.text_area(
    "Message", 
    placeholder="Enter your email or SMS text here...",
    height=150,
    label_visibility="collapsed",
    key="message_input"
)

# Predict button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_clicked = st.button('🔍 Analyze Message', type="primary", use_container_width=True)

# Keyboard hint

# Function to perform prediction
def predict_spam():
    if input_sms.strip():
        # 1. preprocess
        transformed_sms = transform_text(input_sms)
        # 2. vectorize
        vector_input = tfidf.transform([transformed_sms])
        # 3. predict
        result = model.predict(vector_input)[0]
        st.session_state.result = result
        return result
    return None

# Handle prediction
if predict_clicked:
    predict_spam()

# Keyboard shortcut handling using streamlit-shortcuts
try:
    from streamlit_shortcuts import add_shortcuts
    add_shortcuts(
        message_input="ctrl+enter"
    )
    # Check if enter was pressed while in text area
    if st.session_state.get('message_input') and input_sms:
        predict_spam()
except ImportError:
    # Fallback: JavaScript-based keyboard shortcut
    st.markdown("""
        <script>
        document.addEventListener('keydown', function(e) {
            if (e.ctrlKey && e.key === 'Enter') {
                e.preventDefault();
                const buttons = window.parent.document.querySelectorAll('button[kind="primary"]');
                if (buttons.length > 0) {
                    buttons[0].click();
                }
            }
        });
        </script>
    """, unsafe_allow_html=True)

# Display result
if st.session_state.result is not None:
    if st.session_state.result == 1:
        st.markdown("""
            <div class="result-card spam-result">
                <div class="result-icon">🚫</div>
                <h2>Spam Detected</h2>
                <p style="margin-top: 0.5rem; opacity: 0.95;">This message appears to be spam</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class="result-card not-spam-result">
                <div class="result-icon">✅</div>
                <h2>Safe Message</h2>
                <p style="margin-top: 0.5rem; opacity: 0.95;">This message appears to be legitimate</p>
            </div>
        """, unsafe_allow_html=True)
