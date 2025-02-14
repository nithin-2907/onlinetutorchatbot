import spacy
import streamlit as st
import wikipediaapi
from langchain_groq import ChatGroq
from textblob import TextBlob

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Initialize Wikipedia API
wiki = wikipediaapi.Wikipedia(user_agent="AI-TutorBot/1.0", language='en')

# Initialize Language Model
def initialize_llm():
    return ChatGroq(
        temperature=0.5,
        groq_api_key="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",  # Replace with your API key
        model_name="llama-3.3-70b-versatile"
    )

# Initialize LLM
llm = initialize_llm()

# Streamlit UI
st.title("🧑‍🏫 AI Online Tutor Bot ")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    st.session_state.chat_history.append({"user": "", "bot": "Hello! How can I assist you today? 😊"})

# Function to perform tokenization, lemmatization, and POS tagging
def process_text(text):
    doc = nlp(text)
    tokens = [token.text for token in doc]  # Tokenization
    lemmas = [token.lemma_ for token in doc]  # Lemmatization
    pos_tags = [(token.text, token.pos_) for token in doc]  # POS tagging
    return tokens, lemmas, pos_tags

# Function to extract named entities
def extract_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

# Function to analyze sentiment
def analyze_sentiment(text):
    sentiment_score = TextBlob(text).sentiment.polarity
    if sentiment_score > 0:
        return "Positive 😊"
    elif sentiment_score < 0:
        return "Negative 😔"
    else:
        return "Neutral 😐"

# Function to classify user question type
def classify_question(text):
    text = text.lower()
    if "what" in text:
        return "Definition"
    elif "who" in text:
        return "Person"
    elif "how" in text:
        return "Process"
    elif "why" in text:
        return "Reason"
    elif "where" in text:
        return "Location"
    elif "when" in text:
        return "Time"
    else:
        return "General"

# Function to retrieve Wikipedia answers
def search_wikipedia(query):
    page = wiki.page(query)
    return page.summary[:500] if page.exists() else None

# Function to process user input
def chatbot_response(user_input):
    if not user_input.strip():
        return "Please ask a valid question."

    # Handle greetings
    if user_input.lower() in ["hi", "hello"]:
        return "Hello! 😊 How can I help you today?"

    # Handle exit command
    if user_input.lower() == "exit":
        st.session_state.chat_history.append({"user": user_input, "bot": "Goodbye! Have a great day! 👋"})
        st.rerun()
        st.stop()

    # Extract NLP features
    tokens, lemmas, pos_tags = process_text(user_input)
    entities = extract_entities(user_input)
    sentiment = analyze_sentiment(user_input)
    question_type = classify_question(user_input)

    # Search Wikipedia for factual answers
    wiki_answer = search_wikipedia(user_input)

    # Generate AI response
    response = wiki_answer if wiki_answer else llm.invoke(user_input).content

    # Store conversation history
    st.session_state.chat_history.append({
        "user": user_input,
        "bot": response,
        "tokens": tokens,
        "lemmas": lemmas,
        "pos_tags": pos_tags,
        "sentiment": sentiment,
        "entities": entities,
        "question_type": question_type
    })

    st.rerun()  # ✅ Refresh UI after response

# **DISPLAY CHAT HISTORY (WhatsApp Format)**
chat_container = st.container()
with chat_container:
    for chat in st.session_state.chat_history:
        if chat["user"]:  # Ignore initial empty user message
            with st.chat_message("user"):
                st.markdown(f"**User:** {chat['user']}")
        with st.chat_message("assistant"):
            st.markdown(f"**Tutor Bot:** {chat['bot']}")

            # Display NLP concepts
            # Display NLP concepts only if they exist
            if "tokens" in chat:
                st.markdown(f"📌 **Tokens:** {', '.join(chat['tokens'])}")
            if "lemmas" in chat:
                st.markdown(f"📌 **Lemmas:** {', '.join(chat['lemmas'])}")
            if "pos_tags" in chat:
                st.markdown(f"📌 **POS Tags:** {', '.join([f'{word} ({tag})' for word, tag in chat['pos_tags']])}")
            if "entities" in chat and chat["entities"]:
                st.markdown(f"📌 **Named Entities:** {', '.join([f'{ent} ({label})' for ent, label in chat['entities']])}")
            if "sentiment" in chat:
                st.markdown(f"📌 **Sentiment:** {chat['sentiment']}")


# **INPUT SECTION AT BOTTOM**
st.markdown("---")  # Divider for better UI
user_input = st.text_input("Type your message...", key="user_input", label_visibility="collapsed")

if st.button("Send", use_container_width=True) and user_input:
    chatbot_response(user_input)

# **Clear Chat Button**
if st.button("Clear Chat", use_container_width=True):
    st.session_state.chat_history = [{"user": "", "bot": "Hello! How can I assist you today? 😊"}]
    st.rerun()

