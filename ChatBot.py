import streamlit as st
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertForSequenceClassification, BertTokenizer, BertModel
import time
import os
import requests
import json
import re

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

# Download necessary NLTK data
nltk.download('wordnet')
nltk.download('stopwords')
from MBTI_BERT import predict_mbti
from match import match_main

# Set page configuration
st.set_page_config(
    page_title="Intelligent MBTI Personality Analysis System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add CSS style
st.markdown("""
<style>
   .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
   .sub-header {
        font-size: 1.5rem;
        color: #3B82F6;
        margin-bottom: 1rem;
    }
   .result-card {
        background-color: #F0F9FF;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
   .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
    }
   .chat-message.user {
        background-color: #E0F2FE;
        margin-left: 20%;
    }
   .chat-message.assistant {
        background-color: #F0F9FF;
        margin-right: 20%;
    }
   .thinking {
        font-style: italic;
        color: #6B7280;
    }
</style>
""", unsafe_allow_html=True)

# MBTI related data and functions
mbti_types = [
    "INTJ", "INTP", "ENTJ", "ENTP",
    "INFJ", "INFP", "ENFJ", "ENFP",
    "ISTJ", "ISFJ", "ESTJ", "ESFJ",
    "ISTP", "ISFP", "ESTP", "ESFP"
]

mbti_descriptions = {
    "INTJ": "Architect - Imaginative and strategic thinkers with creative, logical, and analytical minds",
    "INTP": "Logician - Innovative inventors with an insatiable curiosity and a thirst for knowledge",
    "ENTJ": "Commander - Bold, imaginative, and strong - willed leaders who always find or create solutions",
    "ENTP": "Debater - Smart and curious thinkers who never say no to an intellectual challenge",
    "INFJ": "Advocate - Quiet and mysterious, but very inspiring and idealistic individuals",
    "INFP": "Mediator - Poetic, kind, and altruistic personalities, always eager to help a worthy cause",
    "ENFJ": "Protagonist - Charismatic and inspiring leaders who can captivate an audience",
    "ENFP": "Campaigner - Enthusiastic, creative, and sociable free spirits who always find a reason to smile",
    "ISTJ": "Logistician - Practical and fact - oriented individuals whose reliability is beyond question",
    "ISFJ": "Defender - Very dedicated and warm protectors, always ready to safeguard those they care about",
    "ESTJ": "Executive - Excellent managers with unparalleled ability to manage people and things",
    "ESFJ": "Consul - Extremely caring, social, and popular, always eager to help",
    "ISTP": "Virtuoso - Bold and practical experimenters who master a variety of tools",
    "ISFP": "Adventurer - Flexible and charming artists who are always ready to explore and experience new things",
    "ESTP": "Entrepreneur - Smart, energetic, and very perceptive individuals who truly enjoy taking risks",
    "ESFP": "Entertainer - Spontaneous, energetic, and enthusiastic entertainers whose lives are never dull"
}

# MBTI dimension mapping table - used to guide the direction that the LLM focuses on
mbti_dimensions = {
    "EI": {"focus": "Social interaction preferences, energy sources", "keywords": ["alone time", "socializing", "extroverted", "introverted", "group", "quiet"]},
    "SN": {"focus": "Information collection and processing methods", "keywords": ["details", "facts", "imagination", "possibilities", "innovation", "tradition"]},
    "TF": {"focus": "Decision - making methods", "keywords": ["logic", "analysis", "emotions", "values", "objective", "subjective"]},
    "JP": {"focus": "Lifestyle preferences", "keywords": ["planning", "organization", "flexibility", "spontaneity", "deadlines", "openness"]}
}

# LLM system prompt
system_prompt = """
You are a friendly and good listener dialogue assistant, aiming to understand the user's personality traits through natural conversation.
You need to dynamically adjust the conversation according to the user's answers. If the user answers in English, you should also answer in English, guiding the user to share more information about themselves.

During the conversation, you need to softly explore the following aspects:
1. Social preferences and energy sources (E/I dimension): Social activities, alone time, group interaction preferences
2. Information processing methods (S/N dimension): Focus on details or concepts, focus on reality or possibilities
3. Decision - making methods (T/F dimension): Logical analysis or values and feelings
4. Lifestyle (J/P dimension): Planning, flexibility, attitude towards deadlines

Please follow these principles:
- Keep the conversation natural and smooth, avoiding an obvious sense of psychological assessment
- First give a sincere response to what the user has shared, and then naturally guide to the relevant topic
- Use the information provided by the user as a bridge to introduce new questions
- Each question should have a natural connection with the previous answer
- Use open - ended questions to encourage the user to express in detail
- Each response should be short and friendly, not exceeding 2 - 3 sentences
- Show appropriate empathy for emotional content

Guiding techniques:
- Use natural transition words like "This makes me think..." or "Speaking of which..."
- Find clues related to a certain dimension from the content shared by the user and ask questions along that line
- Do not directly use psychological terms, but explore through daily life scenarios

Your goal is to gradually understand the user's tendencies in each dimension while keeping the conversation natural and smooth.
"""

# Function: Count the number of words in the text (supports both Chinese and English)
def count_words(text):
    # Count English words
    english_words = len(re.findall(r'\b[a-zA-Z]+\b', text))
    
    # Count Chinese characters
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    
    # Total word count: English words + Chinese characters
    return english_words + chinese_chars

# Initialize the BERT model
@st.cache_resource
def load_bert_model():
    # Use a placeholder path here, which needs to be replaced with the actual model path
    bert_model_path = "./best_model.pth"
    
    # If in test mode, use a simple simulation
    if os.environ.get("STREAMLIT_TEST_MODE", "false").lower() == "true":
        class MockBertModel:
            def __init__(self):
                pass
            
            def eval(self):
                return self
            
            def __call__(self, **kwargs):
                class MockOutput:
                    def __init__(self):
                        self.logits = torch.rand(1, 16)
                return MockOutput()
        
        class MockTokenizer:
            def __init__(self):
                pass
            
            def __call__(self, text, padding=True, truncation=True, return_tensors="pt", max_length=512):
                return {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]])}

        return MockTokenizer(), MockBertModel()
    
    try:
        tokenizer = BertTokenizer.from_pretrained(bert_model_path)
        model = BertForSequenceClassification.from_pretrained(bert_model_path, num_labels=16)
        model.eval()
        return tokenizer, model
    except Exception as e:
        st.error(f"Failed to load the BERT model: {e}")
        # Return a simple simulation model for testing
        return MockTokenizer(), MockBertModel()

# Ollama API interface
class OllamaLLM:
    def __init__(self, model_name="deepseek - r1:1.5b", api_url="http://localhost:11434/api/generate"):
        self.model_name = model_name
        self.api_url = api_url
        
    def __call__(self, prompt, stop=None, temperature=0.5, max_tokens=50):
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            if stop:
                payload["stop"] = stop if isinstance(stop, list) else [stop]
                
            response = requests.post(self.api_url, json=payload)
            
            if response.status_code == 200:
                # Parse the response
                response_text = response.text
                
                # Ollama API returns JSONL format, where each line is a JSON object
                # We only care about the final completed text
                last_line = None
                full_response = ""
                
                for line in response_text.strip().split('\n'):
                    if line:
                        last_line = json.loads(line)
                        full_response += last_line.get("response", "")
                
                # Process the reply and remove the thinking process
                final_response = self._clean_response(full_response)
                return final_response
            else:
                st.error(f"Ollama API error: {response.status_code}")
                return "Sorry, I can't answer for now. Please try again later."
        
        except Exception as e:
            st.error(f"Error when calling the Ollama model: {e}")
            return "Sorry, unable to connect to the model service. Please check if Ollama is running."

    def _clean_response(self, text):
        """Clean the model's reply and keep the natural dialogue part"""
        # If the reply contains obvious thinking process indicator words, remove these parts
        thinking_patterns = [
            r"Let me think", r"I should ask", r"I need to know", r"Based on the user's answer",
            r"I should focus on", r"The next question should", r"I will ask", r"Thinking:",
            r"Analysis:", r"Strategy:"
        ]
        
        # Preprocessing: Remove obvious thinking processes
        for pattern in thinking_patterns:
            text = re.sub(pattern + r".*?\n", "", text, flags=re.IGNORECASE | re.DOTALL)
        
        # Split into sentences
        sentences = re.split(r'[.!?。！？]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return "I understand. Can you tell me more?"
        
        # Handle long replies
        if len(sentences) > 3:
            # Keep the empathetic part and the question part
            # Usually the first sentence is empathetic, and the last sentence is a question
            if len(sentences) >= 4:
                return sentences[0] + ". " + ". ".join(sentences[-2:]) + "?"
            else:
                return ". ".join(sentences[-3:]) + "?"
        
        # Handle short replies
        cleaned_text = ". ".join(sentences)
        if not cleaned_text.endswith(("?", "？")):
            cleaned_text += "?"
            
        return cleaned_text

    def stream(self, prompt, stop=None, temperature=0.5, max_tokens=50):
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True
            }
            
            if stop:
                payload["stop"] = stop if isinstance(stop, list) else [stop]
                
            response = requests.post(self.api_url, json=payload, stream=True)
            
            if response.status_code == 200:
                collected_response = ""
                for line in response.iter_lines():
                    if line:
                        try:
                            json_line = json.loads(line)
                            chunk = json_line.get("response", "")
                            collected_response += chunk
                            yield chunk
                        except json.JSONDecodeError:
                            continue
                
                # At the end of the streaming response, clean and replace the output
                cleaned_response = self._clean_response(collected_response)
                yield "\nFinal question: " + cleaned_response
            else:
                yield f"Ollama API error: {response.status_code}"
        
        except Exception as e:
            yield f"Error when calling the Ollama model: {e}"

# Initialize the Ollama model
@st.cache_resource
def load_ollama_model(model_name="deepseek - r1:1.5b"):
    return OllamaLLM(model_name=model_name)

# Dynamic conversation management system
class DynamicConversationManager:
    def __init__(self, llm, bert_model, tokenizer, mbti_types):
        self.llm = llm
        self.bert_model = bert_model
        self.tokenizer = tokenizer
        self.mbti_types = mbti_types
        self.conversation_history = []
        self.collected_text = ""
        self.dimension_coverage = {"EI": 0, "SN": 0, "TF": 0, "JP": 0}
        self.total_word_count = 0
        
    def start_conversation(self):
        # The initial prompt is more natural and friendly
        initial_message = """
        Hi! I'm glad to chat with you. I want to get to know you better through a relaxed conversation.
        How have you been recently? Is there anything you'd like to share?
        """
        self.conversation_history.append({"role": "assistant", "content": initial_message})
        return initial_message
        
    def process_user_input(self, user_input):
        # Add the user's input to the history and collected text
        self.conversation_history.append({"role": "user", "content": user_input})
        self.collected_text += " " + user_input
        
        # Update the total word count
        self.total_word_count += count_words(user_input)
        
        # Analyze which MBTI dimensions are mentioned in the answer
        self._analyze_dimension_coverage(user_input)
        
        # Generate the next question/response
        next_question = self._generate_next_question()
        return next_question
    
    def _analyze_dimension_coverage(self, text):
        # Use a more balanced weight for dimension analysis
        for dim, info in mbti_dimensions.items():
            for keyword in info["keywords"]:
                if keyword.lower() in text.lower():
                    # Use a medium weight
                    self.dimension_coverage[dim] += 0.7
    
    def _generate_next_question(self):
        # Find the dimension with relatively less coverage (but not mandatory)
        least_covered = min(self.dimension_coverage, key=self.dimension_coverage.get)
        focus_dimension = mbti_dimensions[least_covered]["focus"]
        keywords = mbti_dimensions[least_covered]["keywords"]
        
        # Get the user's last message
        last_user_message = ""
        for msg in reversed(self.conversation_history):
            if msg["role"] == "user":
                last_user_message = msg["content"]
                break
        
        # Build the prompt to guide the LLM to generate a balanced question
        prompt = f"""
        {system_prompt}
        
        Current conversation history:
        {self._format_conversation_history()}
        
        Analysis:
        - Aspect that the user has less involved in: {focus_dimension}
        - Related keywords: {', '.join(keywords[:3])}
        - The user's last answer is: "{last_user_message}"
        
        Please generate a natural follow - up question or response based on the user's answer. First, respond to what the user just shared, and then guide to a topic that can naturally explore "{focus_dimension}", but don't point to this dimension obviously.
        
        Note:
        1. The response should show that you are listening carefully to the user's sharing
        2. The new topic should naturally extend from what the user just mentioned
        3. The question should indirectly explore the target dimension, but not obviously target psychological traits
        4. Maintaining the naturalness and smoothness of the conversation is a top priority
        5. Don't start with a greeting, just jump into the conversation directly
        6. Don't have prompts like ASSISTANT: or USER:
        
        The reply should be short, friendly, and like a conversation between friends. Don't explain why you are asking this question.
        """
        
        # Use the LLM to generate a response, moderately increase the temperature to make the response more natural
        next_message = self.llm(prompt, temperature=0.55, max_tokens=80)
        
        # Add to the conversation history
        self.conversation_history.append({"role": "assistant", "content": next_message})
        
        return next_message

    def _format_conversation_history(self):
        formatted = ""
        for msg in self.conversation_history:
            formatted += f"{msg['role'].upper()}: {msg['content']}\n\n"
        return formatted

    def get_personality_explanation(self, mbti_type):
        """Use the LLM to explain the MBTI analysis result"""
        prompt = f"""
        Based on the conversation content with the user:
        
        {self.collected_text[:1500]}... 
        
        The analysis result shows that the user may belong to the {mbti_type} personality type.
        
        Please provide the following:
        1. A brief description and main characteristics of this personality type
        2. Based on the actual content shared by the user, point out which characteristics best match this type
        3. The main advantages and growth directions of this personality type
        
        The answer should be friendly, personalized, and avoid overly academic expressions.
        """
        
        explanation = self.llm(prompt, temperature=0.2, max_tokens=100)
        return explanation

    def get_dimension_analysis(self):
        """Analyze the user's performance in the four dimensions"""
        # Here, use a simplified method to calculate dimension preferences based on keyword matching
        # In a real application, more complex NLP techniques can be used
        
        dimension_scores = {
            "E-I": 0,  # Positive value indicates E, negative value indicates I
            "S-N": 0,  # Positive value indicates S, negative value indicates N
            "T-F": 0,  # Positive value indicates T, negative value indicates F
            "J-P": 0   # Positive value indicates J, negative value indicates P
        }
        
        # Based on the collected text, use the LLM to analyze the dimension tendencies
        prompt = f"""
        Based on the user's following conversation content:
        
        {self.collected_text[:1500]}...
        
        Please analyze the user's tendencies in the four MBTI dimensions and provide a score from -10 to 10 for each dimension:
        1. Extroverted (E, +10) to Introverted (I, -10)
        2. Sensing (S, +10) to Intuitive (N, -10)
        3. Thinking (T, +10) to Feeling (F, -10)
        4. Judging (J, +10) to Perceiving (P, -10)
        
        Only return the four scores in the following format:
        E-I: X
        S-N: X
        T-F: X
        J-P: X
        
        Where X is an integer between -10 and 10.
        """
        
        try:
            analysis = self.llm(prompt)
            
            # Extract scores from the LLM's reply
            lines = analysis.strip().split('\n')
            for line in lines:
                if ':' in line:
                    dimension, score_str = line.split(':')
                    dimension = dimension.strip()
                    if dimension in dimension_scores:
                        try:
                            dimension_scores[dimension] = int(score_str.strip())
                        except ValueError:
                            pass
        except Exception as e:
            st.error(f"Error in dimension analysis: {e}")
        
        return dimension_scores

# Main application function
def main():
    # Initialize session state
    if "conversation_manager" not in st.session_state:
        # Load the model
        tokenizer, bert_model = load_bert_model()
        ollama_model = load_ollama_model("deepseek-r1:1.5b")  # Use the DeepSeek - R1 model
        
        # Initialize the conversation manager
        st.session_state.conversation_manager = DynamicConversationManager(
            ollama_model, bert_model, tokenizer, mbti_types
        )
        st.session_state.conversation_started = False
        st.session_state.analysis_complete = False
        st.session_state.messages = []
        st.session_state.chat_turns = 0
        st.session_state.word_count = 0
        st.session_state.max_word_count = 200  # Default maximum word count
    
    # Page title
    st.markdown("<h1 class='main-header'>MBTI Personality Analysis System</h1>", unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### About MBTI Personality Analysis")
        st.write("""
        MBTI (Myers - Briggs Type Indicator) is a personality type indicator that divides people's personalities into 16 different types based on preferences in four dimensions:
        - Extroverted (E) vs Introverted (I): Energy sources
        - Sensing (S) vs Intuitive (N): Information collection methods
        - Thinking (T) vs Feeling (F): Decision - making methods
        - Judging (J) vs Perceiving (P): Lifestyle
        """)
        if st.session_state.conversation_started and not st.session_state.analysis_complete:
            # Current word count/total word count display
            current_word_count = st.session_state.conversation_manager.total_word_count
            progress_text = f"Conversation progress: {current_word_count}/{st.session_state.max_word_count} words"
            progress_value = min(1.0, current_word_count / st.session_state.max_word_count)
            st.progress(progress_value, text=progress_text)
        
        # Ollama model selection
        if not st.session_state.conversation_started and not st.session_state.analysis_complete:
            model_options = {
                "deepseek-r1:1.5b": "DeepSeek R1 (1.5B) - Faster",
                "gemma3:4b": "Gemma 3 (4B) - More balanced",
                "llama3:8b": "Llama 3 (8B) - More accurate"
            }
            selected_model = st.selectbox(
                "Select Ollama model",
                options=list(model_options.keys()),
                format_func=lambda x: model_options[x]
            )
            if st.button("Test Ollama connection"):
                try:
                    test_llm = OllamaLLM(model_name=selected_model)
                    response = test_llm("Hello", max_tokens=10)
                    st.success(f"Ollama connection successful! Model returned: {response}")
                except Exception as e:
                    st.error(f"Ollama connection failed: {e}")
        
        elif st.button("Start a new conversation", key="restart_button"):
            # Reset session state
            tokenizer, bert_model = load_bert_model()
            ollama_model = load_ollama_model(selected_model if 'selected_model' in locals() else "deepseek-r1:1.5b")
            st.session_state.conversation_manager = DynamicConversationManager(
                ollama_model, bert_model, tokenizer, mbti_types
            )
            st.session_state.conversation_started = False
            st.session_state.analysis_complete = False
            st.session_state.messages = []
            st.session_state.chat_turns = 0
            st.session_state.word_count = 0
            st.rerun()
    
    # Main content area
    if not st.session_state.conversation_started:
        st.markdown("<div class='sub-header'>Analyze your MBTI personality type through natural conversation</div>", unsafe_allow_html=True)
        st.write("This system will have a relaxed conversation with you and then analyze your MBTI personality type based on your answers.")
        st.write("All analyses are performed locally, and your data will not be stored or shared.")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            start_button = st.button("Start conversation", use_container_width=True)
        with col2:
            max_word_count = st.number_input("Maximum word count", min_value=100, max_value=1000, value=200, 
                                           help="The conversation will end after collecting this many words (including Chinese and English)")
            
        if start_button:
            st.session_state.max_word_count = max_word_count
            st.session_state.conversation_started = True
            
            # If the model has been selected, update the model
            if 'selected_model' in locals():
                tokenizer, bert_model = load_bert_model()
                ollama_model = load_ollama_model(selected_model)
                st.session_state.conversation_manager = DynamicConversationManager(
                    ollama_model, bert_model, tokenizer, mbti_types
                )
                
            # Send the initial message
            initial_message = st.session_state.conversation_manager.start_conversation()
            st.session_state.messages.append({"role": "assistant", "content": initial_message})
            st.rerun()
    
    elif not st.session_state.analysis_complete:
        # Current word count/total word count display
        current_word_count = st.session_state.conversation_manager.total_word_count
        progress_text = f"Conversation progress: {current_word_count}/{st.session_state.max_word_count} words"
        progress_value = min(1.0, current_word_count / st.session_state.max_word_count)
        st.progress(progress_value, text=progress_text)
        
        # Display conversation history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # User input
        user_input = st.chat_input("Enter your reply here...")
        
        if user_input:
            # Add the user's message to the chat
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Process the user's input and get the next question
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.conversation_manager.process_user_input(user_input)
                st.write(response)
            
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.chat_turns += 1
            
            # Update the current word count
            current_word_count = st.session_state.conversation_manager.total_word_count
            
            # Check if enough words have been collected
            if current_word_count >= st.session_state.max_word_count:
                with st.spinner("Analyzing your personality type..."):
                    model_path = "./model/best_model1.pth"
                    collected_text = st.session_state.conversation_manager.collected_text
                    results_df = predict_mbti(collected_text, model_path)
                    print(results_df)
                    
                    # Extract the necessary information from the result DataFrame
                    primary_type = results_df.iloc[0]['mbti_type']
                    
                    # Calculate the overall confidence (can take the average of the four dimension scores)
                    scores = results_df.iloc[0][['score_EI', 'score_NS', 'score_TF', 'score_JP']]
                    pred_types = results_df.iloc[0][['pred_EI', 'pred_NS', 'pred_TF', 'pred_JP']]
                    
                    print(scores)
                    primary_confidence = scores.mean()
                    for i in range(4):
                        if pred_types[i] == 1:
                            scores[i] = ( scores[i] - 0.5 ) * 20
                        else:
                            scores[i] = -( scores[i] - 0.5 ) * 20
                    

                    # Get the personality explanation
                    # personality_explanation = st.session_state.conversation_manager.get_personality_explanation(primary_type)
                    
                    # Get the dimension analysis
                    #dimension_scores = st.session_state.conversation_manager.get_dimension_analysis()

                    # Get the matching results
                    match_results = match_main(primary_type)
                    print(match_results)
                    
                    
                    # Save the analysis results
                    st.session_state.analysis_results = {
                        "primary_type": primary_type,
                        "primary_confidence": primary_confidence,
                        # "top_types": top_types,
                        "explanation": match_results,
                        "dimension_scores": scores,
                        "collected_text": collected_text
                    }
                    
                    st.session_state.analysis_complete = True
                    st.rerun()
            st.rerun()    
    else:
        # Display the analysis results
        results = st.session_state.analysis_results
        
        st.markdown("<div class='sub-header'>Your MBTI Personality Analysis Results</div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 3])
        
        with col1:
            st.markdown(f"### 🏆 Primary Type: {results['primary_type']}")
            st.markdown(f"##### {mbti_descriptions[results['primary_type']]}")
            st.write(f"Confidence: {results['primary_confidence']:.1%}")
            
                
            # Add a button to download the analysis report
            report = f"""
            # MBTI Personality Analysis Report
            
            ## Primary Type: {results['primary_type']} - {mbti_descriptions[results['primary_type']]}
            Confidence: {results['primary_confidence']:.1%}
            
            
            ## Personality Description:
            {results['explanation']}
            
            ## Dimension Analysis:
            - Extroverted (E) vs Introverted (I): {results['dimension_scores']['score_EI']}
            - Sensing (S) vs Intuitive (N): {results['dimension_scores']['score_NS']}
            - Thinking (T) vs Feeling (F): {results['dimension_scores']['score_TF']}
            - Judging (J) vs Perceiving (P): {results['dimension_scores']['score_JP']}
            
            ## Conversation content based on analysis:
            {results['collected_text'][:500]}...
            """
            
            st.download_button(
                label="Download full analysis report",
                data=report,
                file_name=f"MBTI_Analysis_{results['primary_type']}.md",
                mime="text/markdown"
            )
            
        with col2:
            
            st.markdown("### 📊 Dimension Analysis")
            
            # Draw the dimension bar chart
            fig, ax = plt.subplots(figsize=(12, 6))
            
            dimensions = ['E vs I', 'N vs S', 'F vs T', 'P vs J']
            scores = [
                results['dimension_scores']['score_EI'],
                results['dimension_scores']['score_NS'],
                results['dimension_scores']['score_TF'],
                results['dimension_scores']['score_JP']
            ]
            
            # Create a horizontal bar chart
            bars = ax.barh(['E vs I', 'N vs S', 'F vs T', 'P vs J'], scores, color=['#3B82F6', '#10B981', '#F59E0B', '#6366F1'])
            
            # Set different colors for negative and positive values
            # for i, score in enumerate(scores):
            #     if score < 0:
            #         bars[i].set_color('#EF4444')
            
            # Set axes and grid
            ax.set_xlim(-10, 10)
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            ax.grid(axis='x', linestyle='--', alpha=0.3)
            
            # Add labels
            for i, dim in enumerate(dimensions):
                left, right = dim.split(' vs ')
                ax.text(-12, i, left, ha='right', va='center', fontweight='bold')
                ax.text(12, i, right, ha='left', va='center', fontweight='bold')
            
            # Beautify the chart
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_title('Your tendencies in the four MBTI dimensions', fontsize=14)
            
            # Display the chart
            st.pyplot(fig)
            
            # Display the personality explanation       
            st.markdown("### 🧠 Personality Description")
            st.write(results['explanation'])
            
            
        # Provide some follow - up suggestions
        st.markdown("<div class='sub-header' style='margin-top: 30px;'>Follow - up Suggestions</div>", unsafe_allow_html=True)
        st.write("""
        1. **Learn more**: In - depth understanding of your MBTI type and its characteristics can help you better understand yourself.
        2. **Career development**: Different MBTI types are suitable for different careers. Understanding your type can help you find a suitable career direction.
        3. **Interpersonal relationships**: Understanding your own and others' MBTI types can help you better communicate and cooperate with others.
        4. **Personal growth**: Each type has its own advantages and disadvantages. Understanding these can help you play to your strengths and make up for your weaknesses.
        """)

        with st.expander("About the System"):
            st.write("""
            This intelligent MBTI personality analysis system utilizes the following technologies:
            - **Natural Language Dialogue**: Uses large language models for natural conversational assessments
            - **BERT Deep Learning Model**: Used for MBTI personality classification
            - **Dimension Analysis**: Conducts detailed analysis of the four MBTI dimensions based on conversation content
            - **Streamlit Interface**: Provides a user-friendly interactive experience
            
            The system runs locally, and all data processing is completed on your device. Your conversation content will not be uploaded or stored.
            
            Note: The analysis provi
            ded by this system is for reference only and should not replace professional psychological assessments.
            """)
        
if __name__ == "__main__":
    os.environ["STREAMLIT_TEST_MODE"] = "true"
    main()
       