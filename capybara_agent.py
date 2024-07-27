from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from typing import List, Dict
import math
import faiss
import hashlib
import hmac
import time
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_experimental.generative_agents import GenerativeAgent, GenerativeAgentMemory
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from datetime import datetime
import os

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
if not BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN environment variable is not set")

LLM = ChatOpenAI(max_tokens=1500)

def relevance_score_fn(score: float) -> float:
    return 1.0 - score / math.sqrt(2)

def create_new_memory_retriever():
    embeddings_model = OpenAIEmbeddings()
    embedding_size = 1536
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = FAISS(
        embeddings_model.embed_query,
        index,
        InMemoryDocstore({}),
        {},
        relevance_score_fn=relevance_score_fn,
    )
    return TimeWeightedVectorStoreRetriever(
        vectorstore=vectorstore, other_score_keys=["importance"], k=15
    )

PERSONALITY_TYPES = {
    "Romantic Idealist": "Seeks deep emotional connections and believes in soulmates",
    "Adventure Seeker": "Thrives on new experiences and loves to explore",
    "Intellectual Companion": "Values deep conversations and mental stimulation",
    "Social Butterfly": "Energized by social interactions and meeting new people",
    "Nurturing Partner": "Caring, supportive, and focused on emotional well-being",
    "Ambitious Go-Getter": "Driven by goals and seeks a partner with similar ambitions",
    "Creative Spirit": "Expresses themselves through art and values originality",
    "Steady Reliable": "Consistent, dependable, and values stability in relationships"
}

QUESTION_TEMPLATES = [
    "What's your idea of a perfect date?",
    "How do you handle conflicts in a relationship?",
    "What's the most spontaneous thing you've ever done?",
    "How important is personal space to you in a relationship?",
    "What's a deal-breaker for you in a potential partner?",
    "How do you show affection to someone you care about?",
    "What role does humor play in your relationships?",
    "How do you balance your personal goals with a romantic relationship?",
    "What's the most important quality you look for in a partner?",
    "How do you envision your ideal future with a partner?"
]

class QuestionGeneratorAgent(GenerativeAgent):
    def __init__(self, name: str, age: int, traits: str):
        memory = GenerativeAgentMemory(
            llm=LLM,
            memory_retriever=create_new_memory_retriever(),
            verbose=False,
            reflection_threshold=5
        )
        super().__init__(name=name, age=age, traits=traits, status="active", memory=memory, llm=LLM)
        logger.info(f"Initialized QuestionGeneratorAgent: {name}, age {age}, traits: {traits}")

    def generate_question(self, previous_questions: List[str], previous_answers: List[str], telegram_data: Dict) -> str:
        logger.info(f"Generating new question for Telegram user: {telegram_data.get('id')}")
        logger.debug(f"Previous questions: {previous_questions}")
        logger.debug(f"Previous answers: {previous_answers}")
        logger.debug(f"Telegram data: {telegram_data}")
        
        context = "\n".join([f"Q: {q}\nA: {a}" for q, a in zip(previous_questions, previous_answers)])
        templates = "\n".join([f"- {template}" for template in QUESTION_TEMPLATES])
        prompt = f"""As a dating app personality quiz, generate a new, interesting question based on the following context and question templates:

Previous Questions and Answers:
{context}

Question Templates:
{templates}

Create a unique question inspired by these templates, but don't repeat them exactly. The question should be engaging, thought-provoking, and reveal aspects of the person's dating personality. Ensure it's different from the previous questions.

Generate only the question, nothing else."""

        logger.debug(f"Prompt for question generation: {prompt}")
        response = self.generate_dialogue_response(prompt)
        question = response[1].strip('"')
        logger.info(f"Generated question: {question}")
        return question

class PersonalityClassifierAgent(GenerativeAgent):
    def __init__(self, name: str, age: int, traits: str):
        memory = GenerativeAgentMemory(
            llm=LLM,
            memory_retriever=create_new_memory_retriever(),
            verbose=False,
            reflection_threshold=5
        )
        super().__init__(name=name, age=age, traits=traits, status="active", memory=memory, llm=LLM)
        logger.info(f"Initialized PersonalityClassifierAgent: {name}, age {age}, traits: {traits}")

    def classify_personality(self, questions: List[str], answers: List[str], telegram_data: Dict) -> dict:
        logger.info(f"Classifying personality for Telegram user: {telegram_data.get('id')}")
        logger.debug(f"Questions for classification: {questions}")
        logger.debug(f"Answers for classification: {answers}")
        logger.debug(f"Telegram data: {telegram_data}")
        
        context = "\n".join([f"Q: {q}\nA: {a}" for q, a in zip(questions, answers)])
        personality_types = "\n".join([f"- {type}: {desc}" for type, desc in PERSONALITY_TYPES.items()])
        prompt = f"""As a dating app personality classifier, categorize the person's dating personality based on these questions and answers:

{personality_types}

Questions and Answers:
{context}

Provide the personality type and a brief explanation for your choice. Also, suggest potential compatible personality types for dating. Format your response as:
Personality Type: [chosen type]
Explanation: [your explanation]
Compatible Types: [list of compatible types]
Dating Advice: [brief advice based on their personality type]
"""
        logger.debug(f"Prompt for personality classification: {prompt}")
        response = self.generate_dialogue_response(prompt)
        personality_summary = response[1]
        logger.info(f"Generated personality summary: {personality_summary}")

        # Save the personality summary to a text file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"personality_summary_{telegram_data.get('id')}_{timestamp}.txt"
        with open(filename, "w") as f:
            f.write("Capybara Dating App - Personality Summary\n")
            f.write("=========================================\n\n")
            f.write(f"Telegram User ID: {telegram_data.get('id')}\n")
            f.write(f"Username: {telegram_data.get('username')}\n\n")
            f.write("Questions and Answers:\n")
            for q, a in zip(questions, answers):
                f.write(f"Q: {q}\n")
                f.write(f"A: {a}\n\n")
            f.write("Personality Classification:\n")
            f.write(personality_summary)
        logger.info(f"Saved personality summary to file: {filename}")

        return {"personality": personality_summary, "summary_file": filename}

question_agent = QuestionGeneratorAgent("CapybaraQuestionBot", 25, "curious, romantic, empathetic")
classifier_agent = PersonalityClassifierAgent("CapybaraMatchBot", 30, "insightful, compassionate, intuitive")

def validate_telegram_data(data: Dict) -> bool:
    if 'hash' not in data:
        logger.error("No hash found in Telegram data")
        return False

    received_hash = data['hash']
    
    data_to_check = data.copy()
    data_to_check.pop('hash', None)
    data_check_string = '\n'.join(sorted(f'{k}={v}' for k, v in data_to_check.items()))
    secret_key = hashlib.sha256(BOT_TOKEN.encode()).digest()
    computed_hash = hmac.new(secret_key, data_check_string.encode(), hashlib.sha256).hexdigest()
    
    if computed_hash != received_hash:
        logger.error("Invalid hash in Telegram data")
        return False
    

    if time.time() - int(data['auth_date']) > 3600:
        logger.error("Telegram data is outdated")
        return False
    
    return True

@app.route('/validate-telegram-login', methods=['POST'])
def validate_telegram_login():
    data = request.json
    logger.info(f"Received Telegram login data for validation: {data}")
    
    if validate_telegram_data(data):
        logger.info("Telegram login data validated successfully")
        return jsonify({"isValid": True})
    else:
        logger.warning("Telegram login data validation failed")
        return jsonify({"isValid": False}), 400

@app.route('/generate_question', methods=['POST'])
def generate_question():
    logger.info("Received request to generate question")
    data = request.json
    previous_questions = data.get('previous_questions', [])
    previous_answers = data.get('previous_answers', [])
    telegram_data = data.get('telegram_data', {})

    if not validate_telegram_data(telegram_data):
        logger.error("Invalid Telegram data in question generation request")
        return jsonify({"error": "Invalid Telegram data"}), 400

    logger.debug(f"Previous questions: {previous_questions}")
    logger.debug(f"Previous answers: {previous_answers}")
    logger.debug(f"Telegram data: {telegram_data}")
    
    question = question_agent.generate_question(previous_questions, previous_answers, telegram_data)
    logger.info(f"Generated question: {question}")
    return jsonify({'question': question})

@app.route('/classify_personality', methods=['POST'])
def classify_personality():
    logger.info("Received request to classify personality")
    data = request.json
    questions = data.get('questions', [])
    answers = data.get('answers', [])
    telegram_data = data.get('telegram_data', {})

    if not validate_telegram_data(telegram_data):
        logger.error("Invalid Telegram data in personality classification request")
        return jsonify({"error": "Invalid Telegram data"}), 400

    logger.debug(f"Questions for classification: {questions}")
    logger.debug(f"Answers for classification: {answers}")
    logger.debug(f"Telegram data: {telegram_data}")
    
    result = classifier_agent.classify_personality(questions, answers, telegram_data)
    logger.info(f"Personality classification result: {result}")
    return jsonify(result)

if __name__ == '__main__':
    logger.info("Starting Capybara Dating App API")
    app.run(debug=True)
