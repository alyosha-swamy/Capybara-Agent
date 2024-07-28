from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from datetime import datetime
import os
import hmac
import hashlib
import psycopg2
from psycopg2.extras import Json

from langchain_openai import ChatOpenAI
from langchain_core.runnables import Runnable
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

app = Flask(__name__)
# CORS(app)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

LLM = ChatOpenAI(model_name="gpt-4o-mini")
load_dotenv()


# Telegram Bot Token (use environment variable in production)
BOT_TOKEN = os.environ.get(
    'BOT_TOKEN', "dummy_bot_token_for_testing_1234567890")
# PostgreSQL connection parameters


def get_db_connection():
    return psycopg2.connect(
        dbname=os.environ.get("DB_NAME"),
        user=os.environ.get("DB_USER"),
        password=os.environ.get("DB_PASSWORD"),
        host=os.environ.get("DB_HOST"),
        port=os.environ.get("DB_PORT")
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
    "Can you describe a moment when you felt truly loved and appreciated in a relationship? What made it special?",
    "How do you prefer to receive affection from your partner? Words of affirmation, acts of service, receiving gifts, quality time, or physical touch?",
    "When you're feeling stressed or overwhelmed, how would you like your partner to support you?",
    "What are some ways you like to express love and appreciation to your partner?",
    "How do you handle misunderstandings or disagreements with your partner? Can you provide an example?",
    "What are your expectations for communication in a relationship, especially during conflicts?",
    "How important is it for you to share hobbies or interests with your partner? Why?",
    "Describe a perfect weekend with your partner. What activities would you do together?",
    "How do you maintain your individuality while being in a committed relationship?",
    "What is one significant lesson you've learned from past relationships that you would bring into a new relationship?",
    "How do you envision balancing career aspirations and a romantic relationship?",
    "What role does mutual respect play in your ideal relationship, and how do you show respect to your partner?",
    "How do you prefer to celebrate special occasions, like anniversaries or birthdays, with your partner?",
    "What is your approach to financial planning and management in a relationship?",
    "How do you ensure emotional and physical intimacy is maintained in a long-term relationship?"
]


class QuestionGeneratorTool(Runnable):
    def __init__(self):
        self.model = LLM
        self.prompt = ChatPromptTemplate.from_template(
            """
"As a dating app personality quiz, generate a new, interesting question based on the following context and question templates:

Previous Questions and Answers:
{context}

Question Templates:
{templates}

Create a unique question inspired by these templates, but don't repeat them exactly. The question should be engaging, thought-provoking, and reveal aspects of the person's dating personality. Ensure it's different from the previous questions.

Generate only the question, nothing else."""
        )

    def invoke(self, inputs: dict) -> dict:
        context = "\n".join([f"Q: {q}\nA: {a}" for q, a in zip(
            inputs['previous_questions'], inputs['previous_answers'])])
        templates = "\n".join(
            [f"- {template}" for template in QUESTION_TEMPLATES])
        prompt_text = self.prompt.invoke(
            {'context': context, 'templates': templates})
        response = self.model.invoke(prompt_text)
        print("Response metadata:")
        print(response.response_metadata)
        return {
            "question": response.content,
        }


class PersonalityClassifierTool(Runnable):
    def __init__(self):
        self.model = LLM
        self.prompt_template = ChatPromptTemplate.from_template(
            """As a dating app personality classifier, categorize the person's dating personality based on these questions and answers:

{personality_types}

Questions and Answers:
{context}

Provide the personality type and a brief explanation for your choice. Also, suggest potential compatible personality types for dating. Format your response as:
Personality Type: [chosen type]
Explanation: [your explanation]
Compatible Types: [list of compatible types]
Dating Advice: [brief advice based on their personality type]"""
        )

    def invoke(self, inputs: dict) -> dict:
        context = "\n".join([f"Q: {q}\nA: {a}" for q, a in zip(
            inputs['questions'], inputs['answers'])])
        personality_types = "\n".join(
            [f"- {type}: {desc}" for type, desc in PERSONALITY_TYPES.items()])
        print(personality_types)

        prompt_text = self.prompt_template.invoke(
            {'personality_types': personality_types, 'context': context})
        response = self.model.invoke(prompt_text)
        # usage = response.usage
        # Parse response (assuming response format as described)
        lines = response.content.split('\n')
        personality_type = next((line.split(
            ': ')[1] for line in lines if line.startswith('Personality Type:')), '')
        explanation = next(
            (line.split(': ')[1] for line in lines if line.startswith('Explanation:')), '')
        compatible_types = [t.strip() for t in next((line.split(': ')[1].split(
            ', ') for line in lines if line.startswith('Compatible Types:')), [])]
        dating_advice = next(
            (line.split(': ')[1] for line in lines if line.startswith('Dating Advice:')), '')

        return {
            "personality_type": personality_type.strip(),
            "explanation": explanation.strip(),
            "compatible_types": compatible_types,
            "dating_advice": dating_advice.strip(),
            # "usage": usage
        }


question_generator = QuestionGeneratorTool()
personality_classifier = PersonalityClassifierTool()

# Define a chain for question generation and personality classification
# chain = RunnableSequence([
#     question_generator,
#     personality_classifier
# ])


@app.route('/generate_question', methods=['POST'])
def generate_question():
    logger.info("Received request to generate question")
    data = request.json
    # init_data = data.get('initData')

    # if not validate_telegram_data(init_data):
    #     logger.warning("Invalid Telegram data received")
    #     return jsonify({'error': 'Invalid data'}), 400

    previous_questions = data.get('previous_questions', [])
    previous_answers = data.get('previous_answers', [])
    telegram_data = data.get('telegram_data', {})
    logger.debug(f"Previous questions: {previous_questions}")
    logger.debug(f"Previous answers: {previous_answers}")
    logger.debug(f"Telegram data: {telegram_data}")

    response = question_generator.invoke({
        'previous_questions': previous_questions,
        'previous_answers': previous_answers
    })

    logger.info(f"Generated response: {response}")
    return jsonify({'question': response["question"]})


@app.route('/classify_personality', methods=['POST'])
def classify_personality():
    logger.info("Received request to classify personality")
    data = request.json
    init_data = data.get('initData')

    # if not validate_telegram_data(init_data):
    #     logger.warning("Invalid Telegram data received")
    #     return jsonify({'error': 'Invalid data'}), 400
    telegram_user_id = init_data.get('id')
    quiz_data = {}

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                print(telegram_user_id)
                cur.execute("""
                                SELECT personality_type, explanation, compatible_types, dating_advice
                                FROM personality_summaries
                                WHERE telegram_user_id = %s
                            """, (telegram_user_id,))
                print("Personality data fetched")
                quiz_data = cur.fetchone()
                if (quiz_data is not None):
                    logger.info("Personality classification already exists")
                    print(quiz_data)
                    quiz_data = {
                        'personality_type': quiz_data[0],
                        'explanation': quiz_data[1],
                        'compatible_types': quiz_data[2],
                        'dating_advice': quiz_data[3]
                    }
                else:
                    questions = data.get('questions', [])
                    answers = data.get('answers', [])

                    result = personality_classifier.invoke({
                        'questions': questions,
                        'answers': answers,
                    })
                    logger.info(f"Personality classification result: {result}")

                    cur.execute("""INSERT INTO personality_summaries (telegram_user_id, personality_type, explanation, compatible_types, dating_advice) VALUES (%s, %s, %s, %s, %s)
                                """, (telegram_user_id, result["personality_type"], result["explanation"], result["compatible_types"], result["dating_advice"]))
                    quiz_data = {
                        "personality_type": result["personality_type"],
                        "explanation": result["explanation"],
                        "compatible_types": result["compatible_types"],
                        "dating_advice": result["dating_advice"]
                    }

        return jsonify(quiz_data)
    except Exception as e:
        logger.error(f"Error fetching user data from database: {e}")
        return jsonify({'error': 'Internal server error'}), 500


def generate_secret_key(bot_token: str) -> bytes:
    return hmac.new("WebAppData".encode(), bot_token.encode(), hashlib.sha256).digest()


def validate_telegram_data(parsed_data: dict) -> bool:
    try:
        received_hash = parsed_data.pop('hash', None)
        if not received_hash:
            return False
        print(sorted(parsed_data.items()))
        data_check_string = '/n'.join([f"{k}={v}" for k,
                                      v in sorted(parsed_data.items())])
        print(data_check_string)
        secret_key = generate_secret_key(BOT_TOKEN)
        calculated_hash = hmac.new(
            secret_key, data_check_string.encode(), hashlib.sha256).hexdigest()
        return calculated_hash == received_hash[0]
    except Exception as e:
        logger.error(f"Error validating Telegram data: {e}")
        return False


@app.route('/tg-auth', methods=['POST'])
def telegram_auth():
    data = request.json
    user_data = data.get('initData')
    # if not validate_telegram_data(init_data):
    #     logger.warning("Invalid Telegram data received")
    #     return jsonify({'error': 'Invalid data'}), 403

    # Store user data in the database
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT telegram_user_id, first_name, last_name, photo_url, auth_date
                    FROM users
                    WHERE telegram_user_id = %s
                """, (user_data['id'],))
                existing_user = cur.fetchone()

                if existing_user:
                    user_id, first_name, last_name, photo_url, auth_date = existing_user
                    logger.info(
                        f"Existing user data in database with ID: {user_id}")
                else:
                    cur.execute("""
                        INSERT INTO users (telegram_user_id, first_name, last_name, photo_url, auth_date)
                        VALUES (%s, %s, %s, %s, %s)
                        RETURNING telegram_user_id, first_name, last_name, photo_url, auth_date
                    """, (
                        user_data['id'],
                        user_data['first_name'],
                        user_data.get('last_name'),
                        user_data.get('photo_url'),
                        datetime.fromtimestamp(
                            int(user_data['auth_date'][0]))
                    ))
                    new_user = cur.fetchone()
                    user_id, first_name, last_name, photo_url, auth_date = new_user
                    conn.commit()
                    logger.info(
                        f"Stored or updated user data in database with ID: {user_id}")

            return jsonify({
                'id': user_id,
                'first_name': first_name,
                'last_name': last_name,
                'photo_url': photo_url,
                'auth_date': auth_date
            })

    except Exception as e:
        logger.error(f"Error storing user data in database: {e}")

    return jsonify({
        'id': user_data['id'],
        'first_name': user_data['first_name'],
        'username': user_data.get('username'),
        'photo_url': user_data.get('photo_url'),
        'auth_date': user_data['auth_date']
    })


@app.route("/onboarding", methods=["POST"])
def onboarding():
    data = request.json
    init_data = data.get('initData')

    # if not validate_telegram_data(init_data):
    #     logger.warning("Invalid Telegram data received")
    #     return jsonify({'error': 'Invalid data'}), 403
    onboarding_data = data.get('data', {})
    telegram_user_id = init_data.get('id')

    gender = onboarding_data.get("gender")
    sexuality = onboarding_data.get("sexuality")
    family_plans = onboarding_data.get("familyPlans")
    personal_values = onboarding_data.get("personalValues")
    interests = onboarding_data.get("interests")

    logger.info(
        f"Received onboarding data for Telegram user: {telegram_user_id}")
    # Store these onboarding data into the database for the user and move onboarding to true
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT telegram_user_id
                    FROM onboarding
                    WHERE telegram_user_id = %s
                """, (telegram_user_id,))
                onboarding_completed = cur.fetchone()
                if onboarding_completed is None:
                    try:
                        cur.execute("""
                            INSERT INTO onboarding
                            (telegram_user_id, gender, sexuality, family_plans, personal_values, interests)
                            VALUES (%s, %s, %s, %s, %s, %s)
                        """, (telegram_user_id, gender, sexuality, family_plans, Json(personal_values), Json(interests)))
                        conn.commit()
                        logger.info(
                            f"Inserted onboarding data for Telegram user: {telegram_user_id}")
                    except Exception as e:
                        logger.error(
                            f"Error storing onboarding data in database: {e}")
                        return jsonify({'error': 'Internal server error'}), 500
                else:
                    return jsonify({'error': 'User already onboarded'}), 400

        return jsonify({
            'gender': gender,
            'sexuality': sexuality,
            'family_plans': family_plans,
            'personal_values': personal_values,
            'interests': interests
        })

    except Exception as e:
        logger.error(f"Error storing onboarding data in database: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/overview', methods=['POST'])
def overview():
    data = request.json
    init_data = data.get('initData')

    # if not validate_telegram_data(init_data):
    #     logger.warning("Invalid Telegram data received")
    #     return jsonify({'error': 'Invalid data'}), 400

    telegram_user_id = init_data.get('id')

    try:
        with get_db_connection() as conn:
            onboarding_completed = False
            quiz_completed = False
            onboarding_data = {}
            quiz_data = {}
            with conn.cursor() as cur:
                # Check if the user has completed onboarding and quiz
                cur.execute("""
                            SELECT gender, sexuality, family_plans, personal_values, interests
                            FROM onboarding
                            WHERE telegram_user_id = %s
                        """, (telegram_user_id,))
                onboarding_data = cur.fetchone()
                if onboarding_data is not None:
                    onboarding_completed = True
                    onboarding_data = {
                        'gender': onboarding_data[0],
                        'sexuality': onboarding_data[1],
                        'family_plans': onboarding_data[2],
                        'personal_values': onboarding_data[3],
                        'interests': onboarding_data[4]
                    }
                    cur.execute("""
                                SELECT personality_type, explanation, compatible_types, dating_advice
                                FROM personality_summaries
                                WHERE telegram_user_id = %s
                            """, (telegram_user_id,))
                    quiz_data = cur.fetchone()
                    if (quiz_data is not None):
                        quiz_completed = True
                        quiz_data = {
                            'personality_type': quiz_data[0],
                            'explanation': quiz_data[1],
                            'compatible_types': quiz_data[2],
                            'dating_advice': quiz_data[3]
                        }

            return jsonify({
                'onboarding_completed': onboarding_completed,
                'quiz_completed': quiz_completed,
                'onboarding_data': onboarding_data,
                'quiz_data': quiz_data
            })

    except Exception as e:
        logger.error(f"Error fetching user data from database: {e}")
        return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    logger.info("Starting Capybara Dating App API")
    app.run(port=6000, debug=True)
