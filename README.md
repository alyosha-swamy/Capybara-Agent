# Capybara Dating App

![Capybara Dating App Demo](capybara.mp4)

Capybara Dating App is an interactive personality quiz that uses AI to generate questions and classify users' dating personalities. It's built with Python (Flask) for the backend and HTML/JavaScript for the frontend.

## Features

- AI-generated questions tailored to previous responses
- Personality classification based on user answers
- Saves personality summaries to text files
- Interactive web interface

## Requirements

- Python 3.8+
- Flask
- Flask-CORS
- LangChain
- OpenAI API key

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/capybara-dating-app.git
   cd capybara-dating-app
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set your OpenAI API key as an environment variable:
   ```
   export OPENAI_API_KEY='your-api-key-here'
   ```

## Usage

1. Start the Flask server:
   ```
   python capx/capybara_dating_app_api.py
   ```

2. Open `capx/capybara_dating_app.html` in your web browser.

3. Answer the questions presented by the app.

4. After answering 5 questions, you'll receive your personality classification.

5. A summary of your responses and personality classification will be saved as a text file in the project directory.

## File Structure

- `capybara_dating_app_api.py`: Flask backend
- `capybara_dating_app.html`: Frontend HTML/JavaScript
- `requirements.txt`: List of Python dependencies
- `README.md`: This file
- `capybara.mp4`: Introductory video about the app