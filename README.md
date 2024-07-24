# Capybara Agent: AI-Powered Personality Quiz

Capybara Agent is an interactive personality quiz that uses AI to generate questions and classify users' personalities. It's built with Python (Flask) for the backend and HTML/JavaScript for the frontend.

## Introduction Video

<iframe width="560" height="315" src="https://www.youtube.com/embed/Zn7A01MHLOw?si=lOZcZbOdiJtU60pL" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

## Features

- AI-powered Capybara Agent generates tailored questions based on previous responses
- Personality classification by the Capybara Agent based on user answers
- Saves personality summaries to text files
- Interactive web interface featuring the Capybara Agent

## Requirements

- Python 3.8+
- Flask
- Flask-CORS
- LangChain
- OpenAI API key

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/capybara-agent.git
   cd capybara-agent
   ```
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Set your Capx API key as an environment variable:
   ```
   export Capx_API_KEY='your-api-key-here'
   ```

## Usage

1. Start the Capybara Agent server:
   ```
   python capybara_agent.py
   ```
2. Open `capybara_agent.html` in your web browser.
3. Interact with the Capybara Agent by answering its questions.
4. After answering 5 questions, you'll receive your personality classification from the Capybara Agent.
5. A summary of your responses and personality classification will be saved as a text file in the project directory.

## File Structure

- `capybara_agent.py`: Flask backend with Capybara Agent logic
- `capybara_agent.html`: Frontend HTML/JavaScript featuring the Capybara Agent
- `requirements.txt`: List of Python dependencies
- `README.md`: This file
- `capybara.mp4`: Introductory video about the Capybara Agent

## Contributing

Contributions to improve the Capybara Agent are welcome! Please feel free to submit pull requests or open issues to discuss potential enhancements.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
