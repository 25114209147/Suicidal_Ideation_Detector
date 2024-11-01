# 🌟 Suicidal Ideation Detection

## 📖 Project Overview
This project aims to develop machine learning models that utilize linguistic patterns extracted using Natural Language Processing (NLP) from Reddit data to detect suicidal ideation. The primary goal is to enhance early detection and provide real-time monitoring support for individuals at risk.


## 🎯 Research Objectives
1. Develop Machine Learning Models: Create models using linguistic features extracted from Reddit data to predict suicidal ideation.
2. Evaluate Model Effectiveness: Assess the performance of various models to identify the most effective one.
3. Design a Real-Time Monitoring System: Implement a user-friendly system to provide real-time support and monitoring for individuals expressing suicidal thoughts.


## 🏆 Results
Among all the models created, DistilBERT performed best and was used for deployment using Streamlit. The user interface offers real-time monitoring feature and integrates with Gemini to provide advice if the text is detected as SUICIDAL.


## 🔑 Required Credentials
To run this project, you need the following credentials:
- `GEMINI_API_KEY`: Your API key for the service.
- `GOOGLE_CLIENT_ID`: Your Google OAuth client ID.
- `GOOGLE_CLIENT_SECRET`: Your Google OAuth client secret.
- `REDIRECT_URI`: The redirect URI for OAuth.
- `CLIENT_ID`: Your Reddit API client ID.
- `CLIENT_SECRET`: Your Reddit API client secret.
- `OPENAI_API_KEY`: Your OpenAI API key.

*Don’t worry if you don’t have these yet! You can easily create accounts for these services to get started.*

## 🛠️ Installation
To install the necessary dependencies, ensure you have Python 3.8 or higher, and then run:

### 🎉 Run Locally:
```bash
pip install -r requirements.txt
streamlit run app.py --server.port=8080 --server.address=0.0.0.0
```

### 🐳 Run using Docker:
```bash
docker build -t streamlit .
docker run -p 8080:8080 streamlit
```

## 💬 Conclusion
This project showcases the powerful intersection of linguistic analysis and machine learning to detect suicidal ideation. By providing timely support, we can make a difference in the lives of those in need. Together, we can help ensure that no one feels alone or overlooked. 💖
