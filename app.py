import warnings
warnings.filterwarnings("ignore")
import streamlit as st
import praw
from urllib.parse import urlencode
import pandas as pd
import joblib
import preprocessor as pp2
import os
from dotenv import load_dotenv
import google.generativeai as ggi
from google.api_core.exceptions import ServiceUnavailable
from google.oauth2 import id_token
from google.auth.transport import requests
import folium
from streamlit_folium import folium_static, st_folium
import streamlit.components.v1 as components
from datetime import datetime
import threading
import time
import queue  

# Load environment variables
load_dotenv(".env")
    
# For Gemini
# Fetch API key from Streamlit secrets
fetched_api_key = st.secrets["API_KEY"]
ggi.configure(api_key=fetched_api_key)

# Set page configuration at the beginning
st.set_page_config(page_title="Suicidal Ideation Detection", page_icon="💬", layout="wide")


# Retrieve query parameters
# Initialize session state for navigation
if 'page' not in st.session_state:
    st.session_state.page = 'Main Page'



from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

# Load the BERT model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert_suicidal_classifier")
bert_model = DistilBertForSequenceClassification.from_pretrained("distilbert_suicidal_classifier")
bert_model = bert_model.half()

# Set up the device for BERT
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model.to(device)
bert_model.eval()  



# Define the function to navigate to Chatbot page
def navigate_to_chatbot():
    st.session_state.page = "Chatbot"  # Set session state to navigate to Chatbot page
    st.query_params.update(page="Chatbot")

# Main page code
def main_page():
    st.header("🏠 Welcome to")
    st.title("Suicidal Ideation Detector")
    st.divider()

    # feature_extractor = FeatureExtractor()

    # Main container
    with st.container():
        st.markdown('<div class="main">', unsafe_allow_html=True)
        
        # Selection of input method
        st.subheader("Input Method")
        input_method = st.radio(
            "*Choose here:*",
            options=["Text Input", "File Upload", "Example Text"], horizontal=True)
        
        user_text = ""
        if input_method == "Text Input":
            user_text = st.text_area("Enter text to check for suicidal content and issue category:", help="Press 'Submit' button to analyze the text.")
        elif input_method == "File Upload":
            uploaded_file = st.file_uploader("Upload a text file for analysis", type=["txt"])
            if uploaded_file is not None:
                file_contents = uploaded_file.read().decode("utf-8")
                st.write("File uploaded successfully!")
                st.text_area("File Content", file_contents, height=100)
                user_text = file_contents
        elif input_method == "Example Text":
            options = [
                "It’s hard to see a future when the present feels so heavy and endless. It feels like there's no way out.",
                "I'm feeling completely overwhelmed, and I just don't see the point in staying with my family anymore. It feels like everything would be better if I just disappeared.",
                "The pressure from school is so intense that I sometimes think it would be easier to end it all.",
                "I am feeling so lonely..maybe I shouldn't exist..",
                "Finally! Today will be my last therapy session, and I'm feeling so proud of the progress I've made. I'm excited to continue applying what I've learned in my everyday life.",
                "The Olympics results are really exciting!",
                "I love to play badminton with my family.",
                "I am moving into my new house today!"
            ]
            selected_option = st.selectbox("Choose a sample text for analysis", options, index=None)
            st.write("Selected text:", selected_option)
            user_text = selected_option
        
        submit_button = st.button("Submit")

        if submit_button and user_text:
            current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            predicted_label, advice = pp2.predict(user_text, bert_model, current_datetime)
            st.subheader(f"The text is classified as: **{predicted_label.upper()}**")

            st.divider()

            if predicted_label == "suicidal":
                if advice:
                    st.write(advice)

                st.button("Need more advice? Chat with us", on_click=navigate_to_chatbot)
            
        elif submit_button:
            st.write('Please enter text or choose a file to analyze.')



# For Gemini
# Function to get response from Google Gemini
model = ggi.GenerativeModel("gemini-pro") 
chat = model.start_chat()

# Customized response for sensitive content
def custom_suicide_response():
    return (
        "I am so sorry to hear that you are feeling this way. I can't imagine how difficult it must be to feel this low. "
        "Please know that there are people who care about you and want to help. If you are feeling suicidal, please reach out for help. "
        "There are many resources available to you. You can call or text the Sage Centre at 012-339 7121. "
        "Find a helpline at https://findahelpline.com/countries/my."
    )

# Function to check for suicidal ideation
def is_suicidal(text):
    suicidal_keywords = ["want to end my life", "suicidal", "kill myself", "want to die", "can't go on", "give up", "no point in living", 'wanna die',
                         'I just can’t take it anymore', 'I wish I were dead', 'There is no way out', 'All my problems will end soon']
    return any(keyword in text.lower() for keyword in suicidal_keywords)

# Function to get response from Google Gemini
def LLM_Response(question):
    try:
        model = ggi.GenerativeModel("gemini-pro")
        chat = model.start_chat()
        response = chat.send_message(question, stream=True)
        return response
    except ServiceUnavailable:
        return None

# Chatbot page content
def chatbot_page():
    st.title("🤖Chatbot")
    st.divider()
    st.markdown("###### Hi! Type a message and press Enter if you need more info or want to chat with me!")

    # Initialize session state for messages and history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Type something"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Check for suicidal content
        if is_suicidal(prompt):
            response_text = custom_suicide_response()
        else:
            # Get assistant response from Google Gemini
            response = LLM_Response(prompt)
            if response is None:
                response_text = "The prompt was flagged for potentially harmful content. Please try a different input."
            else:
                try:
                    response_text = ''.join([part.text for part in response])
                except ValueError:
                    response_text = "The prompt was flagged for potentially harmful content. Please try a different input."

        # Add assistant response to chat historyT
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        with st.chat_message("assistant"):
            st.markdown(response_text)



# Help page code
def help_page():
    st.title("💡 Help and Resources")
    st.divider()
    # Create a smaller DataFrame with the specified countries
    data = {
    'country': ['United States', 'India', 'United Kingdom', 'Canada', 'Brazil', 
                'Philippines', 'Australia', 'Germany', 'Spain', 'France', 'Malaysia'],
    'emergency': ['911', '112', '999', '911', '188', 
                  '911', '000', '112', '112', '112', '999'],
    'suicide_hotline': ['988', '8888817666', '0800 689 5652', '1 (833) 456 4566', '188', 
                        '028969191', '131114', '0800 111 0 111', '914590050', '0145394000', '(06) 2842500'],
    'hotline_link': ['https://988lifeline.org', 'http://behtarindia.com', 'https://www.supportlineuk.org.uk/', 
                     'https://www.crisisservicescanada.ca/en/', 'http://argentina.gov.ar', 
                     'http://www.e-tu.org/pl', 'https://www.lifeline.org.au', 
                     'https://www.telefonseelsorge.de', 'https://www.supportlineuk.org.uk/', 
                     'https://peaasi.ee/en', 'https://helpline.mv']
    }

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Add coordinates for the specified countries
    country_coords = {
        'United States': [37.0902, -95.7129],
        'India': [20.5937, 78.9629],
        'United Kingdom': [55.3781, -3.4360],
        'Canada': [56.1304, -106.3468],
        'Brazil': [-14.2350, -51.9253],
        'Philippines': [12.8797, 121.7740],
        'Australia': [-25.2744, 133.7751],
        'Germany': [51.1657, 10.4515],
        'Spain': [40.4637, -3.7492],
        'France': [46.6034, 1.8883],
        'Malaysia': [4.2105, 101.9758]
    }

    df['latitude'] = df['country'].apply(lambda x: country_coords[x][0])
    df['longitude'] = df['country'].apply(lambda x: country_coords[x][1])

    st.markdown("#### Interactive Map: Suicide Hotlines by Country")
    # Adding interactivity to select and show hotline
    st.markdown("Click on a location marker to view the hotline information.")
    
    # Create map
    m = folium.Map(location=[20, 0], zoom_start=2)

    for index, row in df.iterrows():
        country = row['country']
        suicide_hotline = row['suicide_hotline']
        hotline_link = row['hotline_link']
        emergency = row['emergency']
        latitude = row['latitude']
        longitude = row['longitude']
        
        if pd.notna(latitude) and pd.notna(longitude):
            popup_text = f"<b>Country:</b> {country}<br>" \
                         f"<b>Suicide Hotline:</b> {suicide_hotline}<br>" \
                         f"<b>Emergency Hotline:</b> {emergency}<br>" \
            
            folium.Marker(
                location=[latitude, longitude],
                popup=folium.Popup(popup_text, max_width=300)
            ).add_to(m)
    
    folium_static(m, width=980, height=600)
    st.caption("_Displayed are countries with the top 10 highest counts of active Reddit users, along with Malaysia._")
    st.divider()


# About page code
def about_page():
    st.title("❓About")
    st.markdown('Welcome to the Suicidal Ideation Detector, your companion in identifying and understanding emotional distress conveyed through text.')
    st.divider()
    
    st.markdown('#### 🏠 Main Page')
        # Introduction to the tool
    st.write("""
        This page is designed to classify the content as either **Suicidal** or **Non-Suicidal**.\n
        If the content is identified as **Suicidal**, this tool will attempt to categorize the text into **specific issue categories** if detectable. 
        The issue will be identified using Google Gemini's GenerativeModel ("gemini-pro").
             
        **Please note:** This tool provides insights based on text analysis and is not a substitute for professional medical advice or diagnosis. 
        If you or someone you know is experiencing distress or thoughts of self-harm, please seek immediate help from a mental health professional or contact emergency services.
        """)
    
    st.divider()
    st.markdown('#### 🤖 Chatbot Page')
    st.markdown('''Explore the Chatbot Page to interact with our AI-powered chatbot, driven by Google Gemini's GenerativeModel ("gemini-pro"). 
                Whether you need **information**, **advice**, or just **someone to talk to**, the chatbot is here to assist you! 
                ''')


    st.divider()
    st.markdown('#### 💡 Help & Resources Page')
    # Define the URL and text for the Backlinko link
    backlinko_url = "https://backlinko.com/reddit-users"
    backlinko_text = "Backlinko"
    hotline_url = "https://blog.opencounseling.com/suicide-hotlines/"
    hotline_text = "here"
    st.markdown(f'''Access crucial support information on the Help & Resources Page:<br>
    <p>🗺️ **Interactive Map**: Find suicide hotlines and emergency contact numbers across the top 10 countries with the most active Reddit users, according to [{backlinko_text}]({backlinko_url}), along with Malaysia.<br>
    <p>🛈 **Resource**: Each marker on the map provides the country's name, suicide hotline number, and emergency contact details.<br>
    <p>Hotline for more countries can be referred [{hotline_text}]({hotline_url}).''', unsafe_allow_html=True)


    st.divider()
    st.markdown('#### 👥 Community Support Page')
    st.markdown('''
This page provides a feature that allows authenticated users to **send supportive messages to individuals posting in the SuicideWatch subreddit on Reddit**.

1️⃣ **Login**: Login using Reddit credentials.

2️⃣ **Start Monitoring button**: If clicked, the system automatically detects new posts in the SuicideWatch subreddit. For each post predicted to be suicidal, your account will automatically send a supportive message to the author. The system offers real-time status updates, showing the progress and outcomes of each interaction.

3️⃣ **Stop Monitoring button**: You can stop the monitoring process at any time.

This feature aims to provide immediate emotional support and show that there are people who care and are available to listen.
''')
    


    st.divider()
    st.markdown('#### 📝 Feedback Page')
    st.markdown('''Your feedback is invaluable to us. Visit the Feedback Page to share your insights, suggestions, or issues. Your input helps us improve our services and better support our users.''')
    st.divider()



# Feedback page code
def feedback_page():
    st.title("📝Feedback")
    st.divider()
    # Construct Google Form URL
    google_form_url = "https://forms.gle/f6JwnKZPrvSwHoHx8"
    st.markdown("<p>We value your feedback! Please share any comments, suggestions, or issues you have encountered while using this app. Your input helps us improve our services and better support our users.</p>", unsafe_allow_html=True)
    # Redirect user to Google Form to submit feedback
    st.markdown("Submit your feedback directly here:")
    # Embed the Google Form using iframe
    st.components.v1.iframe(google_form_url, width=940, height=550, scrolling=True)
    st.divider()



    
# Real-time Inteference using Reddit
# Define your Reddit application credentials
CLIENT_ID = os.getenv('client_id')
CLIENT_SECRET = os.getenv('client_secret')
USER_AGENT = os.getenv('user_agent')

def create_reddit_session(username_or_email, password, login_type):
    try:
        if login_type == "Username":
            reddit = praw.Reddit(
                client_id=CLIENT_ID,
                client_secret=CLIENT_SECRET,
                user_agent=USER_AGENT,
                username=username_or_email,
                password=password
            )
        else:
            reddit = praw.Reddit(
                client_id=CLIENT_ID,
                client_secret=CLIENT_SECRET,
                user_agent=USER_AGENT,
                password=password
            )
            reddit.user.me()  # This will trigger an API call that uses the email and password

        # Test authentication
        reddit.user.me()
        return reddit
    except Exception as e:
        st.error(f"Authentication failed during session creation: {str(e)}")
        return None

def send_message_to_user(reddit, user, subject, message):
    try:
        reddit.redditor(user).message(subject, message)
        return f"Message sent to user {user}."
    except praw.exceptions.APIException as e:
        return f"API Exception: {str(e)}"
    except Exception as e:
        return f"Failed to send message to user {user}: {str(e)}"

def predict_suicidal(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    logits = outputs.logits
    predicted_label = torch.argmax(logits, dim=1).item()
    return predicted_label == 1  # Assuming label 1 corresponds to 'suicidal'

def generate_and_send_advice(reddit, user, post_title, post_content):
    full_post_text = f"Title: {post_title}\n\nContent: {post_content}"
    
    # Use the custom predict function to determine if the post is suicidal
    predicted_label, advice = pp2.predict_for_community(full_post_text, bert_model)

    if predicted_label == "suicidal":
        # Send the generated advice to the user
        subject = "I am here to listen"
        result = send_message_to_user(reddit, user.name, subject, advice)
        return result, advice
    else:
        # Return only the non-suicidal prediction
        return "Prediction: Non-suicidal"

def monitor_suicide_watch(reddit, stop_event, status_queue):
    sent_users = set()  # Keep track of users to avoid sending multiple messages
    processed_posts = set()  # Keep track of processed post IDs

    while not stop_event.is_set():
        try:
            for submission in reddit.subreddit('SuicideWatch').new(limit=1):
                if submission.id not in processed_posts:
                    user = submission.author
                    full_post_text = f"Title: {submission.title}\n\nContent: {submission.selftext}"

                    # Display the post being monitored
                    status_queue.put(f"**Monitoring post:** {submission.title}\n {submission.selftext}\n\n---\n")

                    if user and user.name not in sent_users:
                        # Generate advice and send it automatically
                        result, advice = generate_and_send_advice(reddit, user, submission.title, submission.selftext)

                        if result:
                            # Store the advice and user in session state
                            status_queue.put(f"**Prediction: SUICIDAL**\n")
                            status_queue.put(f"Generated and sent advice to {user.name}:\n\n{advice}\n")
                            sent_users.add(user.name)  # Mark the user as having received advice

                        # Mark post as processed
                        processed_posts.add(submission.id)

            if not processed_posts:
                status_queue.put("No new posts found.\n")

        except praw.exceptions.APIException as e:
            status_queue.put(f"API Exception during monitoring: {str(e)}\n")
        except Exception as e:
            status_queue.put(f"An error occurred during monitoring: {str(e)}\n")

        status_queue.put("Searching for new posts...\n")
        stop_event.wait(timeout=5)  

def community_page():
    st.title("Community Support")
    st.divider()

    # Initialize session state variables 
    if 'monitoring_active' not in st.session_state:
        st.session_state['monitoring_active'] = False
    if 'stop_event' not in st.session_state:
        st.session_state['stop_event'] = None
    if 'monitor_thread' not in st.session_state:
        st.session_state['monitor_thread'] = None
    if 'status_queue' not in st.session_state:
        st.session_state['status_queue'] = queue.Queue()
    if 'status_history' not in st.session_state:
        st.session_state['status_history'] = []

    if 'reddit' not in st.session_state:
        st.write("Please log in to start or stop sending supportive messages to new posts in SuicideWatch (a subreddit for suicidal thoughts).")
        with st.form(key='login_form'):
            login_type = st.radio("Login with", ("Username", "Email"))
            username_or_email = st.text_input("Reddit Username or Email")
            password = st.text_input("Reddit Password", type='password')
            submit_button = st.form_submit_button("Login")

            if submit_button:
                reddit = create_reddit_session(username_or_email, password, login_type)
                if reddit:
                    st.session_state['reddit'] = reddit
                    st.success("Authentication successful!")
                    st.session_state['login_successful'] = True
                else:
                    st.error("Authentication failed. Please check your credentials.")
    else:
        st.write("You are already logged in.")
        st.session_state['login_successful'] = True

    # Create a placeholder for status updates
    status_placeholder = st.empty()

    if st.session_state.get('login_successful', False):
        # Start Monitoring Button
        if st.button("Start Monitoring"):
            if not st.session_state['monitoring_active']:
                if st.session_state['monitor_thread'] is None or not st.session_state['monitor_thread'].is_alive():
                    stop_event = threading.Event()
                    status_queue = queue.Queue()
                    monitor_thread = threading.Thread(target=monitor_suicide_watch, args=(st.session_state['reddit'], stop_event, status_queue), daemon=True)
                    monitor_thread.start()
                    st.session_state['stop_event'] = stop_event
                    st.session_state['monitor_thread'] = monitor_thread
                    st.session_state['status_queue'] = status_queue
                    st.session_state['monitoring_active'] = True
                    st.success("Monitoring started.")
            else:
                st.warning("Monitoring is already running.")

        # Stop Monitoring Button
        if st.button("Stop Monitoring"):
            if st.session_state['monitoring_active']:
                if st.session_state['stop_event'] is not None and st.session_state['monitor_thread'] is not None:
                    st.session_state['stop_event'].set()
                    st.session_state['monitor_thread'].join()  
                    st.session_state['monitoring_active'] = False
                    st.session_state['stop_event'] = None
                    st.session_state['monitor_thread'] = None
                    
                    # Clear status history and update the placeholder
                    st.session_state['status_history'] = []
                    status_placeholder.write("Monitoring stopped. Status cleared.")
                    st.success("Monitoring stopped.")
            else:
                st.warning("Monitoring is not active.")

        # Display Monitoring Status
        if st.session_state['monitoring_active']:
            while st.session_state['monitor_thread'] is not None and st.session_state['monitor_thread'].is_alive():
                if not st.session_state['status_queue'].empty():
                    # Get the new status message
                    new_status = st.session_state['status_queue'].get()
                    # Append new status to history
                    st.session_state['status_history'].append(new_status)
                    # Display the entire status history
                    status_placeholder.write("\n".join(st.session_state['status_history']))
                time.sleep(1)
        else:
            status_placeholder.write("Monitoring is not active.")

        

# Define CSS for sidebar navigation buttons
button_css = """
<style>
/* Target only the sidebar buttons */
[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    padding: 10px;
    border: 0.5px solid black;
    border-radius: 5px;
    box-sizing: border-box;
    display: flex;
    justify-content: flex-start; /* Align content to the left */
    text-align: left; /* Ensure text aligns to the left */
}

[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #ddd;
}
</style>
"""
# Inject CSS into the app
st.markdown(button_css, unsafe_allow_html=True)

# Define page links and icons
pages = {
    "Main Page": "🏠",
    "Chatbot": "🤖",
    "Help & Resources": "💡",
    "Community Support": "👥",
    "Feedback": "📝",
    "About": "❓"
}

# Display buttons with icons and links
st.sidebar.title("Navigation")
for page_label, icon in pages.items():
    if st.sidebar.button(f"{icon} {page_label}"):
        st.session_state.page = page_label  # Set the session state for the current page
        st.query_params.from_dict({"page": page_label})
        st.experimental_rerun()  # Rerun the app to navigate to the selected page

# Display the selected page
if st.session_state.page == 'Main Page':
    main_page()
elif st.session_state.page == 'Chatbot':
    chatbot_page()
elif st.session_state.page == 'Help & Resources':
    help_page()
elif st.session_state.page == 'Community Support':
    community_page()
elif st.session_state.page == 'Feedback':
    feedback_page()
elif st.session_state.page == 'About':
    about_page()

