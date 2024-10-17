import streamlit as st
import random
from datetime import datetime
import time

st.title('Hey Beautiful, Need Some Love?')

# Initialize session state for user's name
if 'user_name' not in st.session_state:
    st.session_state.user_name = ''

# Get user's name if not already set
if not st.session_state.user_name:
    user_input = st.text_input("What's your name, gorgeous?")
    if user_input:
        st.session_state.user_name = user_input

# Use the user's name throughout the app
user_name = st.session_state.user_name

# Dictionary of validation messages by category
validations = {
    'Just Because': [
        f"{user_name}, you light up my world every single day.",
        f"I'm so lucky to have you in my life, {user_name}.",
        f"Your smile is my favorite sight, {user_name}.",
        f"I love the way your eyes sparkle when you laugh, {user_name}.",
        f"You're the most beautiful person I know, inside and out, {user_name}."
    ],
    'Tough Day': [
        f"{user_name}, I believe in you. You've got this!",
        f"You're stronger than you know, {user_name}. I'm here for you.",
        f"This too shall pass, {user_name}. I'm proud of how you handle challenges.",
        f"You're doing amazing, {user_name}. Take it one step at a time.",
        f"I admire your resilience, {user_name}. You're incredible."
    ],
    'Accomplishments': [
        f"You're unstoppable, {user_name}! I'm so proud of you.",
        f"Your hard work is paying off, {user_name}. You deserve all the success.",
        f"I'm in awe of your talents, {user_name}. You're amazing!",
        f"You continue to impress me every day, {user_name}.",
        f"The world is lucky to have someone as brilliant as you, {user_name}."
    ],
    'Future Together': [
        f"I can't wait to build our future together, {user_name}.",
        f"Every day with you is an adventure, {user_name}. I'm excited for what's to come.",
        f"You make me want to be a better person, {user_name}.",
        f"I fall in love with you more each day, {user_name}.",
        f"Our love story is my favorite, {user_name}. Here's to many more chapters!"
    ],
    'Reassurance': [
        f"{user_name}, you are enough. You always have been, and you always will be.",
        f"I choose you, {user_name}. Every single day, I choose you.",
        f"Your worth isn't determined by anyone else, {user_name}. You are inherently valuable.",
        f"I love you unconditionally, {user_name}. Nothing can change that.",
        f"You belong here, {user_name}. Your presence makes the world better."
    ],
    'Body Positivity': [
        f"{user_name}, your body is perfect just the way it is.",
        f"I love every inch of you, {user_name}. You're beautiful.",
        f"Your body is strong and capable, {user_name}. It deserves your love.",
        f"You radiate beauty from the inside out, {user_name}.",
        f"I'm in awe of your body and all it does for you, {user_name}."
    ],
    'Anxiety Support': [
        f"{user_name}, your feelings are valid. It's okay to feel anxious sometimes.",
        f"I'm here for you, {user_name}. We'll get through this together.",
        f"You've overcome anxiety before, {user_name}. You can do it again.",
        f"Take a deep breath, {user_name}. You're safe and loved.",
        f"Your anxiety doesn't define you, {user_name}. You are so much more."
    ],
    'Self-Worth': [
        f"{user_name}, your value doesn't decrease based on someone's inability to see your worth.",
        f"You are worthy of love and respect, {user_name}. Never doubt that.",
        f"Your uniqueness is your strength, {user_name}. Embrace it.",
        f"You have so much to offer the world, {user_name}. Don't underestimate yourself.",
        f"I see your worth, {user_name}, and it's immeasurable."
    ]
}

# Category selection
category = st.selectbox('What kind of love do you need today?', list(validations.keys()))

# Button to get validation
if st.button('Give me some love'):
    validation = random.choice(validations[category])
    st.success(validation)
    
    # Add a cute emoji
    emojis = ['😘', '❤️', '😍', '🥰', '💖', '💕', '💓', '💗', '💞']
    st.write(random.choice(emojis))

# Option to add custom validations
st.subheader('Add Your Own Special Message')
custom_category = st.text_input('Enter a category:')
custom_validation = st.text_input('Enter a sweet message:')

if st.button('Add Our Special Message'):
    if custom_category and custom_validation:
        if custom_category not in validations:
            validations[custom_category] = []
        validations[custom_category].append(custom_validation)
        st.success(f'Added "{custom_validation}" to category "{custom_category}"')
    else:
        st.warning('Please enter both a category and a message, love.')

# Display all validations
if st.checkbox('Show all our special messages'):
    for cat, messages in validations.items():
        st.subheader(cat)
        for msg in messages:
            st.write(f"- {msg}")

# Add a daily love note
st.subheader("Your Daily Love Note")
today = datetime.now().strftime("%B %d, %Y")
st.write(f"Date: {today}")
daily_note = f"Hey {user_name}, just a reminder that you're the best thing that's ever happened to me. I love you more than words can express. Have an amazing day, beautiful! 💖"
st.write(daily_note)

# Add a virtual hug button
if st.button('Virtual Hug'):
    st.balloons()
    st.write("Sending you the biggest, warmest hug right now! 🤗💕")

# Add a more interactive virtual kiss button
if st.button('Virtual Kiss'):
    kiss_container = st.empty()
    
    # Simple kiss animation
    for _ in range(3):
        kiss_container.write("💋")
        time.sleep(0.3)
        kiss_container.write("💋💖")
        time.sleep(0.3)
        kiss_container.write("💋💖💕")
        time.sleep(0.3)
        kiss_container.empty()
        time.sleep(0.2)
    
    kiss_container.write(f"Mwah! A sweet kiss just for you, {user_name}! 😘")
    st.balloons()
