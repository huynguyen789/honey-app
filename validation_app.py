import streamlit as st
import random
from datetime import datetime
import time
from openai import AsyncOpenAI
import asyncio
import networkx as nx
import matplotlib.pyplot as plt
import json

# Initialize OpenAI client
openai_client = AsyncOpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Function to generate mindmap by section
def generate_mindmap_by_section(notes_by_section):
    G = nx.Graph()
    main_topic = "Main Topic"
    G.add_node(main_topic)

    for section, notes in notes_by_section.items():
        G.add_node(section)
        G.add_edge(main_topic, section)
        for i, note in enumerate(notes):
            node_name = f"{section} Note {i+1}"
            G.add_node(node_name)
            G.add_edge(section, node_name)

    return G

# Function to explain CPA topic
async def explain_cpa_topic(topic):
    """
    Explains a CPA exam topic in a simple way with examples using GPT-4o.
    """
    prompt = f"""
    Explain the CPA exam topic '{topic}' in simple terms. 
    Your explanation should:
    1. Use language suitable for a CPA exam candidate
    2. Include at least two relevant examples or scenarios
    3. Highlight key points that are likely to be tested
    4. Be concise but comprehensive, aiming for about 5-6 sentences of explanation plus the examples
    5. If applicable, mention any recent changes or updates in accounting standards related to this topic
    6. Briefly mention which section of the CPA exam (FAR, AUD, BEC, REG) this topic is most likely to appear in
    """

    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a knowledgeable CPA exam tutor."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {str(e)}"
# Function to save notes to a JSON file
def save_notes_to_file(notes_by_section, filename='notes.json'):
    with open(filename, 'w') as f:
        json.dump(notes_by_section, f)
    st.success("Notes saved successfully!")

# Function to load notes from a JSON file
def load_notes_from_file(filename='notes.json'):
    try:
        with open(filename, 'r') as f:
            notes_by_section = json.load(f)
        st.session_state.notes_by_section = notes_by_section
        st.success("Notes loaded successfully!")
    except FileNotFoundError:
        st.warning("No saved notes found.")


# Initialize session state for current page and notes
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'love_notes'
if 'notes' not in st.session_state:
    st.session_state.notes = []
if 'mindmap' not in st.session_state:
    st.session_state.mindmap = nx.Graph()

# Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ('Love Notes', 'CPA Exam Concepts', 'Notes Organizer'))

if page == 'Love Notes':
    st.session_state.current_page = 'love_notes'
elif page == 'CPA Exam Concepts':
    st.session_state.current_page = 'cpa_exam_concepts'
else:
    st.session_state.current_page = 'notes_organizer'

# Page content
if st.session_state.current_page == 'love_notes':
    st.title('Love Notes')
    st.write("This is the Love Notes page.")
elif st.session_state.current_page == 'cpa_exam_concepts':
    st.title('CPA Exam Concepts Explainer')
    st.write("Need help understanding CPA exam topics? Let me explain them in simple terms with examples!")

    cpa_topic = st.text_input("Enter a CPA exam topic you'd like explained:")
    if st.button("Explain CPA Topic"):
        with st.spinner("Generating explanation..."):
            explanation = asyncio.run(explain_cpa_topic(cpa_topic))
        st.write(explanation)
elif st.session_state.current_page == 'notes_organizer':
    st.title('Notes Organizer and Mindmap Creator')

    # Initialize session state for notes by section
    if 'notes_by_section' not in st.session_state:
        st.session_state.notes_by_section = {}

    # Load notes from file
    if st.button("Load Notes"):
        load_notes_from_file()

    # Add new note with section
    section_name = st.text_input("Enter the section name:")
    new_note = st.text_input("Enter a new note:")
    if st.button("Add Note"):
        if section_name:
            if section_name not in st.session_state.notes_by_section:
                st.session_state.notes_by_section[section_name] = []
            st.session_state.notes_by_section[section_name].append(new_note)
            st.success(f"Note added to section '{section_name}' successfully!")
        else:
            st.error("Please enter a section name.")

    # Save notes to file
    if st.button("Save Notes"):
        save_notes_to_file(st.session_state.notes_by_section)

    # Display existing notes organized by section
    st.subheader("Your Notes by Section")
    for section, notes in st.session_state.notes_by_section.items():
        st.write(f"### {section}")
        for i, note in enumerate(notes):
            st.write(f"{i+1}. {note}")

    # Create mindmap
    if st.button("Generate Mindmap"):
        with st.spinner("Generating mindmap..."):
            mindmap = generate_mindmap_by_section(st.session_state.notes_by_section)
            st.session_state.mindmap = mindmap
        st.success("Mindmap generated!")

    # Display mindmap
    if st.session_state.mindmap.number_of_nodes() > 0:
        st.subheader("Your Mindmap")
        fig, ax = plt.subplots(figsize=(12, 8))
        pos = nx.spring_layout(st.session_state.mindmap)
        nx.draw(st.session_state.mindmap, pos, with_labels=True, node_color='lightblue', 
                node_size=3000, font_size=8, font_weight='bold', ax=ax)
        st.pyplot(fig)

