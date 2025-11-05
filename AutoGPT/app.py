import streamlit as st
from planner import generate_subtasks

st.set_page_config(page_title="Research Agent Planner MVP", page_icon="🤖")

st.title("🧠 Goal-Oriented Research Agent ")

st.write("""
This AI break your research question into smaller subtasks and excute them autonomously.
The AI decides the number of subtasks automatically based on the question complexity.
""")

# User input
user_query = st.text_area("Enter your research question:", height=120)

# Generate subtasks button
if st.button("Generate Subtasks"):
    if not user_query.strip():
        st.warning("Please enter a research question first.")
    else:
        with st.spinner("Generating subtasks..."):
            try:
                subtasks = generate_subtasks(user_query)
                if subtasks:
                    st.success(f"Generated {len(subtasks)} subtasks ✅")
                    st.write("Here are the subtasks:")
                    for idx, task in enumerate(subtasks, 1):
                        st.write(f"{idx}. {task}")
                else:
                    st.warning("No subtasks were generated. Try rephrasing your question.")
            except Exception as e:
                st.error(f"Error: {e}")
