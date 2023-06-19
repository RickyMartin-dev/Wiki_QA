# Building a Question and Answering Application using HuggingFace models
# And the Streamlit library

# Imports
import torch
import wikipedia
import transformers
import streamlit as st
from transformers import pipeline, Pipeline

# Helper Functions
# Loads Summary of Topic From WikiPedia
def load_wiki_summary(query:str) -> str:
    results = wikipedia.search(query)
    summary = wikipedia.summary(results[0], sentences=10)
    return summary

# Load Question and Answering Bert Pipeline
def load_qa_pipeline() -> Pipeline:
    qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
    return qa_pipeline

# Answer the question given the pipeline input
def answer_question(pipeline:Pipeline, question:str, paragraph:str) -> dict:
    input = {
        "question":question,
        "context":paragraph
    }
    output = pipeline(input)
    return output 

# Main app
if __name__ == "__main__":
    # Display title and description
    st.title("Wikipedia Question Answering")
    st.write("Search a topic, Ask a Questions, and Get Answers!!")

    # Display Topic input slot
    topic = st.text_input("SEARCH TOPIC", "")
    
    # Display article paragraph
    article_paragraph = st.empty()

    # Display questino input slot
    question = st.text_input("QUESTON", "")

    if topic:
        # load wikipedia summary of topic
        summary = load_wiki_summary(topic)

        # Display
        article_paragraph.markdown(summary)

        # Perform Question Answering
        if question != "":
            # Load Question Answering Pipeline
            qa_pipeline = load_qa_pipeline()

            # Answer Query Question using article Summary
            result = answer_question(qa_pipeline, question, summary)
            answer = result["answer"]

            # display answer
            st.write(answer)