import gradio as gr
from rag_pipeline import qa_bot  # Ensure this import works
import os
from dotenv import load_dotenv
load_dotenv()  # Load HF_TOKEN from .env
# Set Hugging Face token (if not in .env)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HF_TOKEN") # Replace with your token

def respond(question):
    try:
        return qa_bot.run(question)
    except Exception as e:
        return f"Error: {str(e)}"  # Show errors in Gradio

demo = gr.Interface(
    fn=respond,
    inputs=gr.Textbox(label="Ask about CNN/DailyMail articles"),
    outputs=gr.Textbox(label="Answer"),
    examples=["Summarize the article in one sentence.", "What was the key event?"],
    title="RAG Q&A Bot"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861)