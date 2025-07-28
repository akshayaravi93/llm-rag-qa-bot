import unittest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.rag_pipeline import qa_bot
import os
from dotenv import load_dotenv
from huggingface_hub import whoami

class TestRAG(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Verify critical components exist
        load_dotenv() 
        assert hasattr(qa_bot, 'run'), "qa_bot missing run() method"
        assert os.getenv("HF_TOKEN"), "Missing HF token"
        print(whoami()) 

    def test_components(self):
        """Test all pipeline components separately"""
        # Test retriever
        docs = qa_bot.retriever.get_relevant_documents("capital of France")
        self.assertGreater(len(docs), 0)
        
        # Test LLM
        llm_response = qa_bot.llm("Capital of France:")
        self.assertIsInstance(llm_response, str)
        
        # Test full pipeline
        response = qa_bot.run("What is the capital of France?")
        self.assertIsNotNone(response)
        self.assertGreater(len(response), 5)  # Minimum response length

    def test_known_query(self):
        response = qa_bot.run("What is the capital of France?")
        print("Full response:", response)  # Debug output
        self.assertIn("Paris", response.capitalize())  # Case-insensitive check

if __name__ == "__main__":
    unittest.main(verbosity=2)