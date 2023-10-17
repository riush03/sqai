import os
import ast  # for converting embeddings saved as strings back to arrays
import pprint
import google.generativeai as palm
import pandas as pd  # for storing text and embeddings data
import tiktoken  # for counting tokens
from scipy import spatial  # for calculating vector similarities for search
from flask import request

PALM_API_KEY = os.environ.get("PALM_API_KEY")
palm.configure(api_key=PALM_API_KEY)


defaults = {
    'model': 'models/chat-bison-001',
    'temperature': 0.6,
    'candidate_count': 1,
    'top_k': 35,
    'top_p': 0.85,
    }

def ask(
    messages: list[object],
    token_budget: int = 4096 - 500,
    print_message: bool = False,
) -> str:
    """Answers a query using GPT"""
    response =  palm.chat(
        **defaults,
        messages=messages
    )

    return response.last
