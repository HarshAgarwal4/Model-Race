import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

from models.testModel import BatchPredictionModel

load_dotenv()

async def generate_response(reviews: list[str] , provider) -> BatchPredictionModel:
    if provider == 'Ollama':
        llm = ChatOllama(model="llama3.2:latest",temperature=0.1, api_key=os.getenv('GROQ_API_KEY'))
    elif provider == 'Groq':
        llm = ChatGroq(model="openai/gpt-oss-120b",temperature=0.1)
    else:
        print('Invalid Model')
        return

    try:
        llm_structured = llm.with_structured_output(BatchPredictionModel)
        response = await llm_structured.ainvoke(
            [
                (
                    "system",
                    """
                    You are a sentiment classification assistant.

                    Classify every given review as either:
                    - positive
                    - negative

                    Return one prediction for every input review.
                    Do not skip or modify any review.
                    """
                ),
                (
                    "human",
                    f"Classify the sentiment of these reviews: {reviews}"
                )
            ]
        )

        return response
    except Exception as e:
        print(f"Sentiment classification failed: {e}")
        raise