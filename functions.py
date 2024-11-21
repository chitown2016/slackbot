
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from dotenv import find_dotenv, load_dotenv
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

load_dotenv(find_dotenv())


def draft_email(user_input, name="Dave"):
    chat = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

    template = """
    
    You are a helpful english tutor for a russian student.
    
    For each given english word provide five example english sentences and their russian translations.

    Your reply should be in the following format:

    bullet point Sentence 1 in English
    bullet point Sentence 1 in Russian

    bullet point Sentence 2 in English
    bullet point Sentence 2 in Russian

    bullet point Sentence 3 in English
    bullet point Sentence 3 in Russian

    bullet point Sentence 4 in English
    bullet point Sentence 4 in Russian

    bullet point Sentence 5 in English
    bullet point Sentence 5 in Russian
    
    """

    signature = f"Kind regards, \n\{name}"
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    human_template = "Here's english word I would like you to provide 5 examples for: {user_input}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)
    response = chain.run(user_input=user_input, signature=signature, name=name)

    return response
