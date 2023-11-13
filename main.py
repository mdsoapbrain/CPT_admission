from dotenv import load_dotenv
import os
import streamlit as st
from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.chains import ConversationChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
from langchain.chains import SequentialChain
import openai
#
# from langchain.chains import RetrievalQA
# from langchain.document_loaders import PyPDFLoader
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.llms import GPT4All


# /v1/chat/completions: gpt-4, gpt-4-0314, gpt-4-32k, gpt-4-32k-0314, gpt-3.5-turbo, gpt-3.5-turbo-0301
# /v1/completions:    text-davinci-003, text-davinci-002, text-curie-001, text-babbage-001, text-ada-001


def get_chain(llm, admission_template):
    admission_chain = LLMChain(
        llm=llm,
        prompt=admission_template,
        output_key="admission",  # the output from this chain will be called 'meals'
        verbose=True
    )

    plan_chain = LLMChain(
        llm=llm,
        prompt=plan_template,
        # the output from this chain will be called 'gangster_meals'
        output_key="plan_template",
        verbose=True
    )

    overall_chain = SequentialChain(
        chains=[admission_chain, plan_chain],
        input_variables=["draft"],
        output_variables=["admission", "plan_template"],
        verbose=True
    )
    return overall_chain


#
load_dotenv()  # take environment variables from .env.

# API_KEY = os.environ['OPENAI_API_KEY']

st.set_page_config(
    page_title="Admission note Creator",
    page_icon=":shark:",
    layout="wide",
    initial_sidebar_state="auto",
)
with st.sidebar:
    st.title('Build your Second Brain 🧠')
    st.markdown(f'''
    ## About
    This app is an LLM-powered Admission note creator built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
    - [FAISS](https://github.com/facebookresearch/faiss)


    ## Contact
    [Haydn Cheong](https://www.linkedin.com/in/haydnc/)\n
    [Hung Sheng Shih](https://www.linkedin.com/in/danny-hung-sheng-shih-97528a176/)

    ## Feedback
    [Feedback]()

    ''')

    html = f"<a href='https://www.buymeacoffee.com/qmi0000011'><img style='max-width: 100%;' src='https://miro.medium.com/v2/resize:fit:1100/format:webp/1*CEZSIxeYr6PCxsN6Gr38MQ.png'></a>"
    st.markdown(html, unsafe_allow_html=True)
    st.write('Made with 🍣 by Sushi Go')


# meal template (admission)
admission_template = PromptTemplate(
    input_variables=["draft"],
    template="Assuming you are a neurologist, please according to the {draft} to complete the section of present illiness in the admission note in English ()",
)

# gangster template
plan_template = """According to the admission note to right the plan section in the admission note:

admission note:
{admission}
"""

plan_template = PromptTemplate(
    input_variables=['admission'],
    template=plan_template
)

# gpt_model_list = ["gpt-3.5-turbo-16k", "gpt-4-32k", "gpt-4-1106-preview"]
gpt_model_list = ["text-davinci-003", "text-davinci-002",
                  "text-curie-001", "text-babbage-001", "text-ada-001"]

st.title("Admission note & Plan")
API_KEY = st.text_input(
    "Enter your OpenAI API key", value="", type="password")

openai.api_key = os.environ.get('OPENAI_API_KEY')

temp = st.slider("Set the creativity level (Conventionally best: 0.4)",
                 min_value=0.0, max_value=1.5, value=0.4)
model_name = st.selectbox("Select your GPT model: ",
                          options=gpt_model_list, index=0)
user_prompt = st.text_area('病人的PI (草稿)',
                           "Clinical note")


if st.button("產生病歷！") and user_prompt:
    with st.spinner("Generating..."):

        llm = OpenAI(openai_api_key=API_KEY,
                     model_name=model_name, temperature=temp)
        overall_chain = get_chain(llm, admission_template)
        output = overall_chain({'draft': user_prompt})

        col1, col2 = st.columns(2)
        col1.write(output['admission'])
        col2.write(output['plan_template'])
