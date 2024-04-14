from fastapi import FastAPI, Form, Request, Response, File, Depends, HTTPException, status
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.encoders import jsonable_encoder
from langchain_openai import ChatOpenAI
from langchain.text_splitter import TokenTextSplitter
from langchain_community.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import RetrievalQA

import pandas as pd
import os 
import json
import uvicorn
import aiofiles
import csv
import ocrmypdf
from api_keys.Constants import OPENAI_API
from api_keys.Constants import OPENAI_API

app = FastAPI()

API_KEY = str(OPENAI_API())

API_KEY = str(OPENAI_API())

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

os.environ["OPENAI_API_KEY"] = API_KEY

#Nhận vào đường dẫn đến file pdf gốc và đường dẫn file được xử lý
def ocr_pdf(file_path, save_path):
    ocrmypdf.ocr(file_path, save_path, skip_text=True)
    return save_path #Trả về đường dần file được xử lý

def file_processing(file_path):
    #Load file PDF
    loader = PyPDFLoader(file_path)
    data = loader.load()
    question_gen = ''

    #Tách văn bản thành các chunks
    for page in data:
        question_gen += page.page_content
        
    splitter_ques_gen = TokenTextSplitter(
        model_name = 'gpt-3.5-turbo',
        chunk_size = 10000,
        chunk_overlap = 200
    )

    chunks_ques_gen = splitter_ques_gen.split_text(question_gen)

    document_ques_gen = [Document(page_content=t) for t in chunks_ques_gen]

    splitter_ans_gen = TokenTextSplitter(
        model_name = 'gpt-3.5-turbo',
        chunk_size = 1000,
        chunk_overlap = 100
    )

    document_answer_gen = splitter_ans_gen.split_documents(
        document_ques_gen
    )

    return document_ques_gen, document_answer_gen

def llm_pipeline(file_path):

    document_ques_gen, document_answer_gen = file_processing(file_path)

    llm_ques_gen_pipeline = ChatOpenAI(
        temperature = 0.3,
        model = "gpt-3.5-turbo"
    )

    prompt_template = """
    You are an expert at creating questions based on studying materials and documentation.
    Your goal is to prepare a student for their exam and tests.
    You do this by creating questions about the text below:

    ------------
    {text}
    ------------

    Create questions in English that will prepare the student for their tests.
    Make sure not to lose any important information.

    QUESTIONS:
    """

    PROMPT_QUESTIONS = PromptTemplate(template=prompt_template, input_variables=["text"])

    refine_template = ("""
    You are an expert at creating practice questions based on studying material and documentation.
    Your goal is to help a student prepare for their test.
    We have received some practice questions to a certain extent: {existing_answer}.
    We have the option to refine the existing questions or add new ones.
    (only if necessary) with some more context below.
    ------------
    {text}
    ------------

    Given the new context, refine the original questions in English.
    If the context is not helpful, please provide the original questions.
    QUESTIONS:
    """
    )

    REFINE_PROMPT_QUESTIONS = PromptTemplate(
        input_variables=["existing_answer", "text"],
        template=refine_template,
    )

    ques_gen_chain = load_summarize_chain(llm = llm_ques_gen_pipeline, 
                                            chain_type = "refine", 
                                            verbose = True, 
                                            question_prompt=PROMPT_QUESTIONS, 
                                            refine_prompt=REFINE_PROMPT_QUESTIONS)

    ques = ques_gen_chain.run(document_ques_gen)

    embeddings = OpenAIEmbeddings()

    vector_store = FAISS.from_documents(document_answer_gen, embeddings)

    llm_answer_gen = ChatOpenAI(temperature=0.1, model="gpt-3.5-turbo")

    ques_list = ques.split("\n")
    filtered_ques_list = [element for element in ques_list if element.endswith('?') or element.endswith('.')]

    answer_generation_chain = RetrievalQA.from_chain_type(
                                                            llm=llm_answer_gen, 
                                                            chain_type="stuff", 
                                                            retriever=vector_store.as_retriever()
                                                        )

    return answer_generation_chain, filtered_ques_list

def get_csv (file_path):
    answer_generation_chain, ques_list = llm_pipeline(file_path)
    base_folder = 'static/output/'
    if not os.path.isdir(base_folder):
        os.mkdir(base_folder)
    output_file = base_folder+"QA.csv"    
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Question", "Answer"])

        for question in ques_list:
            print("Question: ", question)
            answer = answer_generation_chain.run(question)
            print("Answer: ", answer)
            print("--------------------------------------------------\n\n")

            # Save answer to CSV file
            csv_writer.writerow([question, answer])
    return output_file

def write_gen():
    output_file = 'static/output/QA.csv'
    with open(output_file, "r") as f:
        csv_reader = csv.DictReader(f)
        question_bank = []
        for row in csv_reader:
            question_bank.append(row)
    for question in question_bank:
        print(question)

def answer_check(question, input_text, answer_text):
    model = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.3)
    template = """
    You are an expert at grading student answers. I was given this question: "{question}"
    Your goal is to grade my answer by comparing it to the actual answer using this grading scale:
        From 0 - 6: Incorrect. You should give out this grading if the my answer is entirely different to the actual answer; has major mistakes (like containing 1 or more points with opposite/contradictive meaning to the actual answer; missing keywords/key points, etc)
        From 7 - 10: Correct. You should give out this grading if the my answer satisfied all the keypoints of the questions and the actual answer. The phrasing may be different but the answer should be 80-percent-or-above similar to the actual answer. The actual answer and my answer may not have the same examples. You will IGNORE ALL THE EXAMPLES OF THE ACTUAL ANSWER and only focus on the my examples to see if their examples stick to the main points of the question.
    Remember not to provide the grading and only say Incorrect or Correct.
    
    Here is my answer: "{input_text}"

    Here is the correct answer: "{answer_text}"

    Keep your response short by only grading the answer and pointing out the main differences. Do not give out compliments, keep it strictly analytical.
    My answer may not be 100 percent similar to the actual answer. I may use synonyms and rephrasal. Make sure to capture the key details of the actual answer and the my answer to provide the most accurate rating.
    The actual answer and my answer may not have the same examples. You will IGNORE ALL THE EXAMPLES OF THE ACTUAL ANSWER and only focus on the my examples to see if my examples stick to the main points of the question.
    Make sure not to lose any important information.
    """

    PROMPT = ChatPromptTemplate.from_template(template)

    chain = PROMPT | model

    result = chain.invoke({"question": question,
                  "input_text": input_text,
                  "answer_text": answer_text})
    return result.content

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def chat(request: Request, pdf_file: bytes = File(), filename: str = Form(...)):
    base_folder = 'static/docs/'
    if not os.path.isdir(base_folder):
        os.mkdir(base_folder)
    pdf_filename = os.path.join(base_folder, filename)

    work_folder = 'static/temp'

    temp_pdf = "temp.pdf"

    if not os.path.isdir(work_folder):
        os.mkdir(work_folder)
    temp_filename = os.path.join(work_folder, temp_pdf)

    ocr_pdf(pdf_filename, temp_filename)

    async with aiofiles.open(pdf_filename, 'wb') as f:
        await f.write(pdf_file)

    response_data = jsonable_encoder(json.dumps({"msg": 'success',"pdf_filename": temp_filename}))
    res = Response(response_data)
    return res

question_index = 0
question_bank = None

@app.post("/analyze")
async def chat(request: Request, pdf_filename: str = Form(...)):
    global question_bank
    output_file = get_csv(pdf_filename)
    question_bank = pd.read_csv(output_file)
    response_data = jsonable_encoder(json.dumps({"output_file": output_file}))
    res = Response(response_data)
    return res

@app.post("/quiz")
def quiz(request: Request):
    global question_index

    question = question_bank['Question'][question_index]
    result = ''
    answer = ''
    next_question = False

    return templates.TemplateResponse("quiz.html", {"request": request, "question": question, "answer": answer, "result": result, "next_question": next_question})

@app.post("/check_answer")
async def check_answer(request: Request):
    global question_index

    form_data = await request.form()
    user_answer = form_data['answer']
    correct_answer = question_bank['Answer'][question_index]
    question = question_bank['Question'][question_index]
    next_question = True

    result = answer_check(question, user_answer, correct_answer)

    return templates.TemplateResponse("quiz.html", {"request": request, "question": question, "user_answer": user_answer, "answer": correct_answer, "result": result, "next_question": next_question})

@app.post("/next_question")
async def next_question(request: Request):
    global question_index
    question_index += 1

    # Check if we have reached the end of the question bank
    if question_index >= len(question_bank):
        question = 'End of Question'
        answer = ''
        next_question = False
    else: 
        question = question_bank['Question'][question_index]
        answer = ''
        next_question = False
    return templates.TemplateResponse("quiz.html", {"request": request, "question": question, "answer": answer, "result": '', "next_question": next_question})

if __name__ == "__main__":
    uvicorn.run("app:app", port=8000, reload=True)
    
