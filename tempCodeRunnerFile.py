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