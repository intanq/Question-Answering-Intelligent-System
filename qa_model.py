import logging
from simpletransformers.question_answering import QuestionAnsweringModel, QuestionAnsweringArgs


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


# INITIALIZE THE MODEL - distilbert-base-uncased-distilled-squad
model = QuestionAnsweringModel(
    'distilbert', 'distilbert-base-uncased-distilled-squad', use_cuda=False
)

# FUNCTION TO ANSWER THE QUESTION
def predict_answer(question, context):
    to_predict = [
        {
            "context": context,
            "qas": [
                {
                    "question": question,
                    "id": "0",
                }
            ],
        }
    ]
    answer, probability = model.predict(to_predict)
    return answer, probability


# FUNCTION TO FORMAT THE ANSWER
def format_answer(answer, probability):
    the_best_ans_probability = probability[0]['probability'][0]
    the_best_ans = answer[0]['answer'][0]
    if the_best_ans_probability >= 0.8:
        return the_best_ans
    # elif (the_best_ans_probability <= 0.8) and (the_best_ans_probability > 0.3):
    #     return f"I am not sure. Maybe {the_best_ans}. I am only {round(the_best_ans_probability*100)}% sure about it."
    else:
        return f"I am not sure. Maybe {the_best_ans}. I am only {round(the_best_ans_probability*100)}% sure about it."

