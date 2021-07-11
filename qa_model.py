import logging
from simpletransformers.question_answering import QuestionAnsweringModel, QuestionAnsweringArgs
from context_retriever import query_to_text, summarize_context

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# DistilBERT: a method to pre-train a smaller general-purpose language representation model
# A distilled version of BERT: smaller, faster, cheaper and lighter
model_args = QuestionAnsweringArgs()

model = QuestionAnsweringModel(
    'distilbert', 'distilbert-base-uncased-distilled-squad', use_cuda=False, args=model_args
)


# PREDICT ANSWER - USE LIST CONTEXTS, RETURN (answer, probability)
def predict_answer(question, contexts, seq_len=512, debug=False):
    split_context = []

    if not isinstance(contexts, list):
        contexts = [contexts]

    for context in contexts:
        for i in range(0, len(context), seq_len):
            split_context.append(context[i:i + seq_len])

    split_context = contexts

    f_data = []

    for i, c in enumerate(split_context):
        f_data.append(
            {'qas':
                 [{'question': question,
                   'id': i,
                   'answers': [{'text': ' ', 'answer_start': 0}],
                   'is_impossible': False}],
             'context': c
             })

    answer, probability = model.predict(f_data)
    return answer, probability


# PREDICT ANSWER - USE A SINGLE STRING CONTEXT, RETURN (answer, probability)
def predict_answer_1(question, context):
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


# FUNCTION TO RETURN THE BEST ANSWER ONLY - USE predict_answer() function
def answer_question(question, contexts):
    answer, probability = predict_answer(question, contexts)
    if probability[0]['probability'][0] < 0.8:
        return f"I am not sure... It could be: {answer[0]['answer'][0]} - I am only {probability[0]['probability'][0]*100}% sure about it."
    else:
        return answer[0]['answer'][0]


# FUNCTION TO RETURN THE BEST ANSWER ONLY - USE predict_answer_1() function
def answer_question_1(question, context):
    answer, probability = predict_answer_1(question, context)
    if probability[0]['probability'][0] < 0.8:
        return f"I am not sure... It could be: {answer[0]['answer'][0]} - I am only {probability[0]['probability'][0]*100}% sure about it."
    else:
        return answer[0]['answer'][0]


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


# FUNCTION TO RUN THE PREDICTION AND CHOOSE THE BEST ANSWER GIVEN THE TWO CONTEXT TYPES (WIKIPEDIA AND GOOGLE)
def best_answer (question, contexts_google, context_wikipedia):
    answerG, probabilityG = predict_answer(question, contexts_google)
    answerW, probabilityW = predict_answer_1(question, context_wikipedia)

    best_answer = ''

    print(f"google ans: {answerG[0]['answer'][0]} || wikipedia ans: {answerW[0]['answer'][0]}")

    if probabilityG[0]['probability'][0] > probabilityW[0]['probability'][0]:
        print("contexts from google")
        if probabilityG[0]['probability'][0] > 0.8:
            best_answer = answerG[0]['answer'][0]
        else:
            best_answer = f"I am not sure. It could be: {answerG[0]['answer'][0]}. " \
                          f"I am only {probabilityG[0]['probability'][0]*100}% sure."

    else:
        print("context from wikipedia")
        if probabilityW[0]['probability'][0] > 0.8:
            best_answer = answerW[0]['answer'][0]
        else:
            best_answer = f"I am not sure. It could be: {answerW[0]['answer'][0]}. " \
                          f"I am only {probabilityW[0]['probability'][0]*100}% sure."

    return best_answer


if __name__ == '__main__':
    # context= "The US has passed the peak on new coronavirus cases, " \
    #       "President Donald Trump said and predicted that some states would reopen this month. " \
    #       "The US has over 637,000 confirmed Covid-19 cases and over 30,826 deaths, the highest for any country in the world."
    # questions= ["What was President Donald Trump's prediction?", "How many confirmed Covid-19 cases in the US?"]
    #
    # answers=''
    # for question in questions:
    #     answer = predict_answer_1(question, context)
    #     print(question, answer)

    # NOTE
    # 0 - SUMMARISED
    # 1 - NOT SUMMARISED
    # 2 - NOT SUMMARISED
    question = ['what color is the sky?', 'what is the biggest animal in the world?',
                'what is the smallest planet in our solar system?']  # good for demo

    context1 = "You can prevent sunburn by avoiding exposure to UV radiation. Even when it’s cool or overcast, you can " \
               "still be exposed. The best advice is to stay out of the sun between 9am and 4pm, depending on the time " \
               "of year and where you are in Australia. You can also check the UV levels (called the UV Index), whatever" \
               " the weather. When the 'UV index' is 3 and above, the sun’s rays are strong enough to damage your skin " \
               "and a UV Alert is issued by the Bureau of Meteorology and you should use sun protection. The UV Alert " \
               "is reported on the weather page of all Australian daily newspapers, on the Bureau of Meteorology and " \
               "on some radio and mobile weather forecasts. You can also check the UV Alert for cities and towns across " \
               "Australia with this SunSmart widget, developed by Cancer Council Australia. Select your location and " \
               "find out if sun protection is required. For smartphone users, Cancer Council Australia’s free " \
               "SunSmart app is a great way to check the UV Alert when you are out and about. iPhone users can " \
               "download it at the iTunes App Store, and Android users at Google Play."

    context2 = "You can prevent sunburn by avoiding exposure to UV radiation. Even when it’s cool or overcast, " \
               "you can still be exposed. The best advice is to stay out of the sun between 9am and 4pm, depending " \
               "on the time of year and where you are in Australia."



    question1 = 'how do you prevent sunburn?'

    # context3 = query_to_text(question1, n=3)[0]

    # contexts_google = query_to_text(question[2], n=3)
    # context = summarize_context1(contexts_google[0])
    # summarised
    # answerS, probabilityS = predict_answer_1(question[2], context)
    # not summarised
    # answerNS, probabilityNS = predict_answer_1(question[2], contexts_google[0])
    # answerNS1, probabilityNS1 = predict_answer(question, contexts_google)

    # answer, probability = predict_answer_1(question1, context3)
    # print(answer[0]['answer'][0], probability[0]['probability'][0])

    # formatted_answer = format_answer(answer, probability)
    # print(f"summarised context: {answerS[0]['answer'][0]} {probabilityS[0]['probability'][0]} vs not summarised context: {answerNS[0]['answer'][0]} {probabilityNS[0]['probability'][0]}")
    # print(f"multiple webs: {answerNS1[0]['answer'][0]} {probabilityNS1[0]['probability'][0]} vs first web: {answerNS[0]['answer'][0]} {probabilityNS[0]['probability'][0]}")

    # question = 'what is bitcoin?'
    # contexts_google = query_to_text(question, n=3)
    # context_wikipedia = search_wiki(question)
    # answer = best_answer(question, contexts_google, context_wikipedia)
    # print(f"{question} {answer}")


    q = "how do you prevent sunburn?"
    c = query_to_text(q, n=3)
    answer, probability = predict_answer_1(q, c[0])
    print(answer)
