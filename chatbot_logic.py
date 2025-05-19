import pandas as pd
import difflib

# Q&A data
qa_tourism = pd.read_csv('qa_tourism.csv')
qa_longstay = pd.read_csv('qa_longstay.csv')

# Column names
qa_tourism.columns = ['question', 'answer']
qa_longstay.columns = ['question', 'answer']

def get_best_match(user_input, qa_df):
    questions = qa_df['question'].tolist()
    match = difflib.get_close_matches(user_input.lower(), questions, n=1, cutoff=0.4)
    if match:
        answer_row = qa_df[qa_df['question'] == match[0]]
        return answer_row['answer'].values[0]
    else:
        return "Sorry, I don't understand the question. Please ask again."

def get_response(user_input, bot_type):
    if bot_type == 'tourism':
        return get_best_match(user_input, qa_tourism)
    elif bot_type == 'longstay':
        return get_best_match(user_input, qa_longstay)
    else:
        return "Invalid bot type selected."
