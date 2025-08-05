import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
from pathlib import Path
from PIL import Image
import streamlit as st
import imagify
from bokeh.plotting import figure
import math
from bokeh.palettes import Greens
from itertools import cycle
import string
import re
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load models and data
model = load_model('botmodel.h5')
tok = joblib.load('tokenizer_t.pkl')
words = joblib.load('words.pkl')
df2 = pd.read_csv('bot.csv')
flag = 1

def pie_chart_bokeh(labels, data, title):
    if not labels or not data or sum(data) == 0:
        p = figure(height=400, width=400, title=title, toolbar_location=None, tools="")
        p.text(x=0, y=1, text=["No Data"], text_align="center", text_font_size="16pt")
        return p
    angle = [d / sum(data) * 2 * math.pi for d in data]
    palette_size = max(3, min(len(labels), 9))
    color_palette = Greens[palette_size]
    color = list(cycle(color_palette))
    p = figure(height=400, width=400, title=title, toolbar_location=None, tools="")
    start_angle = 0
    for i in range(len(labels)):
        end_angle = start_angle + angle[i]
        p.wedge(
            x=0, y=1, radius=0.4,
            start_angle=start_angle, end_angle=end_angle, color=color[i],
            legend_label=f"{labels[i]} - {round(data[i])}%"
        )
        start_angle = end_angle
    p.axis.visible = False
    p.grid.visible = False
    p.legend.location = "right"
    return p

def line_chart_bokeh(x, y_list, labels, title):
    p = figure(title=title, x_range=x, height=400, width=600)
    colors = ["Purple", "Blue", "Green", "Magenta", "Red"]
    for i, y in enumerate(y_list):
        p.line(x, y, line_width=2, line_color=colors[i % len(colors)], legend_label=labels[i])
    p.legend.location = "top_left"
    return p

def main():
    lem = WordNetLemmatizer()
    if "flag" not in st.session_state:
        st.session_state.flag = 1

    def tokenizer(x):
        tokens = x.split()
        rep = re.compile('[%s]' % re.escape(string.punctuation))
        tokens = [rep.sub('', i) for i in tokens]
        tokens = [i for i in tokens if i.isalpha()]
        tokens = [lem.lemmatize(i.lower()) for i in tokens]
        tokens = [i.lower() for i in tokens if len(i) > 1]
        return tokens

    def no_stop_inp(tokenizer, df, c):
        no_stop = []
        x = df[c][0]
        tokens = tokenizer(x)
        no_stop.append(' '.join(tokens))
        df[c] = no_stop
        return df

    def inpenc(tok, df, c):
        t = tok
        x = [df[c][0]]
        enc = t.texts_to_sequences(x)
        padded = pad_sequences(enc, maxlen=16, padding='post')
        return padded

    def predinp(model, x):
        pred = np.argmax(model.predict(x))
        return pred

    def botp(df3, pred):
        l = df3.user[0].split()
        if len([i for i in l if i in words]) == 0:
            pred = 1
        return pred

    def botop(df2, pred):
        x2 = df2.groupby('labels').get_group(pred).shape[0]
        idx1 = np.random.randint(0, x2)
        op = list(df2.groupby('labels').get_group(pred).bot)
        return op[idx1]

    def botans(df3):
        tok = joblib.load('tokenizer_t.pkl')
        df3 = no_stop_inp(tokenizer, df3, 'user')
        inp = inpenc(tok, df3, 'user')
        pred = predinp(model, inp)
        pred = botp(df3, pred)
        ans = botop(df2, pred)
        return ans

    def get_text():
        x = st.text_input("You : ")
        x = x.lower()
        xx = x[:13]
        if(xx == "start my test"):
            st.session_state.flag = 0
        input_text = [x]
        df_input = pd.DataFrame(input_text, columns=['user'])
        return df_input

    qvals = {"Select an Option": 0, "Strongly Agree": 5, "Agree": 4, "Neutral": 3, "Disagree": 2, "Strongly Disagree": 1}

    st.title("CounselBot")
    banner = Image.open("img/21.png")
    st.image(banner, use_column_width=True)
    st.write("Hi! I'm CounselBot, your personal career counseling bot. Ask your queries in the text box below and hit enter. If and when you are ready to take our personality test, type \"start my test\".")

    df3 = get_text()
    if (df3.loc[0, 'user'] == ""):
        ans = "Hi, I'm CounselBot. \nHow can I help you?"
    elif (st.session_state.flag == 0):
        ans = "Sure, good luck!"
    else:
        ans = botans(df3)

    st.text_area("CounselBot:", value=ans, height=100, max_chars=None)

    # --- PERSONALITY TEST ---
    if st.session_state.flag == 0:
        st.title("PERSONALITY TEST:")
        kr = st.selectbox("Would you like to begin with the test?", ["Select an Option", "Yes", "No"])
        if (kr == "Yes"):
            kr1 = st.selectbox("Select level of education", ["Select an Option", "Grade 10", "Grade 12", "Undergraduate"])

            # --- GRADE 10 ---
            if kr1 == "Grade 10":
                questions = [
                    "I find writing programs for computer applications interesting",
                    "I can understand mathematical problems with ease",
                    "Learning about the existence of individual chemical components is interesting",
                    "The way plants and animals thrive gets me curious",
                    "Studying about the way fundamental constituents of the universe interact with each other is fascinating",
                    "Accounting and business management is my cup of tea",
                    "I would like to know more about human behaviour, relations and patterns of thinking",
                    "I find the need to be aware of stories from the past.",
                    "I see myself as a sportsperson/professional trainer",
                    "I enjoy creating works of art"
                ]
                if "g10_answers" not in st.session_state:
                    st.session_state.g10_answers = [None] * 10

                for idx, q in enumerate(questions):
                    st.header(f"Question {idx + 1}")
                    st.write(q)
                    n = imagify.imageify(idx+1)
                    answer = st.selectbox(
                        "",
                        ["Select an Option", "Strongly Agree", "Agree", "Neutral", "Disagree", "Strongly Disagree"],
                        key=f"g10_{idx+1}"
                    )
                    # Only update answer if user selects
                    if answer != "Select an Option":
                        st.session_state.g10_answers[idx] = qvals[answer]
                    else:
                        st.session_state.g10_answers[idx] = None

                # Only proceed if all questions answered
                if all(x is not None for x in st.session_state.g10_answers):
                    st.success("Test Completed")
                    st.title("RESULTS:")
                    try:
                        df = pd.read_csv(r"Subjects.csv")
                    except Exception as e:
                        st.error(f"Could not load Subjects.csv: {e}")
                        return

                    input_list = st.session_state.g10_answers

                    subjects = {
                        1: "Computers",
                        2: "Mathematics",
                        3: "Chemistry",
                        4: "Biology",
                        5: "Physics",
                        6: "Commerce",
                        7: "Psychology",
                        8: "History",
                        9: "Physical Education",
                        10: "Design"
                    }

                    def output(listofanswers):
                        class my_dictionary(dict):
                            def __init__(self):
                                self = dict()
                            def add(self, key, value):
                                self[key] = value
                        ques = my_dictionary()
                        for i in range(0, 10):
                            ques.add(i, listofanswers[i])
                        all_scores = [ques[i] / 5 for i in range(10)]
                        li = [[all_scores[i], i] for i in range(len(all_scores))]
                        li.sort(reverse=True)
                        sort_index = [x[1] + 1 for x in li]
                        all_scores.sort(reverse=True)
                        a = sort_index[0:5]
                        b = all_scores[0:5]
                        s = sum(b)
                        d = list(map(lambda x: x * (100 / s) if s != 0 else 0, b))
                        return a, d

                    l, data = output(input_list)
                    out_labels = [subjects[n] for n in l]
                    st.bokeh_chart(pie_chart_bokeh(out_labels, data, "Recommended subjects"), use_container_width=True)

                    st.header('More information on the subjects')
                    for i in range(5):
                        st.subheader(out_labels[i])
                        st.write(df['about'][int(l[i]) - 1])

                    st.header('Choice of Degrees')
                    for i in range(5):
                        st.subheader(out_labels[i])
                        st.write(df['further career'][int(l[i]) - 1])

                    st.header('Trends over the years')
                    x = ['2000', '2005', '2010', '2015', '2020']
                    def Convert(string):
                        li = list(string.split(","))
                        li = list(map(float, li))
                        return li
                    y = [Convert(df['trends'][int(l[i]) - 1]) for i in range(5)]
                    st.bokeh_chart(line_chart_bokeh(x, y, out_labels, "Trends"), use_container_width=True)
                    banner1 = Image.open("img/coun.png")
                    st.image(banner1, use_column_width=True)
                    st.header("Contacts of experts from various fields")
                    for i in range(5):
                        st.subheader(out_labels[i])
                        xl = (df['contacts'][int(l[i]) - 1]).split(",")
                        for kk in xl:
                            st.write(kk)
                else:
                    st.info("Please answer all questions above to see your results.")

if __name__ == "__main__":
    main()
