import streamlit as st
import altair as alt
import nltk
import numpy as np
import pandas as pd
import spacy
import random
from textblob import TextBlob
from nltk.tokenize.toktok import ToktokTokenizer
import re
from nltk.classify import accuracy as nltk_accuracy
from sklearn.utils import shuffle
import string
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from wordcloud import WordCloud
from collections import Counter
import time
from PIL import Image


def app():
    
    from nltk.corpus import stopwords
    from nltk.tokenize import ToktokTokenizer

    nltk.download('stopwords')
    stopwords_list = set(stopwords.words('english')) - {'no', 'not'}
    tokenizer = ToktokTokenizer()
    

    # Set page title
    st.write('<h1 style="font-family: monospace; font-weight:bold; color: #D4AF37;">Sentiment Analysis</h1>', unsafe_allow_html=True)

    # Create two columns
    left_column, right_column = st.columns(2)

    # Add text to the left column
    with left_column:
        st.subheader('Sentiment Analysis')
        st.write('This sentiment analysis will check whether the students of ISAT University Students will agree or disagree that it is more necessary to study grammar than to practice conversation skills.')
        st.write('The respondents in my analysis are from the aforementioned university.\n\nP.S. I would like to express my special thanks to my high school friends, who are currently enrolled at ISAT-U Miagao Campus, for helping me gather and distribute the survey form.')

    # Load and show image in the right column
    with right_column:
        st.markdown(
    f'<div style="display: flex; justify-content: center;"><img src="https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExOGFhNzYxNmQyZTgxNjYyYzc0MTQ2MWY3N2Q5YWM3MTIyY2VlY2ZmNyZlcD12MV9pbnRlcm5hbF9naWZzX2dpZklkJmN0PXM/yQozijZ4xS8krYs1Rp/giphy.gif" style="width:100%; max-width:600px;" /></div>',
    unsafe_allow_html=True
)
    st.write('<p style="font-style:italic; color: #D4AF37;">© Dave Fagarita</p>', unsafe_allow_html=True)
    st.markdown("""<hr style="height:5px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True) 

    with st.echo(code_location='below'):
        def custom_remove_stopwords(text, is_lower_case=False):
            tokens = tokenizer.tokenize(text)
            tokens = [token.strip() for token in tokens]
            if is_lower_case:
                filtered_tokens = [token for token in tokens if token not in stopwords_list]
            else:
                filtered_tokens = [token for token in tokens if token.lower() not in stopwords_list]
            filtered_text = ' '.join(filtered_tokens)
            return filtered_text
    
        def remove_special_characters(text):
            text = re.sub('[^a-zA-z0-9\s]', '', text)
            return text

        def remove_html(text):
            import re
            html_pattern = re.compile('<.*?>')
            return html_pattern.sub(r' ', text)

        def remove_URL(text):
            url = re.compile(r'https?://\S+|www\.\S+')
            return url.sub(r' ', text)

        def remove_numbers(text):
            text =''.join([i for i in text if not i.isdigit()])
            return text

        nlp = []
        if 'en_core_web_sm' in spacy.util.get_installed_models():
            #disable named entity recognizer to reduce memory usage
            nlp = spacy.load('en_core_web_sm', disable=['ner'])
        else:
            from spacy.cli import download
            download("en_core_web_sm")
            nlp = spacy.load('en_core_web_sm', disable=['ner'])
        
        def remove_punctuations(text):
            for punctuation in string.punctuation:
                text = text.replace(punctuation, '')
            return text

        def cleanse(word):
            rx = re.compile(r'\D*\d')
            if rx.match(word):
                return ''      
            return word

        def remove_alphanumeric(strings):
            nstrings= [" ".join(filter(None, (cleanse(word) for word in string.split()))) \
                       for string in strings.split()]
            str1=' '.join(nstrings)
            return str1
        
        def lemmatize_text(text):
            text = nlp(text)
            text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
            return text
        
        if st.button('Load Dataset'):  
            df = pd.read_csv('sentimentcs.csv')

            #remember this very useful function to randomly rearrange the dataset
            train = shuffle(df)
            
            
            st.write('Preview of the Dataset:')
            st.dataframe(train, use_container_width=True)
   
            st.write('Dataset shape: ', df.shape)

            # Define preprocessing functions here...

            
            with st.spinner('Checking for null values...'):
                st.write('Checking for any null values in the dataset. Will not proceed if a null value is found.')
                st.write(train.isnull().sum())
                st.success('No null values found.')

            with st.spinner('Preprocessing the data...'):
                st.text('Doing pre-processing techniques...')

                with st.spinner('Removing symbols...'):
                    train.replace(r'^\s*$', np.nan, regex=True, inplace=True)
                    train.dropna(axis=0, how='any', inplace=True)
                    st.success('Symbols removed successfully.')

                with st.spinner('Removing escape sequences...'):
                    train.replace(to_replace=[r"\\t|\\n|\\r", "\t|\n|\r"], value=["",""], regex=True, inplace=True)
                    st.success('Escape sequences removed successfully.')

                with st.spinner('Removing non ascii data...'):
                    train['text'] = train['text'].str.encode('ascii', 'ignore').str.decode('ascii')
                    st.success('Non-ascii data removed successfully.')

                with st.spinner('Removing punctuations...'):
                    train['text'] = train['text'].apply(remove_punctuations)
                    st.success('Punctuations removed successfully.')

                with st.spinner('Removing stop words...'):
                    train['text'] = train['text'].apply(custom_remove_stopwords)
                    st.success('Stop words removed successfully.')

                with st.spinner('Removing special characters...'):
                    train['text'] = train['text'].apply(remove_special_characters)
                    st.success('Special characters removed successfully.')

                with st.spinner('Removing HTML...'):
                    train['text'] = train['text'].apply(remove_html)
                    st.success('HTML removed successfully.')

                with st.spinner('Removing URL...'):
                    train['text'] = train['text'].apply(remove_URL)
                    st.success('URL removed successfully.')

                with st.spinner('Removing numbers...'):
                    train['text'] = train['text'].apply(remove_numbers)
                    st.success('Numbers removed successfully.')

                st.text('Initial Pre-processing steps')
                st.write(train.head(50))

                with st.spinner('Removing alpha numeric data...'):
                    train['text'] = train['text'].apply(remove_alphanumeric)
                    st.success('Alpha-numeric data removed successfully.')

                st.text('Final Pre-processing steps')
                st.write(train.head(50))

                with st.spinner('Lemmatizing words...'):
                    train['text'] = train['text'].apply(lemmatize_text)
                    st.success('Words lemmatized successfully.')

            st.success('Data preprocessing completed successfully!')

            #We use the TextBlob tweet sentiment function to get the sentiment
            train['sentiment']=train['text'].apply(lambda tweet: TextBlob(tweet).sentiment)
            
            st.write('Preprocessed Dataset:')
            st.dataframe(train, use_container_width=True)

            sentiment_series=train['sentiment'].tolist()
            columns = ['polarity', 'subjectivity']
            df1 = pd.DataFrame(sentiment_series, columns=columns, index=train.index)
            result = pd.concat([train, df1], axis=1)
            result.drop(['sentiment'],axis=1, inplace=True)

            result.loc[result['polarity']>=0.1, 'Sentiment'] = "Positive"
            result.loc[result['polarity']<0.1, 'Sentiment'] = "Negative"

            result.loc[result['label']=="1", 'Sentiment_label'] = 1
            result.loc[result['label']=="0", 'Sentiment_label'] = 0
            result.drop(['label'],axis=1, inplace=True)
            
            st.write('Preprocessed data sentiment labels updated.')
            result = result.sort_values(by=['Sentiment'], ascending=False)
            st.dataframe(result, use_container_width=True)

            counts = result['Sentiment'].value_counts()
            st.write(counts)
            st.subheader('Sentiment Graphs')
            #reads the sample count from the previous line
            labels = ['Negative','Positive']
            sizes = [counts[0], counts[1]]
            custom_colours = ['#F2543D', '#38C477']

            fig = plt.figure(figsize=(8, 3), dpi=100)
            plt.subplot(1, 2, 1)
            plt.pie(sizes, labels = labels, textprops={'fontsize': 10}, startangle=140, \
                    autopct='%1.0f%%', colors=custom_colours, explode=[0, 0.05])
            plt.subplot(1, 2, 2)
            sns.barplot(x = labels, y = sizes, \
                    palette = 'viridis')
            st.pyplot(fig)
            
            st.subheader('Negative Sentiment')
            
            st.write('Word cloud of the negative sentiment')
            
            text = " ".join(result[result['Sentiment'] == 'Negative']['text'])
            fig = plt.figure(figsize = (8, 4))
            wordcloud = WordCloud(max_words=500, height= 800, width = 1500,  \
                                  background_color="black", colormap= 'viridis').generate(text)
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(fig)
            
            all_nodep_words = []
            for sentence in result[result['Sentiment'] == 'Negative']['text'].to_list():
                for word in sentence.split():
                   all_nodep_words.append(word)

            df = pd.DataFrame(Counter(all_nodep_words).most_common(25), columns= ['Word', 'Frequency'])

            sns.set_context('notebook', font_scale= 1)
            fig = plt.figure(figsize=(8,4))
            sns.barplot(y = df['Word'], x= df['Frequency'], palette= 'summer')
            plt.title("Negative Sentiment Most Commonly Used Words")
            plt.xlabel("Frequency")
            plt.ylabel("Words")
            st.pyplot(fig)
            
            
            st.subheader('Positive Sentiment')
            
            st.write('Word cloud of the positive sentiment')
            
            text = " ".join(result[result['Sentiment'] == 'Positive']['text'])
            fig = plt.figure(figsize = (8, 4))
            wordcloud = WordCloud(max_words=500, height= 800, width = 1500,  \
                                  background_color="black", colormap= 'viridis').generate(text)
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(fig)
            
            all_nodep_words = []
            for sentence in result[result['Sentiment'] == 'Positive']['text'].to_list():
                for word in sentence.split():
                   all_nodep_words.append(word)

            df = pd.DataFrame(Counter(all_nodep_words).most_common(25), columns= ['Word', 'Frequency'])

            sns.set_context('notebook', font_scale= 1)
            fig = plt.figure(figsize=(8,4))
            sns.barplot(y = df['Word'], x= df['Frequency'], palette= 'summer')
            plt.title("Positive Sentiment Most Commonly Used Words")
            plt.xlabel("Frequency")
            plt.ylabel("Words")
            st.pyplot(fig)
                                            
            # Save the dataframe to a CSV file
            csv = result.to_csv(index=False)
            if csv:
                b64 = base64.b64encode(csv.encode()).decode()  # Convert to base64
                href = f'<a href="data:file/csv;base64,{b64}" download="data.csv">Download CSV file</a>'
                st.markdown(href, unsafe_allow_html=True)
            else:
                st.error('Error: Unable to generate CSV file.', icon="🚨")
                
# run the app
if __name__ == "__main__":
    app()
