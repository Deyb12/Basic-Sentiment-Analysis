# Sentiment Analysis App

A **Streamlit-based sentiment analysis application** that classifies text responses as **Positive or Negative**.
This project uses NLP preprocessing, **TextBlob sentiment scoring**, and visualizations such as **word clouds and charts** to analyze student opinions.

---

## 🚀 Features

* **Dataset Preprocessing**

  * Removes stopwords, special characters, HTML tags, URLs, numbers, and punctuations
  * Lemmatization using SpaCy
  * Cleans alphanumeric and non-ASCII characters

* **Sentiment Classification**

  * Uses **TextBlob polarity & subjectivity scores**
  * Labels responses as **Positive** or **Negative**

* **Visualization**

  * Pie chart and bar chart of sentiment distribution
  * Word clouds for positive and negative sentiment
  * Frequency plots of most common words

* **Export**

  * Download processed dataset as CSV

---

## 🛠️ Tech Stack

* **Frontend/UI:** [Streamlit](https://streamlit.io/)
* **NLP & ML:** NLTK, SpaCy, TextBlob, scikit-learn
* **Visualization:** Matplotlib, Seaborn, WordCloud, Altair
* **Data Handling:** NumPy, Pandas

---

## 📂 Project Structure

```
Basic-Sentiment-Analysis/
├── App.py               # Main Streamlit app
├── data.csv             # Sample dataset (student survey responses)
├── requirements.txt     # Python dependencies
```

---

## ⚙️ Installation & Setup

1. **Clone repository**

   ```bash
   git clone <repo-url>
   cd Basic-Sentiment-Analysis
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run app**

   ```bash
   streamlit run App.py
   ```

   The app will be available at:

   ```
   http://localhost:8501
   ```

---

## 📊 Sample Output

* **Sentiment Distribution:** Pie & bar charts showing ratio of Positive vs. Negative responses.
* **Word Clouds:** Most frequent words for each sentiment category.
* **Top Words:** Bar chart of most common terms in positive/negative responses.

---

## 👨‍💻 Author

* **Built with ❤ by Dave Fagarita**

---

## 📜 License

This project is for **educational purposes only**.
You may modify and use it for learning or experiments.
