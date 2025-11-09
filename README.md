# Email Spam Detection 📨

## Overview  
This project focuses on detecting whether an email is spam or not using machine learning techniques. The main objective is to classify incoming emails into two categories — **Spam** and **Ham (Not Spam)** — based on the email content.  
The project demonstrates the process of data preprocessing, feature extraction, model building, and evaluation using Python.

---

## Features  
- Cleans and preprocesses raw email text  
- Converts text data into numerical form using TF-IDF vectorization  
- Implements machine learning models such as Logistic Regression for classification  
- Evaluates model performance using metrics like accuracy, precision, recall, and F1 score  
- Includes exploratory data analysis (EDA) for better understanding of the dataset  

---

## Technologies Used  
- **Language:** Python  
- **Libraries:** pandas, numpy, scikit-learn, matplotlib, seaborn, nltk  
- **Environment:** Jupyter Notebook  

---

## Dataset  
The dataset used in this project is **mail_data.csv**, which contains email messages and their corresponding labels (spam or ham).  
Each record includes:  
- **Email text** – The actual content of the email  
- **Label** – Indicates whether the email is spam or not  

---

## Project Structure  
```

Email_Spam_Detection/
│
├── mail_data.csv              # Dataset file
├── detection.ipynb            # Main Jupyter notebook (data preprocessing, training, evaluation)
├── detection-checkpoint.ipynb # Backup notebook file
└── README.md                  # Project documentation

````

---

## Steps to Run the Project  

### 1. Clone the repository  
```bash
git clone https://github.com/Pallabi26313/Email_Spam_Detection.git
cd Email_Spam_Detection
````

### 2. Install dependencies

Make sure you have Python installed, then install required libraries:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn nltk
```

### 3. Run the notebook

```bash
jupyter notebook detection.ipynb
```

### 4. Execute the cells sequentially to:

* Load and explore the dataset
* Preprocess the text data
* Train and test the model
* Evaluate model performance

---

## Results

After training and testing, the model achieved:

* **Accuracy:** *[96%]*

You can further improve accuracy by trying other models such as SVM, Random Forest, or XGBoost.

---

## Future Enhancements

* Add a web app interface using Streamlit or Flask
* Try deep learning models (LSTM or BERT) for better text understanding
* Improve text preprocessing using advanced NLP techniques
* Add visualization dashboards for real-time spam classification

---

## Conclusion

This project successfully demonstrates how machine learning can be applied to classify emails as spam or not spam. It covers end-to-end development — from data preprocessing to model evaluation — making it a useful beginner project for NLP and text classification.

---

## Author

**Pallabi Ghosh**

* GitHub: [Pallabi26313](https://github.com/Pallabi26313)
* Email: *[pallabighosh7142@gmail.com]*
