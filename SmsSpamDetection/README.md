#  Spam SMS Detection Web App  

A lightweight, user-friendly web application that detects spam messages with machine learning. Built using Python, Scikit-learn, and Streamlit. 

<div align="center">
  <img src="https://github.com/user-attachments/assets/c16ed5c0-2c85-4ff1-af26-54dd99ec86ae" width="895"/>
</div>


---

##  Features  

1 **Input SMS Messages**  
 - Users can enter any text message for analysis  

2 **Instant Spam Detection**  
 - Classifies messages as "Spam" or "Ham" in real-time  

3 **Machine Learning Model**  
 - TF-IDF vectorization + Logistic Regression (96% accuracy)  
 - Alternative models: Naive Bayes/SVM supported  

4 **Streamlit Interface**  
 - Responsive design works on desktop/mobile
   
5 **Pre-trained Models**  
- Loads `.pkl` model and vectorizer seamlessly  

<div align="center">
  <img src="https://github.com/user-attachments/assets/97b59bf4-1547-48a5-88c4-c48226220dfb" alt="App Screenshot" width="895"/>
</div>

---

##  How It Works  

1. User enters SMS message  
2. Text is preprocessed and vectorized using TF-IDF  
3. Model predicts spam probability  
4. Results displayed with confidence percentage  


 ## Getting Started

To use the SMS Spam Detection web app locally, follow these steps:

1. **Clone the repository** to your local machine:
   ```bash
   git clone https://github.com/bhargavi-misra/CodSoft.git
   

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt

3. **Run the Program**
   ```bash
   streamlit run app.py

4. **Access the app in your browser at:**
   ```bash
   http://localhost:8501

---

## Dataset Used

Source: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

---

## Future Improvements

 - Add more languages (currently English only)

 - Train on larger datasets

 - Add confidence scores or explanation of prediction

 - Dark mode

---

## Contact
If you have any questions or issues, feel free to contact 

- Bhargavi Misra: bhargavimisra@gmail.com
- Project Link: 
