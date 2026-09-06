## 🚀 Live Streamlit App

Try the Customer Churn AI Retention Advisor here:

[Open Streamlit App](https://exit-exam-executive-program-customer-churn-ai-advisor-qi8gaw3s.streamlit.app/)


Customer Churn Risk & AI Retention Advisor

Project Overview

This project develops a telecom customer churn prediction and AI-powered
retention advisory system using the IBM Telco Customer Churn dataset. It
combines exploratory analysis, machine learning, customer segmentation,
retrieval-based retention rules, grounded Gemini explanations, and a
Streamlit interface.

Dataset

7,043 customers

21 original columns

Target: Churn (Yes/No)

Key EDA Findings

Overall churn rate: 26.54%

Month-to-month churn: 42.71%

One-year churn: 11.27%

Two-year churn: 2.83%

Fiber optic churn: 41.89%

DSL churn: 18.96%

No internet service churn: 7.40%

Average tenure, churned: 17.98 months

Average tenure, not churned: 37.57 months

Tenure/churn correlation: -0.3522

Data Preparation

TotalCharges contained 11 blank values. These records had zero tenure,
so the blanks were treated as zero accumulated charges and converted to
numeric values. customerID was excluded from model features.

Categorical variables were processed with OneHotEncoder, numerical
variables with StandardScaler, and the data was split 80/20 using
stratification.

Model Performance

Logistic Regression

Accuracy: 80.55%

Precision: 65.72%

Recall: 55.88%

F1: 60.40%

ROC-AUC: 84.21%

Random Forest

Test Accuracy: 78.57%

Test Precision: 62.50%

Test Recall: 48.13%

Test F1: 54.38%

Test ROC-AUC: 82.01%

Training Accuracy: 99.80%

Final model: Logistic Regression. It generalized better and achieved
higher recall and ROC-AUC, while Random Forest showed substantial
overfitting.

Customer Segmentation

K-Means clustering used tenure, monthly charges, and predicted churn
probability.

The recommended first retention target is Segment 1 --- New,
High-Spend, High-Risk Customers: - 462 customers - Average tenure:
14.80 months - Average monthly charges: 80.08 - Average churn
probability: 50%

This segment combines relatively high churn risk with meaningful
recurring revenue at stake.

AI Retention Playbook

Clause 1 --- High Risk (probability ≥ 0.70): Offer a loyalty
discount and a callback from a retention specialist within 48 hours.

Clause 2 --- Moderate Risk (0.40--0.70): Send a targeted email
highlighting an underused service or a contract upgrade offer.

Clause 3 --- New Customer, Any Risk, Tenure < 3 months: Route to
the onboarding team instead of the standard retention flow.

Clause 4 --- Non-Discrimination Rule: Retention explanations must
never state or imply that gender, senior-citizen status, or
family/partner status contributed to a customer's risk score, even where
a statistical correlation exists in the data.

Retrieval is re-run for every customer analysis request.

Responsible AI

The explanation layer excludes: - Gender - SeniorCitizen - Partner -
Dependents

These fields are filtered before customer-level contributors are
supplied to the LLM. The LLM receives the churn probability, assigned
risk tier, tenure, permitted contributing features, retrieved playbook
clause, and non-discrimination instruction.

Retrieval Experiment

With retrieval, the LLM followed the supplied retention clause. Without
retrieval, it invented a clause and retention actions not contained in
the approved playbook. This demonstrates the importance of grounding for
business decision support.

Streamlit Application

The Streamlit interface allows a retention agent to enter customer
details and receive: - Churn probability - Risk tier - Retrieved
retention clause - Top permitted churn-risk contributors - Grounded AI
retention explanation - Responsible-AI safeguard information

Project Files

Customer_Churn_Risk_AI_Retention_Advisor.ipynb
app.py
Customer_Churn_AI_Retention_Implementation_Note.docx
README.md

The Streamlit application also requires the trained churn_model.pkl
and preprocessor.pkl files if it is run independently.

Technologies

Python, pandas, NumPy, scikit-learn, Matplotlib, Joblib, Streamlit,
Google Gemini API, Google Colab.

Run the App

Install the required dependencies, make the trained model/preprocessor
files available, securely set the Gemini API key as an environment
variable, and run:

streamlit run app.py

Security: Never commit the Gemini API key or other secret files to
GitHub.

Conclusion

The system combines churn prediction, customer segmentation,
customer-level model explanations, rule-based retrieval, and grounded
generative AI. Logistic Regression was selected as the final predictive
model, and the Streamlit application translates predictions into
retention guidance while keeping protected demographic attributes out of
the LLM explanation layer.
