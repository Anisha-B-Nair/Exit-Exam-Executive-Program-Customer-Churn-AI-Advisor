
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from google import genai


# ============================================================
# LOAD MODEL
# ============================================================

baseline_model = joblib.load("churn_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Customer Churn AI Retention Advisor",
    page_icon="📡",
    layout="wide"
)


# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>

/* Main page background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(
        135deg,
        #f5f8ff 0%,
        #edf3ff 55%,
        #fafcff 100%
    );
}

/* Main content area */
.block-container {
    max-width: 1400px;
    padding-top: 1.4rem;
    padding-bottom: 3rem;
}

/* Force readable text */
[data-testid="stAppViewContainer"] p {
    color: #26364f !important;
}

[data-testid="stAppViewContainer"] label {
    color: #26364f !important;
}

[data-testid="stAppViewContainer"] h1,
[data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3 {
    color: #173b7a !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #f5f8ff;
    border-right: 1px solid #dce5f6;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label {
    color: #203451 !important;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #173b7a !important;
}


/* Hero banner */
.hero {
    background: linear-gradient(
        120deg,
        #2758df,
        #493ed9,
        #7629e6
    );

    padding: 30px 32px;
    border-radius: 22px;
    margin-bottom: 30px;

    box-shadow:
        0px 12px 32px rgba(48, 68, 160, 0.18);
}

.hero-title {
    color: white !important;
    font-size: 36px;
    font-weight: 800;
    margin-bottom: 10px;
}

.hero-subtitle {
    color: #eef2ff !important;
    font-size: 17px;
}


/* Metrics */
[data-testid="stMetric"] {
    background: white;
    border: 1px solid #e0e7f5;
    border-radius: 18px;
    padding: 20px;

    box-shadow:
        0px 5px 20px rgba(31, 50, 95, 0.07);
}

[data-testid="stMetricLabel"] p {
    color: #52627c !important;
}

[data-testid="stMetricValue"] {
    color: #173b7a !important;
}


/* Containers */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255,255,255,0.94);
    border-radius: 18px;
    border-color: #e1e7f4 !important;

    box-shadow:
        0px 5px 18px rgba(32, 52, 95, 0.06);
}


/* Button */
.stButton > button {
    width: 100%;

    background: linear-gradient(
        90deg,
        #2866ef,
        #5844e8,
        #8236eb
    );

    color: white !important;

    border: none;
    border-radius: 12px;

    padding: 0.75rem 1rem;

    font-weight: 700;
}

.stButton > button:hover {
    color: white !important;
    border: none !important;
}


/* Progress bar */
[data-testid="stProgress"] > div > div > div {
    background: linear-gradient(
        90deg,
        #2563eb,
        #7c3aed
    );
}


/* Divider */
hr {
    border-color: #dce4f3;
}


/* Hide Streamlit footer */
footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    "<div class='hero'>"
    "<div class='hero-title'>📡 Customer Churn Risk & AI Retention Advisor</div>"
    "<div class='hero-subtitle'>Machine-learning churn prediction with grounded AI-powered retention guidance.</div>"
    "</div>",
    unsafe_allow_html=True
)


# ============================================================
# RETENTION PLAYBOOK
# ============================================================

retention_playbook = {

    "Clause 1":
    "High Risk (probability ≥ 0.70): Offer a loyalty discount and a callback from a retention specialist within 48 hours.",

    "Clause 2":
    "Moderate Risk (0.40–0.70): Send a targeted email highlighting an underused service or a contract upgrade offer.",

    "Clause 3":
    "New Customer, Any Risk, Tenure < 3 months: Route to the onboarding team instead of the standard retention flow.",

    "Clause 4":
    "Non-Discrimination Rule: Retention explanations must never state or imply that gender, senior-citizen status, or family/partner status contributed to a customer's risk score, even where a statistical correlation exists in the data."
}


# ============================================================
# SIDEBAR INPUTS
# ============================================================

with st.sidebar:

    st.header("👤 Customer Details")

    st.caption(
        "Enter customer service and billing information."
    )

    st.subheader("Demographics")

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1]
    )

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )


    st.divider()

    st.subheader("Account")

    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=100,
        value=7,
        step=1
    )

    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        max_value=1000.0,
        value=80.0,
        step=1.0,
        format="%.2f"
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        max_value=100000.0,
        value=560.0,
        step=10.0,
        format="%.2f"
    )


    st.divider()

    st.subheader("Services")

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        [
            "No",
            "Yes",
            "No phone service"
        ]
    )

    internet_service = st.selectbox(
        "Internet Service",
        [
            "DSL",
            "Fiber optic",
            "No"
        ]
    )

    online_security = st.selectbox(
        "Online Security",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )

    online_backup = st.selectbox(
        "Online Backup",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )

    device_protection = st.selectbox(
        "Device Protection",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )

    tech_support = st.selectbox(
        "Tech Support",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )


    st.divider()

    analyse_button = st.button(
        "🔍 Analyse Customer",
        use_container_width=True
    )


# ============================================================
# DEFAULT SCREEN
# ============================================================

if not analyse_button:

    st.header("How the Advisor Works")

    col1, col2, col3 = st.columns(3)


    with col1:

        with st.container(border=True):

            st.subheader("📊 1. Predict")

            st.write(
                "The Logistic Regression model estimates "
                "the customer's probability of churn."
            )


    with col2:

        with st.container(border=True):

            st.subheader("🎯 2. Retrieve")

            st.write(
                "The system retrieves the appropriate "
                "retention playbook clause."
            )


    with col3:

        with st.container(border=True):

            st.subheader("🤖 3. Explain")

            st.write(
                "Gemini generates a grounded explanation "
                "using permitted risk drivers."
            )


    st.info(
        "👈 Enter customer details in the sidebar "
        "and click Analyse Customer."
    )


# ============================================================
# ANALYSE CUSTOMER
# ============================================================

if analyse_button:


    # ========================================================
    # CUSTOMER DATA
    # ========================================================

    customer = pd.DataFrame([{

        "gender": gender,

        "SeniorCitizen":
        senior_citizen,

        "Partner":
        partner,

        "Dependents":
        dependents,

        "tenure":
        tenure,

        "PhoneService":
        phone_service,

        "MultipleLines":
        multiple_lines,

        "InternetService":
        internet_service,

        "OnlineSecurity":
        online_security,

        "OnlineBackup":
        online_backup,

        "DeviceProtection":
        device_protection,

        "TechSupport":
        tech_support,

        "StreamingTV":
        streaming_tv,

        "StreamingMovies":
        streaming_movies,

        "Contract":
        contract,

        "PaperlessBilling":
        paperless_billing,

        "PaymentMethod":
        payment_method,

        "MonthlyCharges":
        monthly_charges,

        "TotalCharges":
        total_charges

    }])


    # ========================================================
    # MODEL PREDICTION
    # ========================================================

    customer_processed = (
        preprocessor.transform(
            customer
        )
    )


    classes = list(
        baseline_model.classes_
    )


    if "Yes" in classes:

        yes_index = (
            classes.index("Yes")
        )

    elif 1 in classes:

        yes_index = (
            classes.index(1)
        )

    else:

        yes_index = 1


    risk_probability = (
        baseline_model
        .predict_proba(
            customer_processed
        )[0, yes_index]
    )


    # ========================================================
    # RETRIEVAL
    # RUNS ON EVERY REQUEST
    # ========================================================

    if tenure < 3:

        risk_tier = (
            "New Customer"
        )

        clause_name = (
            "Clause 3"
        )

        retrieved_clause = (
            retention_playbook[
                "Clause 3"
            ]
        )


    elif risk_probability >= 0.70:

        risk_tier = (
            "High Risk"
        )

        clause_name = (
            "Clause 1"
        )

        retrieved_clause = (
            retention_playbook[
                "Clause 1"
            ]
        )


    elif risk_probability >= 0.40:

        risk_tier = (
            "Moderate Risk"
        )

        clause_name = (
            "Clause 2"
        )

        retrieved_clause = (
            retention_playbook[
                "Clause 2"
            ]
        )


    else:

        risk_tier = (
            "Low Risk"
        )

        clause_name = (
            "No matching retention clause"
        )

        retrieved_clause = (
            "No specific retention action "
            "is required under the "
            "supplied playbook."
        )


    # ========================================================
    # FEATURE CONTRIBUTIONS
    # ========================================================

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )


    if hasattr(
        customer_processed,
        "toarray"
    ):

        customer_values = (
            customer_processed
            .toarray()[0]
        )

    else:

        customer_values = (
            np.asarray(
                customer_processed
            ).ravel()
        )


    contributions = (
        customer_values
        *
        baseline_model.coef_[0]
    )


    contribution_df = (
        pd.DataFrame({

            "Feature":
            feature_names,

            "Contribution":
            contributions

        })
    )


    # ========================================================
    # EXCLUDE PROTECTED DEMOGRAPHICS
    # ========================================================

    protected_terms = [

        "gender",

        "SeniorCitizen",

        "Partner",

        "Dependents"

    ]


    protected_pattern = (
        "|".join(
            protected_terms
        )
    )


    allowed_contributions = (
        contribution_df[

            ~contribution_df[
                "Feature"
            ].str.contains(

                protected_pattern,

                case=False,

                regex=True

            )

        ].copy()
    )


    # ========================================================
    # TOP 3 RISK CONTRIBUTORS
    # ========================================================

    top_3 = (

        allowed_contributions[

            allowed_contributions[
                "Contribution"
            ] > 0

        ]

        .sort_values(

            "Contribution",

            ascending=False

        )

        .head(3)

    )


    # ========================================================
    # CLEAN FEATURE NAMES
    # ========================================================

    cleaned_features = []


    for feature in (
        top_3["Feature"]
        .tolist()
    ):

        cleaned = (
            feature.replace(
                "categorical__",
                ""
            )
        )

        cleaned = (
            cleaned.replace(
                "numerical__",
                ""
            )
        )

        cleaned = (
            cleaned.replace(
                "_",
                " "
            )
        )

        cleaned_features.append(
            cleaned
        )


    # ========================================================
    # CUSTOMER RISK ASSESSMENT
    # ========================================================

    st.header(
        "📊 Customer Risk Assessment"
    )


    metric1, metric2, metric3 = (
        st.columns(3)
    )


    with metric1:

        st.metric(
            "Churn Probability",
            f"{risk_probability:.2%}"
        )


    with metric2:

        st.metric(
            "Customer Tenure",
            f"{tenure} months"
        )


    with metric3:

        st.metric(
            "Monthly Charges",
            f"{monthly_charges:.2f}"
        )


    st.write("")

    st.subheader(
        "🚦 Risk Classification"
    )


    if risk_tier == "High Risk":

        st.error(
            f"🔴 HIGH RISK — "
            f"{risk_probability:.2%} "
            f"predicted churn probability"
        )


    elif risk_tier == "Moderate Risk":

        st.warning(
            f"🟠 MODERATE RISK — "
            f"{risk_probability:.2%} "
            f"predicted churn probability"
        )


    elif risk_tier == "New Customer":

        st.info(
            f"🔵 NEW CUSTOMER — "
            f"{risk_probability:.2%} "
            f"predicted churn probability. "
            f"Onboarding flow takes priority."
        )


    else:

        st.success(
            f"🟢 LOW RISK — "
            f"{risk_probability:.2%} "
            f"predicted churn probability"
        )


    # ========================================================
    # RISK PROGRESS BAR
    # ========================================================

    st.subheader(
        "Predicted Churn Risk"
    )


    progress_value = int(
        round(
            risk_probability * 100
        )
    )


    progress_value = max(
        0,
        min(
            100,
            progress_value
        )
    )


    st.progress(
        progress_value
    )


    st.caption(
        f"Model probability: "
        f"{risk_probability:.2%}"
    )


    # ========================================================
    # RETENTION PLAYBOOK + DRIVERS
    # ========================================================

    left_col, right_col = (
        st.columns(2)
    )


    with left_col:

        with st.container(
            border=True
        ):

            st.subheader(
                "🎯 Retention Playbook"
            )

            st.write(
                f"**{clause_name}**"
            )

            st.info(
                retrieved_clause
            )


    with right_col:

        with st.container(
            border=True
        ):

            st.subheader(
                "📌 Key Churn-Risk Drivers"
            )


            if len(
                cleaned_features
            ) > 0:

                for i, feature in enumerate(
                    cleaned_features,
                    start=1
                ):

                    st.write(
                        f"**{i}. {feature}**"
                    )


            else:

                st.write(
                    "No positive "
                    "non-demographic "
                    "risk contributors "
                    "were identified."
                )


    # ========================================================
    # BUILD LLM FEATURE TEXT
    # ========================================================

    if len(
        cleaned_features
    ) > 0:

        feature_text = (
            "\n".join(
                [
                    f"{i + 1}. {feature}"

                    for i, feature
                    in enumerate(
                        cleaned_features
                    )
                ]
            )
        )


    else:

        feature_text = (
            "No positive permitted "
            "contributing features "
            "available."
        )


    compliance_clause = (
        retention_playbook[
            "Clause 4"
        ]
    )


    # ========================================================
    # SYSTEM PROMPT
    # ========================================================

    system_prompt = """

You are an AI retention advisor assisting a telecom retention agent.

Write a clear 3-4 sentence explanation for the retention agent.

Base your explanation ONLY on:
- the supplied churn probability,
- assigned risk tier,
- tenure,
- permitted contributing features,
- and the retrieved retention playbook clause.

Use the supplied risk tier exactly as given.

Do not independently create or change the customer's risk tier.

Do not state or imply that gender, SeniorCitizen status,
Partner status, Dependents status, family status,
or any other protected demographic attribute contributed
to the customer's churn risk.

Do not invent customer information.

Do not invent a retention action.

Only recommend an action that appears in the retrieved
retention playbook clause.

"""


    # ========================================================
    # USER PROMPT
    # ========================================================

    user_prompt = f"""

Customer churn probability:
{risk_probability:.2%}

Assigned risk tier:
{risk_tier}

Customer tenure:
{tenure} months

Permitted top contributing features:
{feature_text}

Retrieved retention clause:
{retrieved_clause}

Compliance rule:
{compliance_clause}

Write the retention-agent explanation.

Use the assigned risk tier exactly as provided.

Do not create a different risk classification.

Do not recommend any action that is not contained
in the retrieved retention clause.

"""


    # ========================================================
    # AI RETENTION ADVISOR
    # ========================================================

    st.header(
        "🤖 AI Retention Advisor"
    )


    try:

        api_key = st.secrets.get(
        "GEMINI_API_KEY",
        os.getenv("GEMINI_API_KEY")
    )


        if not api_key:

            st.warning(
                "Gemini API key "
                "is not available. "
                "Prediction and retrieval "
                "still completed successfully."
            )


        else:

            client = (
                genai.Client(
                    api_key=api_key
                )
            )


            full_prompt = (
                system_prompt
                +
                "\n\n"
                +
                user_prompt
            )


            with st.spinner(
                "Generating grounded "
                "retention explanation..."
            ):

                response = (
                    client.models
                    .generate_content(

                        model=
                        "gemini-flash-lite-latest",

                        contents=
                        full_prompt

                    )
                )


            with st.container(
                border=True
            ):

                st.subheader(
                    "💡 AI Recommendation"
                )

                st.write(
                    response.text
                )


    except Exception as e:

        st.warning(
            "The churn prediction "
            "and retrieval worked, "
            "but the AI explanation "
            "could not be generated."
        )

        st.error(
            str(e)
        )


    # ========================================================
    # RESPONSIBLE AI
    # ========================================================

    st.header(
        "🛡️ Responsible AI Safeguard"
    )


    st.info(
        "Gender, SeniorCitizen, Partner and Dependents "
        "are excluded from the LLM explanation layer. "
        "The recommendation is grounded in the retrieved "
        "retention playbook clause."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Customer Churn Risk & AI Retention Advisor • "
    "Machine Learning + Grounded GenAI Decision Support"
)
