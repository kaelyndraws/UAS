import streamlit as st
import joblib
import numpy as np
import pandas as pd

scaler = joblib.load("preprocessor.pkl")
model = joblib.load("model.pkl")
mbl = joblib.load("mbl_new.pkl")

def make_prediction(features):
    feature_names = joblib.load("features.pkl")
    input_df = pd.DataFrame([features])
    input_df = input_df.reindex(columns=feature_names, fill_value=0)
    x_scaled = scaler.transform(input_df)
    prediction = model.predict(x_scaled)

    return prediction[0]

def main():
    st.title("Machine Learning Credit Score Prediction Model Deployment")

    age = st.number_input("Age", min_value = 18, max_value = 100, value = 33)
    annual_income = st.number_input("Annual Income", min_value = 8000.0, max_value = 436000.0, value = 37000.0)
    monthly_inhand_salary = st.number_input("Monthly Inhand Salary", min_value = 600.0, max_value = 13500.0, value = 3100.0)
    num_bank_accounts = st.number_input("Number of Bank Accounts", min_value = 0, max_value = 20, value = 6)
    num_credit_cards = st.number_input("Number of Credit Cards", min_value = 0, max_value = 20, value = 5)
    interest_rate = st.number_input("Interest Rate", min_value = 0, max_value = 50, value = 13)
    num_of_loan = st.number_input("Number of Loans", min_value = 0, max_value = 9, value = 3)
    delay_from_due_date = st.number_input("Delay from Due Date (days)", min_value = 0, max_value = 61, value = 18)
    num_of_delayed_payments = st.number_input("Number of Delayed Payments", min_value = 0, max_value = 27, value = 14)
    changed_credit_limit = st.number_input("Changed Credit Limit", min_value = -1.0, max_value = 28.0, value = 9.0)
    num_credit_inquiries = st.number_input("Number of Credit Inquiries", min_value = 600, max_value = 13500, value = 3100)
    outstanding_debt = st.number_input("Outstanding Debt", min_value = 30.0, max_value = 4780.0, value = 1170.0)
    credit_utilization_ratio = st.number_input("Credit Utilization Ratio", min_value = 30.0, max_value = 40.0, value = 32.0)
    credit_history_age = st.number_input("Credit History Age (months)", min_value = 20, max_value = 390, value = 220)
    payment_of_min_amount = st.selectbox(
        "Payment of Min Amount", [
            "No",
            "Yes"
        ]
    )
    total_EMI_per_month = st.number_input("Total EMI per Month", min_value = 0.0, max_value = 53299.0, value = 69.0)
    amount_invested_monthly = st.number_input("Amount Invested Monthly", min_value = 20.0, max_value = 950.0, value = 130.0)
    monthly_balance = st.number_input("Monthly Balance", min_value = 70.0, max_value = 1120.0, value = 330.0)
    type_of_loan = st.multiselect(
        "Type of Loan", [
            "Auto Loan",
            "Credit-Builder Loan",
            "Debt Consolidation Loan",
            "Home Equity Loan",
            "Mortgage Loan",
            "Payday Loan",
            "Personal Loan",
            "Student Loan"
        ]
    )
    num_loans = len(type_of_loan)

    if num_loans <= 1:
        suggestion = "Bad"
    elif num_loans <= 3:
        suggestion = "Standard"
    else:
        suggestion = "Good"

    credit_mix = st.selectbox(
        "Credit Mix", [
            "Bad",
            "Standard",
            "Good"
        ], index = ["Bad", "Standard", "Good"].index(suggestion)
    )
    month = st.selectbox(
        "Month", [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August"
        ]
    )
    occupation = st.selectbox(
        "Occupation", [
            "Lawyer",
            "Scientist",
            "Accountant",
            "Engineer",
            "Architect",
            "Mechanic",
            "Writer",
            "Entrepreneur",
            "Media_Manager",
            "Developer",
            "Doctor",
            "Journalist",
            "Teacher",
            "Manager",
            "Musician"
        ]
    )
    credit_mix_encoded = {
        "Bad": 0,
        "Standard": 1,
        "Good": 2
    }[credit_mix]
    payment_of_min_amount_encoded = 1 if payment_of_min_amount == "Yes" else 0
    loan_dict = {
        "Auto Loan": 0,
        "Credit-Builder Loan": 0,
        "Debt Consolidation Loan": 0,
        "Home Equity Loan": 0,
        "Mortgage Loan": 0,
        "Payday Loan": 0,
        "Personal Loan": 0,
        "Student Loan": 0
    }

    for loan in type_of_loan:
        loan_dict[loan] = 1

    num_loan_types = sum(loan_dict.values())
    months = ["January", "February", "March", "April", "May", "June", "July", "August"]
    month_features = {f"Month_{m}":0 for m in months}
    month_features[f"Month_{month}"] = 1
    occupations = ["Lawyer", "Scientist", "Accountant", "Engineer", "Architect", "Mechanic", 
                   "Writer", "Entrepreneur", "Media_Manager", "Developer", "Doctor", "Journalist", 
                   "Teacher", "Manager", "Musician"]
    occupation_features = {f"Occupation_{o}":0 for o in occupations}
    occupation_features[f"Occupation_{occupation}"] = 1
    loan_array = mbl.transform([type_of_loan])
    loan_dummies = pd.DataFrame(loan_array, columns = mbl.classes_)
    num_loan_types_input = loan_dummies.sum(axis = 1).iloc[0]
    available_income = monthly_inhand_salary - total_EMI_per_month - amount_invested_monthly
    spending = available_income - monthly_balance
    spending_ratio = spending / available_income

    if available_income <= 0:
        spending_ratio = 1

    def get_spending_level(ratio):
        if ratio < 0.4:
            return "Low"
        else:
            return "High"

    spending_level_encoded = 1 if get_spending_level(spending_ratio) == "High" else 0
    payment_ratio = spending / monthly_inhand_salary

    def get_payment_size(ratio):
        if ratio < 0.25:
            return "Small"
        elif ratio < 0.50:
            return "Medium"
        else:
            return "Large"
        
    payment_size_encoded = {
        "Small": 0,
        "Medium": 1,
        "Large": 2
    }[get_payment_size(payment_ratio)]
    debt_to_income = outstanding_debt / annual_income
    debt_per_loan = outstanding_debt / num_of_loan
    delay_ratio = num_of_delayed_payments / num_of_loan
    high_utilization = int(credit_utilization_ratio > 0.3)

    def get_age_group(age):
        if 18 <= age < 25:
            return "young"
        elif 25 <= age < 35:
            return "adult"
        elif 35 <= age < 50:
            return "mid"
        else:
            return "senior"
        
    age_group = get_age_group(age)
    age_group_dict = {
        "age_group_young": 0,
        "age_group_adult": 0,
        "age_group_mid": 0,
        "age_group_senior": 0
    }
    age_group_dict[f"age_group_{age_group}"] = 1
    long_credit_history = int(credit_history_age > 120)
    income_delay = annual_income / num_of_delayed_payments
    credit_history_per_age = credit_history_age / age
    total_credit_lines = num_bank_accounts + num_credit_cards
    
    if st.button("Make Prediction"):
        features = [age, annual_income, monthly_inhand_salary, num_bank_accounts, 
                    num_credit_cards, interest_rate, num_of_loan, delay_from_due_date, 
                    num_of_delayed_payments, changed_credit_limit, 
                    num_credit_inquiries, outstanding_debt, credit_utilization_ratio, 
                    credit_history_age, total_EMI_per_month, amount_invested_monthly, 
                    monthly_balance, num_loan_types_input, debt_to_income, 
                    debt_per_loan, delay_ratio, high_utilization, long_credit_history,
                    income_delay, credit_history_per_age, total_credit_lines,
                    payment_of_min_amount_encoded, loan_dict, credit_mix_encoded, 
                    month_features, occupation_features, spending_level_encoded, 
                    payment_size_encoded, age_group_dict]
        result = make_prediction(features)
        label_map = {0: "Poor", 1: "Standard", 2: "Good"}

        st.success(f"Prediction: {result} ({label_map[result]})")

if __name__ == "__main__":
    main()
    
