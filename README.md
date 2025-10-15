# Legal Clause Risk Analyzer

This project is a Streamlit web application that analyzes legal clauses from contracts to identify potential risks. It classifies each clause into one of three risk categories: High, Medium, or Low. This tool can help users quickly assess the risk level of their legal documents.

## Features

-   **Multiple Input Formats**: Analyze contracts by pasting raw text, or uploading PDF or DOCX files.
-   **Risk Classification**: Each clause is classified as "High", "Medium", or "Low" risk.
-   **User-Friendly Interface**: A simple and intuitive web interface built with Streamlit.
-   **Detailed Analysis**: Provides an overall risk summary and a color-coded, clause-by-clause breakdown.

## Workflow

1.  **Input**: The user provides a contract as raw text or a file (PDF/DOCX).
2.  **Text Extraction**: The application extracts the text from the input.
3.  **Clause Splitting**: The text is split into individual clauses.
4.  **Text Cleaning**: Each clause is cleaned by converting it to lowercase and removing special characters and extra spaces.
5.  **TF-IDF Vectorization**: The cleaned clauses are transformed into numerical vectors using a pre-trained TF-IDF vectorizer.
6.  **Risk Prediction**: A pre-trained stacking classifier predicts the risk level for each clause.
7.  **Output**: The application displays the risk analysis, including an overall summary and a detailed list of clauses with their corresponding risk levels.

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/legal-clause-risk-analyzer.git
    cd legal-clause-risk-analyzer
    ```

2.  **Create a virtual environment**:
    ```bash
    python -m venv clauseenv
    source clauseenv/bin/activate  # On Windows, use `clauseenv\Scripts\activate`
    ```

3.  **Install the dependencies**:
    ```bash
    pip install -r requirement.txt
    ```

## Usage

To run the Streamlit application, use the following command:

```bash
streamlit run application_of_contract_clause_analyser.py
```

This will open the application in your web browser.

## Model Details

The application uses the following pre-trained machine learning models:

-   **`legal_risk_stacking_model.pkl`**: A stacking classifier trained to predict the risk level of legal clauses.
-   **`tfidf_vectorizer.pkl`**: A TF-IDF vectorizer used to convert the text clauses into a numerical format that the model can understand.
-   **`label_encoder.pkl`**: A label encoder used to transform the categorical risk labels ("High", "Medium", "Low") into numerical values for the model.
