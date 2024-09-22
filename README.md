# MoodSensor

**MoodSensor** is a sentiment analysis project that uses machine learning and NLP to classify text as positive or negative. The project includes a Flask web interface for real-time sentiment analysis.

## Project Structure
```
/MoodSensor            
    ├── templates/
    │   ├── index.html            
    ├── app.py                    
    ├── Train.csv              
    ├── clf.pkl                  
    ├── my_notebook.py
    ├── requirements.txt
    ├── tfidf.pkl
    ├── README.md             
```
## MoodSensor Screenshots

![Screenshot 2024-09-22 191647](https://github.com/user-attachments/assets/8937bdf5-4a83-466f-8524-40a30cc96076)
![Screenshot 2024-09-22 183152](https://github.com/user-attachments/assets/c5085df1-739c-4672-bf46-0675eece1d8c)
![Screenshot 2024-09-22 183217](https://github.com/user-attachments/assets/07213873-9f3d-4c0e-90a8-6ac43e0057e2)

## Setup Instructions
- **`index.html`**: Web interface template.
- **`app.py`**: Flask application script.
- **`clf.pkl`**: Trained model.
- **`tfidf.pkl`**: TF-IDF vectorizer.
- **`Train.csv`**: Training data.
- **`my_notebook.py`**: Data processing and model training notebook.

## Installation

1. Clone the repo and navigate to the project directory:

    ```bash
    git clone https://github.com/yourusername/MoodSensor.git
    cd MoodSensor
    ```

2. Set up a virtual environment and install dependencies:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

## Usage

1. Start the Flask application:

    ```bash
    python app.py
    ```

2. Visit `http://127.0.0.1:5000/` in your browser, enter text, and get sentiment analysis results.
