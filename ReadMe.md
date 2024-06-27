# Activity Monitoring Project

## Jupyter Notebook Instructions

To use the Jupyter Notebook file:

1. Open the file named `ActivityMonitoring_01_08_16.ipynb` in Jupyter Notebook.
2. Change the `data_path`, which is initially set to:

    ```python
    data_path = "D:\\5th Semester\\ML\\ML Project\\bbh\\"
    ```

    Change this path to where you have stored the files, up to `\\bbh\\`, which further contains the `training` and `testing` directories.

## Flask Web App Instructions

To use the web app we have created in Flask for this project:

1. Open the project folder named `MLProjectWebApp` in an IDE or CMD.
2. Make sure you have Python installed.
3. Activate the virtual environment:

    - For Linux/Mac:

        ```bash
        source venv/bin/activate
        ```

    - For Windows:

        ```bash
        venv\Scripts\activate
        ```

4. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

5. Change the `data_path`:

    ```python
    data_path = "D:\\5th Semester\\ML\\ML Project\\bbh\\"
    ```

    Change this path to where you have stored the files, up to `\\bbh\\`, which further contains the `training` and `testing` directories.

6. Run the application:

    ```bash
    python app.py
    ```

7. Access the application:

    Open a web browser and go to `http://localhost:5000` to view our Flask application.
