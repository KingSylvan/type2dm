# Step-by-Step Guide to Running the T2DM Risk Prediction App

This guide provides detailed instructions for a user with minimal Python experience to set up and run this Streamlit application on their local machine.

## 1. Prerequisites

Before you begin, ensure you have the following software installed:

- **Python**: This project requires Python (version 3.9 or newer).
  - **Download link**: [https://www.python.org/downloads/](https://www.python.org/downloads/)
  - **Important**: During installation, make sure to check the box that says **"Add Python to PATH"**. This will make it easier to run Python commands from your terminal.

- **A Code Editor (Recommended)**: While not strictly required, using a code editor like Visual Studio Code can make viewing and editing files easier.
  - **Download link**: [https://code.visualstudio.com/](https://code.visualstudio.com/)

---

## 2. Setup and Installation

Follow these steps carefully to get the application running.

### Step 1: Unzip the Project File

- **Action**: Locate the zipped project file (e.g., `type2dm.zip`) that you downloaded.
- **Right-click** on the file and select **"Extract All..."** or **"Unzip"**.
- Choose a memorable location to save the extracted folder (e.g., your Desktop or Documents folder). This will create a new folder named `type2dm`.

### Step 2: Open PowerShell and Navigate to the Project Folder

- **Action**: Open the Windows PowerShell. You can do this by pressing the **Windows key**, typing `PowerShell`, and selecting it from the search results.
- **Navigate to the directory** where you unzipped the project. Use the `cd` (change directory) command.

  ```powershell
  # Example if you saved it on your Desktop
  cd C:\Users\YourUsername\Desktop\type2dm
  ```
  > **Tip**: You can type `cd ` (with a space), then drag the project folder from your File Explorer into the PowerShell window to automatically paste the correct path. Press Enter.

### Step 3: Set PowerShell Execution Policy (Crucial for Windows Users)

By default, Windows may prevent you from running scripts. You need to set an execution policy to allow it for this session.

- **Action**: In your PowerShell window (which should still be in the project folder), run the following command:

  ```powershell
  Set-ExecutionPolicy RemoteSigned -Scope Process
  ```
- This command allows the PowerShell terminal to run local scripts for the current session only. You will need to do this every time you open a new PowerShell window to run the project.

### Step 4: Create a Python Virtual Environment

A virtual environment is a private, isolated space for your project's packages, preventing conflicts with other Python projects.

- **Action**: Run the following command in PowerShell:

  ```powershell
  python -m venv .venv
  ```
- This creates a new folder named `.venv` inside your project directory.

### Step 5: Activate the Virtual Environment

You must "activate" the environment to start using it.

- **Action**: Run this command:

  ```powershell
  .venv\Scripts\Activate.ps1
  ```
- **Confirmation**: You will know it's active because your terminal prompt will change to show `(.venv)` at the beginning, like this:
  ```powershell
  (.venv) PS C:\Users\YourUsername\Desktop\type2dm>
  ```
> **Note**: You must activate the virtual environment every time you open a new terminal to work on the project.

### Step 6: Install Required Packages

Now, install all the Python packages the application needs using the `requirements.txt` file.

- **Action**: With your virtual environment active, run:

  ```powershell
  pip install -r requirements.txt
  ```
- This will take a few minutes as it downloads and installs all the necessary libraries (like `streamlit`, `pandas`, `scikit-learn`, etc.).

---

## 3. Running the Application

Once the setup is complete, you are ready to start the web application.

### Step 7: Start the Streamlit App

- **Action**: In the same PowerShell window (with the virtual environment still active), run the final command:

  ```powershell
  streamlit run app/t2dm_app.py
  ```
- The application will now start. Your web browser should automatically open a new tab to a local address (like `http://localhost:8501`).
- You can now interact with the T2DM Risk Prediction System!

---

## 4. Troubleshooting and Common Errors

If you encounter issues, here are some common problems and their solutions.

- **Error**: `python' or 'pip' is not recognized as the name of a cmdlet...`
  - **Cause**: Python was not added to your system's PATH during installation.
  - **Solution**: Re-install Python, making sure to check the **"Add Python to PATH"** box.

- **Error**: `ModuleNotFoundError: No module named 'streamlit'` (or any other package)
  - **Cause 1**: The required packages were not installed.
  - **Solution 1**: Make sure your virtual environment is active (you see `(.venv)` in the prompt) and run `pip install -r requirements.txt` again.
  - **Cause 2**: Your virtual environment is not active.
  - **Solution 2**: Run `.venv\Scripts\Activate.ps1` to activate it.

- **Error**: `Set-ExecutionPolicy` command fails with an access denied error.
  - **Cause**: You are using a standard PowerShell instead of one with administrative rights.
  - **Solution**: The command `Set-ExecutionPolicy RemoteSigned -Scope Process` should not require administrator rights. Double-check that you have typed it correctly. If it still fails, try opening PowerShell as an Administrator (right-click the icon and "Run as administrator"), but be aware this is less secure.

- **Error**: `FileNotFoundError: [Errno 2] No such file or directory: 'models/t2dm_... .joblib'`
  - **Cause**: The application cannot find the trained model file.
  - **Solution**: Ensure that the `models` folder exists in your project directory and that the `.joblib` file is inside it. The name in `app/t2dm_app.py` must exactly match the name of the file. 