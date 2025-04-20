
# PANI - Precipitation Analysis for Natural Integration

This project focuses on precipitation analysis using Earth Engine and provides tools for natural integration analysis. Follow the steps below to set up the project on your local machine.

## Setup Instructions

Follow these steps to set up the project on your local machine:

### 1. **Clone the repository:**

```bash
git clone https://github.com/lilpotato03/PANI-Precipitaiton-Analysis-for-Natural-Integration.git
cd PANI-Precipitaiton-Analysis-for-Natural-Integration
```

### 2. **Create a virtual environment:**

```bash
python3 -m venv env
```

### 3. **Activate the virtual environment:**

- On macOS and Linux:
  ```bash
  source env/bin/activate
  ```
- On Windows:
  ```bash
  .\env\Scripts\activate
  ```

### 4. **Install the required packages:**

```bash
pip install -r requirements.txt
```

### 5. **Add the current kernel as a Jupyter kernel:**

```bash
python -m ipykernel install --user --name=env --display-name "Python (PANI-Env)"
```

> This will allow you to use the virtual environment in Jupyter notebooks by selecting **Python (PANI-Env)** as the kernel.

---

### 6. **Create a `.env` file with the following content:**

In the root directory of the project, create a file named `.env` and add the following line:

```env
PROJECT_NAME=be-project-aaronfurtado2003
```

> This environment variable is used for initializing Earth Engine with the appropriate project.

---

## Usage

Once the setup is complete, you can start using the project by running the scripts or opening the Jupyter notebooks.

For example, to analyze precipitation data, you can run the provided notebooks or scripts as per your requirements.

## Troubleshooting

If you encounter any issues with authentication or the environment, ensure that:

- The virtual environment is activated.
- The `.env` file is properly created and contains the correct `PROJECT_NAME`.
- Earth Engine authentication is successful.

## Contributing

Feel free to fork the repository and submit pull requests. Make sure to follow the code style and include tests for new features.
