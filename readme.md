# Instructions Run RAG and Test Results
## Install dependencies
First, we need to install the python libraries and CUDA.

The python used in this project is 3.10.6. It is recommended to use the same version.

Create conda env:
```bash
conda create -n lcad-rag python=3.10.6
conda activate lcad-rag
```

Install pytorch according to yout OS and CUDA version. Follow the instructions in this link: https://pytorch.org/get-started/locally/

Install the projects dependencies with pip:
```bash
pip install -r requirements.txt
```	

## Create .env file
- Copy sample.env to .env in the root directory
- Add Huggingface token to .env
- Optionally change source texts, data and results directories


## Initialize documents database
Our Rag system uses a document database for retrieval that includes raw chunks, summaries and questions and answers extracted from the original text. To initialize theses files for a new set of data, first place the original documents in .pdf or .txt in the path /texts/_exampledir_. For example, creating a document base about machine learning, we put the pdf files in the directory `/texts/ml`. 

If you want to evaluate your results, put a document file that is going to be used as test in a subdir called /test. For example, `/texts/ml/test`.

The next step is to change the script `init_rag.py`, edit text_dir to the path to the files you just created and set data_dir to the directory you want the data for the rag to be stored. (Optionally change create_test_qa to False if you don't want to create the test set).

Summary:
- Edit your .env customizing TEXT_DIR=texts/ml or any other folder you want.
- Put your .pdf or .txt files in a directory inside this one. For example `texts/ml`
- Put one or more .pdf or .txt files to be used only for test in a subdirectory called test, for example `texts/ml/test` 

Run the script
```
python init_rag.py
```

## Run the RAG System
If you want to run the test suite (and if you enabled create_test_qa in the last step), change the script `chat_main.py`, editing the value of the global variable start_gradio to False and the variables data_dir and results_dir to the name of the directories where the RAG documents are saved and where the results should be stored, respectively.

You can also change which tests are being made, by changing the end of the code.

Summary:
- Edit script `chat_main.py`
    - start_gradio to False
Run the script
```
python chat_main.py
```

It's also possible to run the system in a chat interface, just change the flag start_gradio to True inside `chat_main.py` and run it.

## Run the Evaluation Metrics
Edit the script `evaluate_chatbot_answers.py`, changing the results_dir to the path where your results are.

Then run the script.
```
python evaluate_chatbot_answers.py
```

The evaluation results will be printed on the terminal and the metrics will be saved to the results .csv in your results directory.


# Acknowledgements
Agradecimentos à FAPES, por meio do projeto I2CA - RESOLUÇÃO Nº 285/2021, por fornecer a
bolsa de pesquisa e equipamentos necessários para desenvolvimento deste projeto.


Acknowledgements to FAPES, through the project I2CA - RESOLUTION Nº 285/2021, for providing the
scholarship and equipment necessary for the development of this project.
