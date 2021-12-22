### SENDING PROMOTIONAL OFFERS MORE EFFICIENTLY @ STARBUCKS

In order to develop this project, we needed to use some tools, packages, systems and services that could help us achieve our goals.

First of all, we used **Python** to write our scripts not only for algorithm training and serving but also for the orchestration of the whole process. Important packages within this environment are listed below:

* `pandas` so we could work with tabular data in dataframes;
* `numpy` so we could easily manipulate arrays and data structures;
* `seaborn` and `matplotlib` so we could generate insightful visualizations;
* `sklearn` so we could build and develop our model pipeline;
* `imblearn` so we could apply SMOTE to our training data;
* `xgboost` so we could have our main classifier;
* `sagemaker` so we could easily interact with AWS.

Finally, we used AWS environment in order to launch training jobs, deploy our model and serve predictions. The main services used are also listed below:

* __AWS SageMaker__: training, hyperparameter tuning and endpoint serving;
* __Amazon S3__: saving our data and model artifacts;
* __AWS Lambda__: connecting external users to our endpoint so they can make predictions.

____

This project is structured as follows:

#### 00. Proposal

Project proposal documentation.

#### 01. notebook-development

Jupyter notebook containing data preparation, exploration and initial development alternatives.

#### 02. hyperparameter-tuning

Folder to perform hyperparameter tuning which contains:

* `requirements.txt`: environment that needed to be created before launching job;
* `hpo.py`: job script.


#### 03. final-model-training

Folder to perform final model training which contains:

* `requirements.txt`: environment that needed to be created before launching job;
* `model_train.py`: job script.

#### 04. inference-serving

Folder to deploy final model and create an endpoint which contains:

* `requirements.txt`: environment that needed to be created before launching job;
* `serve.py`: methods that allow predictions to take place accordingly.

#### 05. lambda-capstone

Lambda function deployed to send model to production, allowing external users to get prediction requests from our endpoint.

#### 06. Submission Code.ipynb

Jupyter notebook file that orchestrates the end-to-end process in AWS SageMaker and also interacts with other services.