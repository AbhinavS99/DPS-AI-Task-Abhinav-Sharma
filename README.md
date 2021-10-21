# DPS-AI-Task-Abhinav-Sharma
---

## Outline
1. [Description of files](#description-of-files)
2. [Steps undertaken](#steps-undertaken)
3. [Link to Video explaining end-to-end ML System](#link-to-end-to-end-video)
4. [Bonus Component](#bonus-component)

---

## Description of files
1. [DPS_Task_Analysis.ipynb](https://github.com/AbhinavS99/DPS-AI-Task-Abhinav-Sharma/blob/main/DPS_Task_Analysis.ipynb) &#8594; The notebook contains data preprocessing used for the final trainer and the two models tried besides of the one given in tutorial. The two models are as follows:
    *  Linear Regression Model &#8594; A simple linear regression model implemented in tensorflow.
    *  Dense Neural Network &#8594; A dense neural network with architecture Dense(64, relu)||Dense(32, relu)||Dense(16, relu)||Dense(4, relu)||Dense(1, relu)
    *  After analysis **Dense Neural Network** performed **better**, thus has been used in the final deployed endpoint.
    *  **Loss Plots and Loss Statistics** are present in the notebook and are therefore **not being separately attached in README**.
  
2. [mpg/Dockerfile](https://github.com/AbhinavS99/DPS-AI-Task-Abhinav-Sharma/blob/main/mpg/Dockerfile) &#8594; Dockerfile, to build the container registery.

3. [mpg/trainer/train.py](https://github.com/AbhinavS99/DPS-AI-Task-Abhinav-Sharma/blob/main/mpg/trainer/train.py) &#8594; The training module.

4. [mpg/Prediction.ipynb]() &#8594; Notebook showing prediction for tutorial test_mpg and task test_mpg.


---

## Steps undertaken
1. Create a google cloud account
2. Enable the required API's Container Registry, Compute, VertexAI etc.
3. Open Notebook in Vertex AI workbench
4. Create a bucket

Bucket Creation|
:------|
![](https://github.com/AbhinavS99/DPS-AI-Task-Abhinav-Sharma/blob/main/images/creating_bucket.png)|

5. Create Dockerfile and the training module
6. Create the Docker Image using ```docker build ./ -t $IMAGE_URI ```
7. Push the Docker Image to google container registry.

Pushing Docker Image|
:------|
![](https://github.com/AbhinavS99/DPS-AI-Task-Abhinav-Sharma/blob/main/images/pushing_docker_image.png)|

8. Locate the Docker Image and train with Vertex AI 

Training|
:----------|
![](https://github.com/AbhinavS99/DPS-AI-Task-Abhinav-Sharma/blob/main/images/training.png)|

9. After the training is complete deploy the model endpoint for making predictions

**Model Endpoint : projects/538537901097/locations/us-central1/endpoints/329334518844489728**

## Link to end to end Video
[Link]()

## Bonus Component
* For Bonus both **custom model** and **explanantion video** have been added.

---

Abhinav Sharma.   
Hoping for a positive response.

