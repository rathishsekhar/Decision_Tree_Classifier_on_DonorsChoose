# Decision Tree on Donor's Choose Data

Project Name: Decision Tree on Donor Choose - Applying Decision Tree on Donor'sChoose dataset

# Table of Contents
1. Overview
2. Prerequisites
3. Project Structure
4. Input
5. Model
6. Setup
7. Usage
8. Results
9. Contributing
10. License

# Overview
 In this project, the author implements the sklearn's Decision Tree Classifier model on the Donor's Choose data which was the part of a Kaggle competition held in year 2018.  It involves preprocessing the data, training the model, tuning hyperparameters, and evaluating the model's performance. The Decision Tree Classifier is applied


This project represents the author's endeavour to solve the Donor's choose competition much after the problem was solved and awarded in Kaggle. This served as a practice for the author to use various module from Sci-kit learn, and build essential skills in the field of data sciences. This project is being uploaded much later than when it was solved to the Github for demonstration purpose. The goal as mentioned earlier is enumerated below in two tasks: 

Task 1:
1. Apply Decision Tree Classifier on these feature sets.
2. Conduct the hyper parameter tuning and obtain the best minimum number of samples split
3. Represent the results

Task 2:
1. Select all the features which are having non-zero feature importance, discard all other remaining features and then apply model of choice. 
2. Conduct hyper parameter tuning
3. Summarize the results at the end of the notebook

# Prerequisites

Python 3.6 or later preferably
Jupyter Notebook
Libraries listed in requirements.txt

# Project Structure

The project structure is organized as follows: \
├── notebooks \
│─├── DecisionTreeClassifieronDonorsChoose.ipynb \
├── README.md \
├── requirements.txt \
└── LICENSE \


# Input

This project involves working with the Donor's choose dataset available in Kagle. The dataset comprises of using atleast 50,000 data points and includes various features related to donation requests and their outcomes.


# Model
The model used in teh project is a Multinomial Decision Tree Classifier. 

# Setup

1. Clone this repository to your local machine:
git clone https://github.com/rathishsekhar/DecisionTreeClassifieronDonorsChoose.git

2. Navigate to the project directory:
cd Naive_Bayes_on_Donors_Choose

3. Install the required Python packages:
pip install -r requirements.txt

# Usage

For exploring the tasks, refer to the jupyter notebook - notebooks/DecisionTreeClassifieronDonorsChoose.ipynb

# Ethical Considerations
This project strictly adheres to ethical guidelines and practices:

1. **No User Data Used:** We want to emphasize that no user data has been used in the development or testing of this project. We are committed to safeguarding user privacy and ensuring that any data used is anonymized and ethically sourced.

2. **For Learning Purposes:** Though delayed release in Github, this project is created solely for educational purposes. The primary goal is to attempt Decision Tree on a large dataset like Donor's Choose. We encourage responsible and ethical use of this knowledge in real-world applications.

3. **Transparency:** We are dedicated to transparency in our project's goals and functionality. We encourage open discussion and feedback to ensure that our project aligns with ethical best practices.

We welcome collaboration and contributions from the community to help improve this project's ethical stance and effectiveness.

# Results

This project has achieved the significant results:
1. Hyperparameter tuning results
2. Modle performance metrics such a s accuracy, ROC curve and confusion matrix
3. Top features identified by the model. 


# Maintainance and Updates

This project is a product of continuous learning and improvement. For now, there are no updates required howevever, your feedback, suggestions, and contributions are highly valued. If you have ideas, encounter issues, or want to collaborate, please don't hesitate to reach out. Together, we can continue to improve and refine this project.

# Contributions
Contributions to this project are welcome. Feel free to submit issues and conduct pull requests. 
To contribute:

Fork the repository.
Create a new branch for your feature: git checkout -b feature-name
Make your changes and commit them: git commit -m 'Add feature-name'
Push to your branch: git push origin feature-name
Create a pull request.

# License

This project is licensed under the MIT License. Feel free to use and modify the code as needed.
Feel free to adapt this README template to your specific data science project. It provides a clear structure and information for users and collaborators to understand and contribute to your project effectively