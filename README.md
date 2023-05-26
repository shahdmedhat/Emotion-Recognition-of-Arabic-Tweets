Emotion detection in Arabic text is a relatively new field of research that focuses on developing methods for identifying and analyzing emotions expressed in Arabic texts. This field combines techniques from natural language processing, machine learning, and psychology to classify emotions into predefined categories such as anger, sadness, joy, fear, or surprise. In this project, we adopt multiple techniques to pre-process and clean an existing labeled dataset, as well as fine-tune a classifier in order to classify emotion in a given tweet into one of the basic emotion categories.

HOW TO RUN: Run the "training.py" file which handles the training of the model and saves it in the "model" folder as "sentiment-analysis-model.pt". To test the code, go to "testing.py" file (which loads the model that was just saved) and call the "predict_sentiment" method with any sentence as a parameter. The output is the predicted label for that sentence. You will find an example at the end of the "testing.py" file.

Note: we also uploaded the link of our model in case you want to test right away. The link is in "model/model_link.txt".

