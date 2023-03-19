#!/bin/bash

echo -e "\n"
echo "You can use this docker image to train the model, then use it for testing or prediction."
echo -e "\n"

echo "To train the model based on the information in train_input_info.cfg run this command:"
echo "docker run -v textile_classification:/root/application -it <image_name> python textile_classification/train.py textile_classification/train_input_info.cfg"
echo -e "\n"

echo "To test the model based on the information in test_input_info.cfg run this command:"
echo "docker run -v textile_classification:/root/application -it <image_name> python textile_classification/test.py textile_classification/test_input_info.cfg"
echo -e "\n"

echo "To do a prediction by the model based on the information in predict_input_info.cfg run this command:"
echo "docker run -v textile_classification:/root/application -it <image_name> python textile_classification/predict.py textile_classification/predict_input_info.cfg"
echo -e "\n"

echo "Note that you can change the .cfg files or any thing inside the container by first running this command:"
echo "docker run -v textile_classification:/root/application -it --entrypoint /bin/bash <image_name>"
echo -e "\n"