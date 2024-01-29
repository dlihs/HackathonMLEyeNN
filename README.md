Submission for the Hackathon Project for the course Machine Learning in 2023/2024 at the Heinrich Heine University in Düsseldorf.
Participants: qas13riy delih100n amyad100 dowin102 paspe101

preproccesing takes an image path and transforms it into a compatible imput for the model. The preprocessing technique which we used was centered around using the canny edge detection algorithm since we estimated that the blood vessels within the eye will be the best indicator of the age.

Model takes an image path and runs the age prediction and gives back an age. The model is a convolutional neural network which trains for 100 epochs and performs regression for age. 

