Project Overview


The idea and goals of the project:

This project explores controllable symbolic music generation using a Transformer-based architecture.
Our goal is to generate expressive piano solos that follow user-specified chord progressions, enabling interactive and harmonically guided music creation.


A brief description of the data you used:

We collected a dataset of pop piano performances and converted them into event-based sequences using the REMI tokenization method.
To support chord control, we extended REMI by inserting two chord tokens per bar — representing the harmony in the first and second halves — which we call ChordREMI.


The structure and organization of the code:

hello_remi.py
This script implements an extended version of REMI tokenization based on the miditok library.
It adds chord tokens to support harmonic conditioning, forming the basis of our ChordREMI method.

infer.py
This script is used for generating piano music. It loads the trained model and performs inference based on user-defined chord inputs.

tokenizer_exp_02.json
This is the vocabulary file that defines the token set used during training and generation, including pitch, duration, bar, position, and chord tokens.

train.ipynb
A Jupyter Notebook used to train the model. It handles data loading, model initialization, training loop, and evaluation.

Model Link: [https://drive.google.com/file/d/1P92lUBfaDp65xHGGl88UNoz9gvmjDddT/view?usp=drive_link](https://drive.google.com/file/d/1P92lUBfaDp65xHGGl88UNoz9gvmjDddT/view?usp=sharing)


A summary of results and key findings:

Our model achieves 95% accuracy in matching generated melodies to given chord progressions.
Listening tests show that users generally prefer our ChordREMI-based outputs over baseline models, indicating stronger harmonic alignment and musicality.


A description of who did what in the group:

Team Contributions
	•	Kaiqi Chen was responsible for data collection and organization, as well as model evaluation.
	•	Linhan Wang handled data preprocessing.
	•	Yuke Tian worked on chord detection.
	•	Juncheng Yu designed the model architecture, conducted training, and implemented MIDI tokenization.


