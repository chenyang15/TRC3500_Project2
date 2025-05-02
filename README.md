model_training.py takes in all the collected data in terms of all the points from the voltage waveform and inputs it into the model, which then trains it and outputs three classifier models to predict material, height and distance away from sensor.
CLassifier.py takes in real-time data and predicts what kind of material and drop condition it is in real time.
Project2_STM.zip contains the STM project with contains the clock configurations and the main.c code that has an ISR which sends the voltage value everytime the clock is triggered.
