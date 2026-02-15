# Orbis
Model-Run Model Training

An AI training framework where gpt-oss autonomously trains smaller models by analyzing their errors and generating targeted training data in real-time.

## Overview
Model-Run Model Training introduces an innovative approach to AI model training. Using the gpt-oss reasoning model as an intelligent trainer, this framework autonomously improves smaller models by:

Analyzing Weaknesses: Identifies specific errors and gaps in model performance.
Generating Targeted Data: Creates customized training examples to address weaknesses.
Adapting Dynamically: Adjusts training strategies based on model improvement patterns.
Eliminating Manual Effort: Requires no manual curriculum design, enabling fully autonomous learning.

This project was developed for the OpenAI Open Model Hackathon 2025.
# Quick Start
Get up and running in minutes with the following steps:
## Clone the repository
git clone https://github.com/dwan-ith/Orbis.git
cd Orbis

## Install dependencies
pip install -r requirements.txt

## Download pre-trained models (may take some time)
python scripts/download_models.py

## Run a quick demo
python examples/basic_demo.py

## Start the full training loop
python src/main.py


Note: Ensure you have Python 3.8+ installed. Check the Installation Guide for detailed setup instructions.

## Example Results

Before Training

Question: If A > B and B > C, who is smallest?

Target Model: A is smallest (Incorrect)


After gpt-oss Training

Question: If A > B and B > C, who is smallest?

Target Model: C is smallest because in the chain A > B > C, C has the lowest value (Correct with reasoning)

# How It Works
The training pipeline operates in five key phases:

Evaluation: The target model is tested on reasoning tasks to assess performance.

Analysis: gpt-oss analyzes outputs to pinpoint specific failure patterns.

Data Generation: Custom training examples are created to address identified weaknesses.

Training: The target model is fine-tuned using the generated data.

Meta-Learning: The training strategy is evaluated and optimized based on performance trends.

    
# Configuration

Customize the training process by editing src/config/settings.py:

Model Selection: Choose between gpt-oss-20b or gpt-oss-120b.

Training Parameters: Adjust learning rates, batch sizes, and epochs.

Evaluation Metrics: Define success criteria for model performance.

Output Directories: Specify where logs and models are saved.


# Contributing
We welcome contributions! To contribute:

Fork the repository.

Create a feature branch (git checkout -b feature/amazing-feature).

Commit your changes (git commit -m 'Add amazing feature').

Push to the branch (git push origin feature/amazing-feature).

Open a Pull Request.

# Acknowledgments

OpenAI: For providing the gpt-oss models.

Hugging Face: For model hosting and the transformers library.

and also to the Open Source AI Community.
