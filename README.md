# keras-language-modeling

Some code for doing language modeling with Keras, in particular for question-answering tasks. It's code was adapted for code retrieval task. The original repository is https://github.com/codekansas/keras-language-modeling

### Stuff that might be of interest

 - `stack_over_flow_qa_eval.py`: Evaluation framework for the Staqc dataset. To get this working, clone the [data repository](https://github.com/mrezende/stack_over_flow_python) and set the `STACK_OVER_FLOW_QA` environment variable to the cloned repository. Changing `config` will adjust how the model is trained.
 - `keras-language-model.py`: The `LanguageModel` class uses the `config` settings to generate a training model and a testing model. The model can be trained by passing a question vector, a ground truth answer vector, and a bad answer vector to `fit`. Then `predict` calculates the similarity between a question and answer. Override the `build` method with whatever language model you want to get a trainable model. Examples are provided at the bottom, including the `EmbeddingModel`, `ConvolutionModel`.

### Getting Started

````bash
# Install Keras (may also need dependencies)
git clone https://github.com/fchollet/keras
cd keras
sudo python setup.py install

# Clone StackOverFlowQA dataset
git clone https://github.com/mrezende/stack_over_flow_python
export STACK_OVER_FLOW_QA=$(pwd)/insurance_qa_python

# Run stack_over_flow_qa_eval.py
git clone https://github.com/mrezende/keras-language-modeling
cd keras-language-modeling/
python stack_over_flow_qa_eval.py
````

### Data
 - [StaQC data](https://github.com/LittleYUYU/StackOverflow-Question-Code-Dataset)

