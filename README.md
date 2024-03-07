## Create your own basic NLP algorithm from small dataset with only few lines of data

With the **Semantic Hashing Method**, you can create, train and cutomize your own algorithm.

### Dataset
We need around fifteen/twenty sentences or more for each target, following the format:

``` 
[training phrase][tab][target_names]

example :

I would like to reserve a table please. booking
Book a table for dinner at your restaurant today.	booking
I'm interested in booking a suite for a special occasion.	booking
...

I'd like to provide feedback on my recent dining experience at your restaurant.	remarks
Is there a customer feedback hotline I can call to discuss my recent stay?	remarks
....
``` 

Saved in CSV format.

- *train.csv* : training phrases
- *test.csv* : test phrases

### Requirements
```
pip install numpy sckit-learn
``` 

### Usage

For this application, we want to train a bilingual (english and german) intent classifier
``` 
import pickle
from utils import semhash_training, inference_preprocess

# TRAINING
# import the dataset
filename_train, filename_test = 'data/train.csv', 'data/test.csv'

# the list of the different intents
intent_names = ['booking', 'buchung', 'infoline-eng', 'infoline-deu', 'remarks', 'bemerkungen']

# the place where we want to save our model
model_path = "my_model.pkl"


# INFERENCE
# load the model
my_model = pickle.load(open(model_path, 'rb'))

# example phrase for inference
phrase = 'Ich möchte bitte einen Tisch reservieren.'

# get the predict intent
print(intent_names[my_model.predict(inference_preprocess(phrase))[0]])
``` 

### Demo
I integrated the intent classifier into a simple chatbot, using the botpress framework
running the intent classifier as an [API with flask](https://github.com/mzmpiononz/Simple-NLP-Algo-from-Small-Dataset-with-Semhash-method/blob/main/server.py).
![](https://github.com/mzmpiononz/Simple-NLP-Algo-from-Small-Dataset-with-Semhash-method/blob/main/hotresbot.gif)


### References:
- [know-your-intent](https://github.com/kumar-shridhar/Know-Your-Intent/tree/master)
- [Subword Semantic Hashing for Intent Classification on Small Datasets.](https://arxiv.org/abs/1810.07150)
``` 
@article{shridhar2018subword,
  title={Subword Semantic Hashing for Intent Classification on Small Datasets},
  author={Shridhar, Kumar and Sahu, Amit and Dash, Ayushman and Alonso, Pedro and Pihlgren, Gustav and Pondeknath, Vinay and Simistira, Fotini and Liwicki, Marcus},
  journal={arXiv preprint arXiv:1810.07150},
  year={2018}
}
``` 
