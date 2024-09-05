SOME NOTES REGARDING THE PROJECT

to train the model, these will be split up into a bag of words - an array with these words split up
think about it like a plot of points: the x value is dependent on the index in the array of the words, and the y value is the tag associated
training the model will enable it to put together various words, and match them up with the premade patterns in each tag
it will then check the probability of the string inputted being in a certain tag - highest probability determines the tag and thus the response

the process of splitting the string into indices of an array is called tokenization
then, we also use stemming in the process as it finds words with the same root, and reduces them to just the root
examples of this would be working, works, worked - which all have the root 'work.' all these words would then considered to be 'work'
however, stemming needs to be done correctly to ensure that we do not lose the meaning of the word

for the NLP preprocessing, the pipeline will be:
string -> tokenize -> lower + stem -> remove puncuation -> end result (bag of words)
to use this in python, we will use NLTK (natural language toolkit)
