# dnn-neural-structure-analysis
Looking at different network architectures on DNN performance in TensorFlow. I trained a combination of Fully Connected Models on a wide range of designs from 1-10 hidden layers, and 1-512 nodes per layer. 

### The Data
**Reuters** - Dataset of 11,228 newswires from Reuters, labeled over 46 topics.
- https://keras.io/datasets/#reuters-newswire-topics-classification

**IMDB** - Dataset of 25,000 movies reviews from IMDB, labeled by sentiment (positive/negative). 
- https://keras.io/datasets/#imdb-movie-reviews-sentiment-classification

**The Results**
Overall the performance was better across the board than I expected, especially at lower layers. Due to the fact that the inputs were fairly wide (encoded as one-hot for each word in the corpus, ~10,000 on first run) those weights came to contain most of the important information for the model even if the middle layer/s only contained few or one node/s. Especially with the IMDB dataset, where the goal is just a positive/negative response, that information can be compressed down to a single layer and node quite easily which you can see in the image below at the bottom-left of the plot.

Due to Reuters being designed for 46 different multi-class outputs, the model had to work a bit harder to compress the 46 different outputs down to smaller models. This causes there to be a bit more diversity of color/performance across the spectrum, but overall it still performed better than I had assumed initially. 

I had always thought of the power deep learning to be the ability to abstract and find increasingly complex patterns but it appears at least in these two cases that you only need single words/features for the NLP purposes of these models, due to the outcome of fewer layers performing best.

![Reuters_perf](https://github.com/cipher982/dnn-neural-structure-analysis/blob/master/outputs/images/reuters_performance.png?raw=true)

![IMDB_![IMDB_perf](https://github.com/cipher982/dnn-neural-structure-analysis/blob/master/outputs/images/imdb_performance.png?raw=true)
perf](https://github.com/cipher982/dnn-neural-structure-analysis/blob/master/outputs/images/imdb_performance.png?raw=true)
