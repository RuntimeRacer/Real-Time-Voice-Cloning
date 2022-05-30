
## Model parameters
model_hidden_size = 768
model_embedding_size = 768
model_num_layers = 3


## Training parameters
#learning_rate_init = 1e-4 => Worked well until 3.6M steps
#learning_rate_init = 5e-5 => Worked for about 100k more steps until plateauing
#learning_rate_init = 1e-5 => Worked for another 170k more steps until plateauing
learning_rate_init = 1e-6
#speakers_per_batch = 128 => Worked until 4M steps
speakers_per_batch = 64
utterances_per_speaker = 10
