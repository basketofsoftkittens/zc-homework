Solution for Zencastr Machine Learning Assignment

I added 'pad_same' procedure to 'utils.py' because, given the objective, I believe padding with zeros may not be optimal.
Padding with data from the same file preserves the quality throughout the sample (this is only important when batch size > 1).

To use the new procedure, I added 'use_zeros_to_pad' parameter to the 'data_generator' function.
When 'use_zeros_to_pad=True', the old procedure is used.
When 'use_zeros_to_pad=False', the new procedure ('pad_same') is used, which pads each array with its own data.
