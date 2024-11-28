# Title Slide
Welcome everyone. Today I'll be presenting my work on Sign Language Recognition Using Neural Networks, specifically focusing on a two-layer implementation for static gesture recognition.

# Overview
In this presentation, I'll cover the problem statement and motivation, analyze the dataset, discuss the neural network architecture, examine the results, and explore possible future directions for this work.

# Problem Statement & Motivation
The primary goal is to develop an accessible sign language recognition system. I've approached this challenge using a two-layer neural network designed specifically for static gesture classification. This work utilizes the Sign Language MNIST dataset, which contains more than 27k training images and 7k test images, covering 24 letters. Letters J and Z were excluded as they require motion to be properly represented. This project aims to enhance communication tools available to the deaf community.

# Dataset Characteristics
The dataset consists of 28 by 28 grayscale images, each containing 784 pixels/features. They are centered hand captured under varying lighting conditions. The images are labeled with the corresponding letter, ranging from A to Z, with the exception of J and Z. The dataset is well-organized and provides a solid foundation for training and evaluation.

# Dataset Distribution
The dataset is slightly imbalanced, with some letters like ‘E’ appearing more frequently than others like ‘Q.’ This influenced my training approach.

# Dataset Examples
Here you can see actual examples from the dataset for each letter (2 by letter), showing the variety of hand gestures being worked with. These images demonstrate the complexity of the recognition task and the variations that need to be accounted for.

# Preprocessing Pipeline
It's three main steps. First, data normalization by scaling pixel values to range between 0 and 1. Like we did lsat time in the class work. Second, label processing using one-hot encoding for the 24 classes, it also was explained by professora in one of our first classes. I am also making necessary adjustments for the absence of J and Z. Finally, loading data separately from training and testing sets.

# Model Architecture
The neural network I've implemented has a straightforward but effective architecture. It starts with an input layer of 784 neurons, corresponding to the pixels of the images. This feeds into a hidden layer of 256 neurons.
I tested powers of 2 (128, 256, 512) since they often provide efficient memory alignment
128 neurons proved insufficient for capturing complex hand gesture patterns
512 neurons showed signs of overfitting and increased computational cost
256 neurons provided the optimal balance: enough capacity to learn meaningful features while avoiding overfitting

And finally to an output layer of 24 neurons - one for each letter being recognized. For activation functions I wanted to use ReLu because it provides faster training and computional effectivity, but decided to go with sigmoid activation as we did during our classes.
The model contains 207,128 trainable parameters initialized using Xavier initialization.
I used Xavier initialization to ensure proper initial weight scaling, particularly important with Sigmoid activation to prevent saturation early in training.

# Initialization Strategy

First, I calculated two specific scaling factors for each layer. For the first layer, which connects our 784 input neurons to 256 hidden neurons, the scaling factor epsilon-1 is approximately 0.084. For the second layer, connecting 256 hidden neurons to 24 output neurons, epsilon-2 is about 0.149.

For the weight matrices, I used uniform random initialization within these bounds. The first weight matrix W1 is initialized randomly between negative and positive epsilon-1, the same approach for  W2.

For the bias vectors, I took a simpler approach. Both bias vectors - b1 for the hidden layer and b2 for the output layer - are initialized to zero. The hidden layer bias is a 256-dimensional vector, while the output layer bias is 24-dimensional.

This Xavier initialization technique provides three key benefits: it helps prevent the vanishing and exploding gradient problems that often occur in deep networks, maintains consistent activation variance as signals propagate through layers, and enables faster convergence during training.

# Momentum

I also implemented momentum in my training process and why it's important.

In standard gradient descent, we simply update the weights by subtracting the gradient multiplied by the learning rate. However, this can be slow and might get stuck in poor solutions.

That's why I used momentum with a beta value of 0.9. The momentum update has two steps: first, we compute a velocity term that combines the current gradient with previous updates, where beta determines how much past updates influence the current one. Then, we use this velocity to update the weights.

Think of it like a ball rolling down a hill. Without momentum, the ball moves directly downhill at each point. With momentum, it's like the ball builds up speed – keeping 90% of its previous velocity while also being pushed by the current gradient.

This approach provides three key benefits: it accelerates training when gradients point in consistent directions, helps escape local minima by maintaining momentum through small uphill sections, and reduces oscillations by smoothing out the updates.
In practice, the combination of momentum with my learning rate of 0.1 helped achieve stable and efficient training of the network.

# Training Strategy

For the complete training strategy, I combined momentum with same parameters. Let me walk you through them.
For optimization, I used mini-batches of 64 samples - large enough for stable gradients. The initial learning rate was set to 0.1, which works well with the momentum-based approach.

To prevent overfitting, I implemented two key techniques. First, L2 regularization (ridge regularization) with λ = 0.01 prevents weights from growing too large. Second, I decreased the learning rate by a factor of 0.95 every 50 steps, allowing for precise fine-tuning as training progresses.

The loss function uses binary cross-entropy, suitable for our classification task, combined with the L2 regularization term. This combination helps balance between accurate predictions and model simplicity.

# Math Framework

For Forward Propagation, my network processes data in two layers. In the first layer, I compute Z₁ as the product of weights W₁ and input X plus bias b₁, then apply the sigmoid activation function to get A₁. The second layer follows the same pattern: Z₂ is computed using weights W₂ and the previous activation A₁, followed by another sigmoid activation to produce the final output A₂.

The Loss Function has two parts. The first is Binary Cross-Entropy (BCE), which measures how well my predictions match the true labels across all 24 letter classes. I add to this an L2 regularization term, which penalizes large weights by summing their squares. This combination helps prevent overfitting while maintaining good prediction accuracy.

For updating the weights, I use a momentum-based approach where each update considers both the current gradient and previous velocity. The velocity v_W accumulates gradients over time with momentum coefficient β, and then updates the weights accordingly.

<!-- Finally, I implement an adaptive learning rate that decays exponentially. Starting with initial rate α₀, I multiply it by 0.95 every 50 training steps. This schedule allows for larger updates early in training when we're far from the optimum, and smaller, more precise updates as we get closer to convergence. -->

This mathematical framework combines these elements to create a robust and effective training process for sign language recognition.

# Cost Over Time Analysis
Looking at the training progress, I can identify three distinct phases. Initially, there's a sharp drop in cost from 4.0 to about 1.5 within the first 10 iterations, indicating rapid learning of major patterns.

The middle phase, between iterations 10 and 40, shows a more gradual improvement with some fluctuations around iteration 20, as the model fine-tunes its understanding of the sign language patterns.

Finally, after iteration 40, we see steady convergence with minor oscillations, stabilizing around 0.5. This smooth convergence suggests that my chosen hyperparameters – the learning rate decay, momentum, and L2 regularization – worked effectively together to find a stable solution without overfitting.

# Performance Analysis

The model achieved a test accuracy of 77.36%, with some letters like ‘O’ and ‘B’ achieving near-perfect recognition. However, challenges remain for letters like ‘R’ and ‘S,’ which showed F1-scores below 0.57. These results point to both the strengths of the network and areas for improvement

# Confusion Matrix
The confusion matrix reveals several interesting patterns.

First, looking at the strong performers: Letter 'A' shows perfect recognition with 331 correct predictions and no confusion with other letters. Similarly, letters like 'B' and 'D' demonstrate strong diagonal values with minimal misclassifications.
However, there are several notable confusion patterns:

'R' is often confused with 'U', showing 26 instances of misclassification
'T' and 'W' frequently get mixed up, with 62 cases of 'T' being classified as 'W'
'N' and 'M' show mutual confusion, suggesting similarity in their hand gestures

These patterns suggest that letters with similar hand shapes pose the biggest challenge for the model. For example, 'R' and 'U' both involve extended fingers in similar positions, while 'T' and 'W' share some common gestural elements.

# Demo
I've implemented an interactive web interface that brings our trained model to life. The page uses user's webcam feed to capture and recognize sign language gestures in real-time.

The interface shows both the original capture and its processed version, so you can see exactly how the image is transformed before being fed into the neural network. When you capture an image, it's automatically converted to grayscale, scaled down to match our 28×28 input size, and normalized just like our training data.

You can see the activation values at each layer, helping understand how the network makes its decisions. The system displays the top three predictions with confidence scores, giving insight into not just the best guess but also alternative interpretations.

# Improvements & Extensions
Having developed both the core model and a working web implementation, there are several promising directions for future work:
For model enhancements, data augmentation techniques could specifically target the confusions we observed in our confusion matrix. The web interface already gives us a platform to collect new training data, which could help balance our dataset and improve recognition of challenging letters.

The current web implementation could be extended into a more powerful tool. Integration with mobile devices' cameras would make it more accessible, and adding support for continuous recognition would allow for more natural signing flow.

On the research side, expanding to dynamic gesture recognition would allow us to include letters like J and Z, which require motion. The network debug information in our web interface already provides insights that could guide the development of a more sophisticated architecture.

Each of these improvements aims to make this technology more practical and accessible as a real-world communication tool.

# Conclusion
In conclusion, I've successfully developed a foundation for static gesture recognition in sign language. My implementation achieves a good balance between performance and complexity, while clearly identifying paths for future improvement. This work contributes to the development of accessible communication tools and provides valuable benchmarks for future implementations in sign language recognition systems.
