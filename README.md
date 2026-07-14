# 1. Start by setting up the environment

```shell
make          # creates .venv and installs everything from requirements.txt
source .venv/bin/activate
```

--> All the dependencies have been installed, we can now start.

# 2. The dataset

The dataset is a collection of leaf pictures split into 8 classes (4 apple, 4 grape), each folder being one class:

```
leaves/images/
├── Apple_Black_rot
├── Apple_healthy
├── Apple_rust
├── Apple_scab
├── Grape_Black_rot
├── Grape_Esca
├── Grape_healthy
└── Grape_spot
```

--> Every image is a `.JPG`, and the goal is to predict the class of a leaf from its picture.

# 3. What we need to know

A **Convolutional Neural Network (CNN)** is a network specialized in images. It has two parts working back to back:

1. A **convolutional part** that *looks* at the image and extracts a summary of what it contains (edges, textures, disease spots, leaf shapes...).
2. A **classifier part** (a small MLP) that takes this summary and decides which class it belongs to.

The key intuition: instead of connecting every pixel to every neuron (which would be huge and wasteful), the CNN uses small **filters** that slide over the image. The same filter is reused everywhere, so a "brown spot detector" works whether the spot is in the top-left or bottom-right of the leaf.

## 3.1 The convolutional part

The image enters as a tensor of shape `(3, 128, 128)`: 3 color channels (Red, Green, Blue), each a 128×128 grid of pixel values.

It then goes through **4 conv blocks**. Each block does the same 4 operations:

### a. Convolution — sliding a small filter over the image

A **filter** (or *kernel*) is a tiny 3×3 grid of weights. Imagine putting a 3×3 window on the top-left corner of the image, multiplying each pixel under the window by the corresponding filter weight, summing everything → this gives **one number**. Then slide the window one pixel to the right and repeat, until the whole image is covered.

The result is a new 2D grid called a **feature map**: bright where the filter's pattern was found, dark where it wasn't. Different filters detect different things: one might light up on horizontal edges, another on green textures, another on brownish spots.

In our code, each conv layer applies *many* filters at once (32, then 64, then 128, then 256), producing that many feature maps stacked together.

### b. Batch Normalization — keeping the numbers stable

After a convolution, the numbers can drift very large or very small, which makes training unstable. BatchNorm rescales the values of each feature map to have roughly mean 0 and variance 1. Think of it as a "volume knob" that automatically re-centers the signal.

### c. ReLU — keeping only what matters

ReLU is a very simple function: `max(0, x)`. It replaces every negative value with 0 and keeps positive ones as-is. Without this step, stacking layers would be pointless (a stack of linear operations is still linear). ReLU adds the non-linearity that lets the network learn *complex* patterns, not just averages of pixels.

### d. Pooling — zooming out

**MaxPool2d(2)** takes each 2×2 square of the feature map and keeps only the **maximum** value. So a 128×128 map becomes 64×64. Why do this?

- It **reduces the size** (less computation for the next layer).
- It keeps the **strongest signal** in each region ("is there a spot somewhere in this 2×2 area? yes/no") and drops the exact location, which makes the network robust to small shifts.
- It **increases the field of view** of the next filter: after pooling, each 3×3 filter now covers a bigger real-world region of the original image.

After the last conv block we replace `MaxPool2d` with **AdaptiveAvgPool2d(1)**: it collapses each feature map into a single number (its average). We end up with a vector of 256 numbers — a compact summary of the image.

### Putting it together: the pipeline of sizes

Starting from `(3, 128, 128)`, each block roughly does `conv → BN → ReLU → pool`:

| Step        | Channels | Spatial size |
|-------------|----------|--------------|
| Input image | 3        | 128 × 128    |
| Block 1     | 32       | 64 × 64      |
| Block 2     | 64       | 32 × 32      |
| Block 3     | 128      | 16 × 16      |
| Block 4     | 256      | 1 × 1  (adaptive avg pool) |

Notice how the **spatial size shrinks** while the **number of channels grows**: the network progressively trades "where" for "what". The first blocks know precisely where edges are; the last blocks don't care about position anymore, they just know *which high-level features* are present in the image.

![CNN](https://liora.io/app/uploads/2020/06/convolutif-1024x288.png)

## 3.2 The classifier part (the "decision" of the network)

At this point we have a vector of 256 numbers describing the image. The classifier is a small **MLP** that turns this vector into class scores:

1. **Flatten** — turns the `(256, 1, 1)` tensor into a flat vector of 256 numbers.
2. **Dropout(0.3)** — randomly zeroes out 30% of these numbers during training. This forces the network not to rely on any single feature and reduces overfitting.
3. **Linear layer** — a fully-connected layer with 256 inputs and 8 outputs (one score per class).

The predicted class is simply the one with the highest score.

- **Backpropagation** is exactly the same idea as in a regular neural network: after a forward pass we compare the prediction to the true label, compute a loss (here **cross-entropy**, since we have several classes), and push the gradient of that loss back through every layer to know how each weight (and each filter) contributed to the error.

- **Gradient Descent** then updates the weights slightly in the direction that reduces the loss. The size of the step is the **learning rate**. This is repeated many times over the whole training set, one full pass being an **epoch**.

![Gradient Descent](https://camo.githubusercontent.com/8875a53e84240f2d99a685ae750ff71209a2b7ce9ef1697ae84e52b201412ad7/68747470733a2f2f656469746f722e616e616c79746963737669646879612e636f6d2f75706c6f6164732f39373130366764342e6a706567)

# 4. How the training loop works

1. We do a first **feedforward** pass with randomly initialized weights (batch of images in, class scores out).
2. We compute the **cross-entropy loss** between predictions and true labels, then **backpropagate** to get the gradient of every weight.
3. Using **gradient descent** (here through Adam), we update the weights slightly in the direction that reduces the error.
4. We repeat this loop for many mini-batches and many epochs, until the validation accuracy stops improving (early stopping).

# Let's explain a few things:

## Weight initialization and the optimizer

We do not initialize the weights by hand: PyTorch's default init already fits ReLU networks. We just pick a good optimizer.

--> **Adam** is used (`torch.optim.Adam`). It is the same optimizer as in a MLP: it keeps a moving average of the gradient (`m`) and of the squared gradient (`v`), and adapts the learning rate per weight.

```
m = β1 × m + (1 - β1) × gradient        β1 = 0.9
v = β2 × v + (1 - β2) × gradient²       β2 = 0.999
weight -= lr × (m / (√v + ε))           ε = small constant to avoid /0
```

## Class balancing with `augmentation.py`

The raw dataset is heavily **imbalanced** (e.g. `Apple_rust` has 275 images, `Apple_healthy` has 1640). If we train directly on it, the network learns to just predict the majority class.

To fix this, before training we call `process_directory` from `augmentation.py`, which:

1. Copies every original image into `augmented_directory/<class>/`.
2. For each minority class, generates new images from the existing ones until its size matches the largest class.

Each augmentation call produces 6 variants of one source image: horizontal flip, vertical flip, 90° rotation, blur, center crop and brightness shift. This gives the network more diverse examples of the rare classes.

--> After balancing, every class folder has (roughly) the same number of images (±6, since we generate 6 variants at a time).

## Train / validation split

The split happens **before** augmentation, on the original images only:

- 30% of the originals goes to the **validation set**.
- 70% of the originals is copied to `train_originals/`, then passed to `augmentation.process_directory` to be balanced into `augmented_directory/`.

This ordering matters: if we augmented first and split after, an original image and its 6 variants could end up spread across train and val (data leakage), and validation accuracy would no longer reflect real generalization.

The split is deterministic (seed = 42) so re-running the training gives the same folds. Both loaders use the same transform (resize + tensor), since augmentation is already done on disk by `augmentation.py` — no online augmentation on top.

## Early stopping (patience & epsilon)

We do not fix a number of epochs. Instead, at every epoch we look at the validation accuracy:

- If it improved by at least `epsilon` (default 0.01), we save the model and reset the patience counter.
- Otherwise we increment the patience counter.
- If patience exceeds the `--patience` argument (default 4), training stops.

This way training always stops right after the model stopped learning, and the saved `model.pt` is the best one seen.

# How to run the pipeline

## Look at the class distribution

```shell
python Distribution.py leaves/images
```
--> Shows a bar chart and a pie chart of how many pictures each class contains. Useful to *see* the imbalance before training.

## Visualize the image transformations

```shell
python Transformation.py --src <path/to/image.JPG>
python Transformation.py --src <path/to/folder> --dst <output_folder>
```
--> Displays (or saves) six computer-vision views of a leaf: Gaussian blur, disease mask, ROI, contour analysis, pseudolandmarks, color histogram. These are all built with OpenCV in `utils_transformation.py` (mask via LAB color space + Otsu thresholding, disease detection on the `a` channel, etc.). They are **not** used to train the CNN — the CNN learns from the raw RGB pixels — but they are useful to understand what the classifier "sees".

## Train the model

```shell
python train.py --src leaves/images --dst learnings.zip
```
Optional flags: `--batch_size`, `--lr`, `--patience`, `--epsilon`.

What happens:
1. The original images are split deterministically into train (70%) and val (30%).
2. The train split is copied to `train_originals/`, then `augmentation.process_directory` balances its classes into `augmented_directory/`.
3. Loaders are built: train from `augmented_directory/`, val from the untouched original val split.
4. The CNN is trained with Adam + cross-entropy loss + early stopping.
5. The best model is saved to `model.pt`, then bundled together with the augmented dataset into `learnings.zip`.

## Predict

```shell
python predict.py --src <path/to/image.JPG>            # one image
python predict.py --src <path/to/folder> --all         # accuracy per class
```
--> In single-image mode, it prints the predicted class and shows a side-by-side plot of the original image and its disease mask.
