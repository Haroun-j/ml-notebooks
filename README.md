# ml-notebooks

31 lab notebooks from five years of engineering studies: Python data analysis at EILCO, then NLP, deep learning and optimisation at CY Tech. Each is one lab session or exam, kept as handed in with its Colab outputs, mostly in French.

## What it covers

- `nlp/` (8): spam, sentiment, topic and intent classification with TF-IDF, linear models and 1D CNNs, a character-level LSTM, a zero-shot twitter-roberta exam, a regex intro.
- `deep_learning/` (8): CNNs for facial expressions and leaf margins, denoising with autoencoders and a U-Net, COCO segmentation, Keras on MNIST and Iris, a perceptron from scratch.
- `data_analysis/` (10): pandas, NumPy and seaborn sessions, EDA on Titanic, Spotify and Pokémon battles, three past exams, interpolation and ODE schemes.
- `optimisation/` (3): simulated annealing on a 20-city TSP, 44.8 % shorter than the start tour, a VEGA multi-objective genetic algorithm, PuLP integer programming.
- `misc/` (2): a graphviz automaton and a chain check on a graph.

## Results

Single runs as recorded in each notebook, on its own split.

| notebook | data | result |
|---|---|---|
| `nlp/sms_spam_classification` | 5,574 SMS | F1 0.94, Naive Bayes and linear SVM, 20 % test split |
| `nlp/allocine_sentiment_fr` | 160k French reviews | linear SVM 93.5 % vs 1D CNN 90.7 % test accuracy |
| `nlp/agnews_topic_classification` | AG News, 120k train | linear SVM 92.7 % validation accuracy |
| `deep_learning/cnn_facial_expressions` | 13,690 images | CNN 80.7 % validation, 84.6 % with augmentation and dropout |
| `deep_learning/leaf_margin_classification` | 6,527 leaf photos | CNN 93.9 %, macro-F1 0.94 on 981 test images |
| `deep_learning/mnist_denoising` | noisy MNIST, CIFAR-10 | U-Net PSNR 27.8 / 31.9 dB, SSIM 0.992 / 0.995 |

TF-IDF falls short on the three-class tweets exam: macro-F1 0.48 for logistic regression, 0.79 for zero-shot twitter-roberta on 50 tweets. The banking77 and reddit jokes notebooks were never run through and carry no numbers.

## How to run

Open a notebook in Google Colab (File > Open notebook > GitHub), or run `pip install -r requirements.txt` and `jupyter lab` locally.

## Limitations

- Course files are not included: Titanic, SNCF stations, temperatures, phone log, Pokémon CSVs and leaf photos go next to the notebook. The rest downloads from Hugging Face, torchvision or GitHub.
- Outputs are kept as recorded; only progress bars, install logs and widget state were removed, plus a typo in `lp_pulp.ipynb`.
- The deep learning notebooks need a GPU for training; they ran on Colab T4s.

Preparatory cycle at EILCO (2021-2023), engineering cycle at CY Tech (2023-2026). The notebooks are my individual work unless stated inside; the course staff wrote the statements and data loading cells of the data analysis sessions.
