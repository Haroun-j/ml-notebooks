# ml-notebooks

Lab notebooks from my engineering studies: the engineering cycle at CY Tech (2023-2026, NLP, deep learning,
optimisation) and the preparatory cycle at EILCO (2021-2023, data analysis with Python). Each notebook is one
lab session or exam, kept as it was handed in, with the cell outputs when the notebook was run on Colab.
Most of them are written in French because the courses were. They are individual lab work by Haroun Joudi
unless the notebook says otherwise; the course staff wrote the exercise statements and the data loading cells
in the data analysis TDs.

## NLP

| notebook | task | data | method | result |
|---|---|---|---|---|
| `nlp/sms_spam_classification.ipynb` | spam / ham | SMS Spam Collection (5,574 SMS, Hugging Face) | BoW + Naive Bayes, TF-IDF + logistic regression / linear SVM, 5-fold CV, grid search, ROC | F1 0.94 for Naive Bayes and the linear SVM on a 20 % test split |
| `nlp/allocine_sentiment_fr.ipynb` | positive / negative review | Allociné, 160k French film reviews | TF-IDF + logistic regression / linear SVM against a Keras 1D CNN | linear SVM 93.5 % test accuracy, CNN 90.7 % |
| `nlp/agnews_topic_classification.ipynb` | 4 news topics | AG News (120k train, 7.6k test) | TF-IDF + logistic regression / linear SVM, 1D CNN | linear SVM 92.7 % on the validation split |
| `nlp/banking77_intent_classification.ipynb` | 77 banking intents | Banking77 | TF-IDF + linear models, 1D CNN | pipeline written, not run to the end |
| `nlp/reddit_jokes_virality.ipynb` | will a joke score high | Reddit jokes | spaCy lemmatisation, TF-IDF models, Sentence-BERT + logistic regression | pipeline written, not run to the end |
| `nlp/tweets_sentiment_exam.ipynb` | 3-class sentiment (exam) | tweet_eval, 2,000 tweets | TF-IDF + logistic regression / LinearSVC, then a zero-shot twitter-roberta comparison | macro-F1 0.48 (logistic regression); roberta 0.79 on 50 tweets |
| `nlp/lstm_text_generation.ipynb` | character-level language model | Voltaire corpus, 816k characters | Embedding + LSTM in PyTorch, temperature and top-k sampling, save / reload | validation loss 1.19 after 20 epochs on a T4 |
| `nlp/nlp_regex_basics.ipynb` | first NLP session | NLTK twitter_samples | regular expressions, tokenisation, stop words, POS tagging | |

## Deep learning

| notebook | task | data | method | result |
|---|---|---|---|---|
| `deep_learning/cnn_facial_expressions.ipynb` | facial expression classification | muxspace/facial_expressions, 13,690 images | small CNN in PyTorch, then augmentation + dropout to fight overfitting | validation accuracy 80.7 % baseline, 84.6 % after the changes |
| `deep_learning/leaf_margin_classification.ipynb` | smooth vs toothed leaf margin | 6,527 leaf photos (course dataset) | small CNN, mixed precision, early stopping, TensorBoard, confusion matrices | 93.9 % accuracy and macro-F1 0.94 on 981 test images |
| `deep_learning/mnist_denoising.ipynb` | image denoising | MNIST and CIFAR-10 with Gaussian and salt-and-pepper noise | fully connected net, autoencoder and U-Net compared on MSE, PSNR, SSIM | U-Net: PSNR 27.8 dB / SSIM 0.992 on MNIST, 31.9 dB / 0.995 on CIFAR-10 |
| `deep_learning/autoencoder_images.ipynb` | image reconstruction | MNIST, CIFAR-10 (COCO loading at the end) | FCNN, fully connected autoencoder, U-Net, torchmetrics PSNR | U-Net test MSE 0.0071 on CIFAR-10 |
| `deep_learning/coco_segmentation.ipynb` | semantic segmentation | COCO val2017, 128 images | small conv net, mIoU and pixel accuracy | 40-step smoke test on CPU, no usable metric |
| `deep_learning/simple_ann_perceptron.ipynb` | logic gates and XOR | 4 binary inputs | PyTorch 2-2-1 sigmoid network, then MCP neuron and Rosenblatt perceptron in NumPy | XOR solved, 4 / 4 |
| `deep_learning/keras_cnn_mnist.ipynb` | digit classification | MNIST | Keras CNN, confusion matrix, learning curves | 98.0 % test accuracy after 10 epochs |
| `deep_learning/keras_mlp_intro.ipynb` | first Keras network | Iris | Dense(5) + softmax | 42 / 45 correct on the test split |

## Data analysis (EILCO, 2021-2023)

| notebook | content | data |
|---|---|---|
| `data_analysis/pandas_td.ipynb` | Series then DataFrames: indexing, sorting, filtering, loc / iloc | SNCF station referential (open data) |
| `data_analysis/numpy_td.ipynb` | arrays, dtypes, slicing, an image as a NumPy array | |
| `data_analysis/seaborn_td.ipynb` | countplot, histplot, boxplot, relplot, catplot | seaborn tips |
| `data_analysis/titanic_eda.ipynb` | groupby, survival rates, missing values with missingno, then the seaborn session on the same data | Titanic |
| `data_analysis/spotify_eda.ipynb` | most frequent artists and genres, durations, BPM | Spotify top 10s 2010-2019 (Kaggle) |
| `data_analysis/pokemon_battles_analysis.ipynb` | data preparation, win rates per Pokémon and per type, correlation heatmap, logistic regression on stat differences (56.8 % vs 52.6 % for a win-rate rule) | Pokédex + 50,000 battles (Kaggle) |
| `data_analysis/covid19_exam.ipynb` | exam 2022-2023: NaN counts, filtering by date, sorting, plots | OWID covid-data.csv |
| `data_analysis/cp2_exam_temperatures.ipynb` | past exam: reshape 206 years of daily temperatures, lineplots 1815 vs 2020 | Greenwich station |
| `data_analysis/cp2_exam_phone_records.ipynb` | past exam: calls, SMS and data per month with groupby | phone usage log |
| `data_analysis/interpolation_ode.ipynb` | Lagrange and Newton interpolation, explicit Euler and Heun on logistic and Malthusian growth | |

## Optimisation and misc

| notebook | content |
|---|---|
| `optimisation/tsp_heuristics.ipynb` | simulated annealing on 20 random cities, effect of the temperature schedule (44.8 % shorter than the initial tour) |
| `optimisation/vega_multiobjective.ipynb` | VEGA genetic algorithm on a three-objective selection problem with a weight constraint |
| `optimisation/lp_pulp.ipynb` | small integer program with PuLP, feasible regions of linear inequalities drawn with matplotlib |
| `misc/graphviz_automaton.ipynb` | a finite automaton rendered with graphviz |
| `misc/graph_chain_check.ipynb` | checks whether a list of nodes is a chain in a graph |

## Running the notebooks

Open any notebook in Google Colab (File > Open notebook > GitHub, paste the file URL) or locally:

```
pip install -r requirements.txt
jupyter lab
```

`requirements.txt` lists every package the notebooks import; a single notebook needs a few of them. The deep
learning notebooks expect a GPU for the training cells (they ran on Colab T4s). Datasets are downloaded by the
notebooks themselves from Hugging Face, torchvision or GitHub, or were course files (Titanic, stations,
temperatures, phone log, Pokémon CSVs, leaf photos): put those next to the notebook before running.

## Notes

- The data analysis TDs follow the teacher's statement cell by cell; my answers are the code cells and
  the comments under the questions.
- `banking77_intent_classification` and `reddit_jokes_virality` have their pipeline written but were not run
  through, so they carry no numbers.
- Progress bars, install logs and Colab widget state were removed from the outputs; the code was not changed,
  except one typo in `lp_pulp.ipynb` (`import pip pulp` to `import pulp`).
