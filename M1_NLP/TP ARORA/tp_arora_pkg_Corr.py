### Packages


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings

from gensim.models import Word2Vec

from sklearn.decomposition import TruncatedSVD


from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, brier_score_loss, confusion_matrix
)


### PREPROCESS



def load_text_classification_dataset(
    path_data_csv,
    sep=",",
    encoding="ISO-8859-1",
    var_target="target",
    var_text="text"
):
    """
    Load a text–target dataset and clean it.

    Tries several encodings (UTF-8, ISO-8859-1, latin).  
    Keeps only text and target, renames them, and converts the target to 0/1.
    
    Parameters
    ----------
    path_data_csv : path to CSV
    sep : column separator
    encoding : initial encoding to try
    var_target : name of target column
    var_text : name of text column
    """
    
    tried = [encoding, "utf-8", "ISO-8859-1", "latin"]
    df = None

    for enc in tried:
        try:
            df = pd.read_csv(path_data_csv, sep=sep, encoding=enc)
            print(f"Loaded successfully with encoding='{enc}'.")
            break
        except Exception:
            continue
    
    if df is None:
        raise ValueError(f"Could not read file. Tried encodings: {tried}")

    # Keep only the first two columns if dataset has extras
    if df.shape[1] > 2:
        df = df.iloc[:, :2]

    df.columns = [var_target, var_text]  
    df = df[[var_text, var_target]]  

    df[var_target] = df[var_target].apply(
        lambda x: 1 if str(x).lower() == "spam" else 0
    )

    return df



##### DESCRIPTIVE STATS



def descriptive_stats_text_target(df, var_text="text", var_target="target"):
    """
    Compute basic descriptive statistics for text and target columns.
    
    Returns a DataFrame with type info, missing values, 
    text lengths, and target class frequencies.
    """
    
    stats = {}

    # Text statistics
    stats[var_text] = {
        "Type": df[var_text].dtype,
        "Count": df[var_text].count(),
        "Missing": df[var_text].isnull().sum(),
        "Mean length": round(df[var_text].str.len().mean(), 2),
        "Max length": df[var_text].str.len().max(),
        "Min length": df[var_text].str.len().min()
    }

    # Target statistics
    stats[var_target] = {
        "Type": df[var_target].dtype,
        "Count": df[var_target].count(),
        "Missing": df[var_target].isnull().sum(),
        "Class frequencies": df[var_target].value_counts().to_dict()
    }

    return pd.DataFrame(stats).transpose()



def plot_target_distribution(df, var_target="target"):
    """
    Plot the class distribution of the target column (barplot + pie chart).
    """
    
    target_counts = df[var_target].value_counts(normalize=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Barplot
    sns.barplot(
        x=target_counts.index,
        y=target_counts.values,
        hue=target_counts.index,
        palette=["#3498db", "#e74c3c"],
        dodge=False,
        legend=False,
        ax=axes[0]
    )
    axes[0].set_xlabel("Target")
    axes[0].set_ylabel("Percentage")
    axes[0].set_title("Class distribution – Barplot")

    # Pie chart
    axes[1].pie(
        target_counts.values,
        labels=target_counts.index,
        autopct="%1.1f%%",
        colors=["#3498db", "#e74c3c"],
        startangle=90,
        counterclock=False
    )
    axes[1].set_title("Class distribution – Pie chart")

    plt.tight_layout()
    plt.show()




##### NLP PROCESS


def tokenize(text):
    """
    Naive tokenizer: lowercase + split on spaces.
    Students are encouraged to experiment with more advanced tokenizers.
    """
    return text.lower().split()




def extract_word_embeddings(df, vec_size=100):
    """
    Train a Word2Vec model on the text column and extract word-embedding 
    dictionaries for each document.

    This is a simple Word2Vec setup. Higher `vec_size` captures richer 
    semantics but increases computation and risk of overfitting. 
    Returns the DataFrame with a new column containing a dict {token: embedding}.
    """
    
    # Tokenize texts
    df["tokenized_text"] = df["text"].apply(tokenize)

    # Train Word2Vec model
    model = Word2Vec(
        sentences=df["tokenized_text"],
        vector_size=vec_size,
        window=5,
        min_count=1,
        workers=4,
        seed=177
    )

    # Convert token list → embedding dictionary
    def get_word_embedding_dict(tokenized_text):
        # Keep only tokens present in the trained vocabulary
        return {word: model.wv[word] for word in tokenized_text if word in model.wv}
    
    word_embedding_list = []
    for tok_text in tqdm(df["tokenized_text"], desc="Computing word embeddings..."):
        word_embedding_list.append(get_word_embedding_dict(tok_text))

    df["word_embeddings"] = word_embedding_list

    # Store the number of embedded tokens
    df["len_dico"] = df["word_embeddings"].apply(len)

    # Remove temporary token column
    df = df.drop(columns=["tokenized_text"])

    return df


##### ARORA Package


# Disable only the specific RuntimeWarning related to "invalid value encountered in divide"
warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in divide",
    category=RuntimeWarning
)

# Simple class to store parameters
class Params:
    """
    Container for model parameters.
    
    Parameters
    ----------
    rmpc : int or float
        Value controlling the removal of the first principal component 
        (used in Arora-style sentence embedding methods).
    """
    def __init__(self, rmpc):
        self.rmpc = rmpc


def get_weighted_average(We, x, w):
    """
    Compute the weighted average vectors
    :param We: We[i,:] is the vector for word i
    :param x: x[i, :] are the indices of the words in sentence i
    :param w: w[i, :] are the weights for the words in sentence i
    :return: emb[i, :] are the weighted average vector for sentence i
    """
    n_samples = x.shape[0]
    emb = np.zeros((n_samples, We.shape[1]))
    for i in range(n_samples):
        emb[i,:] = w[i,:].dot(We[x[i,:],:]) / np.count_nonzero(w[i,:])
    return emb

def compute_pc(X, npc=1):
    """
    Compute the principal components. DO NOT MAKE THE DATA ZERO MEAN!
    :param X: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: component_[i,:] is the i-th pc
    """
    # Remplacer les NaN et valeurs infinies par 0
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Vérification supplémentaire pour s'assurer que la matrice est correctement nettoyée
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        print("Attention : La matrice contient encore des valeurs non numériques après nettoyage.")

    # Appliquer TruncatedSVD
    svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)
    svd.fit(X)
    return svd.components_

def remove_pc(X, npc=1):
    """
    Remove the projection on the principal components
    :param X: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: XX[i, :] is the data point after removing its projection
    """
    pc = compute_pc(X, npc)
    if npc==1:
        XX = X - X.dot(pc.transpose()) * pc
    else:
        XX = X - X.dot(pc.transpose()).dot(pc)
    return XX


def SIF_embedding(We, x, w, params):
    """
    Compute the scores between pairs of sentences using weighted average + removing the projection on the first principal component
    :param We: We[i,:] is the vector for word i
    :param x: x[i, :] are the indices of the words in the i-th sentence
    :param w: w[i, :] are the weights for the words in the i-th sentence
    :param params.rmpc: if >0, remove the projections of the sentence embeddings to their first principal component
    :return: emb, emb[i, :] is the embedding for sentence i
    """
    emb = get_weighted_average(We, x, w)
    if  params.rmpc > 0:
        emb = remove_pc(emb, npc = params.rmpc)
    return emb



# Function to compute SIF weights
def compute_sif_weights(word_occurence, total_words, a=1e-3):
    """
    Compute SIF weights for each word.

    Parameters
    ----------
    word_occurence : dict
        Dictionary mapping words to their raw counts in the document.
    total_words : int
        Total number of words in the document.
    a : float
        Smoothing parameter for SIF.

    Returns
    -------
    dict
        Mapping {word: SIF weight}.
    """
    sif_weights = {word: a / (a + (freq / total_words)) 
                   for word, freq in word_occurence.items()}
    return sif_weights


def process_report(word_embeddings, a_arora=1e-3):
    """
    Process one document: extract embeddings, compute SIF weights,
    and prepare matrices for the Arora sentence embedding method.
    """

    all_words = []
    all_embeddings = []
    word_occurence = {}

    # Collect embeddings and compute word frequencies for the current document
    for word, embedding in word_embeddings.items():
        all_words.append(word)
        all_embeddings.append(embedding)
        word_occurence[word] = word_occurence.get(word, 0) + 1

    # Convert embeddings to a numpy matrix
    We = np.vstack(all_embeddings)

    # Compute SIF weights
    total_words = len(all_words)
    sif_weights = compute_sif_weights(word_occurence, total_words, a=a_arora)

    # Create x and w matrices for SIF processing
    x = np.array([[i for i, _ in enumerate(all_words)]])
    w = np.array([[sif_weights[word] for word in all_words]])

    return We, x, w


def arora_methods(
    df,
    var_text="text",
    var_word_embd="word_embeddings",
    name_var_sent="sentence_embeddings",
    remove_pc_nbr=1,
    a_arora=1e-3
):
    """
    Compute sentence embeddings using the Arora SIF method.

    Parameters
    ----------
    df : DataFrame
        Input dataset containing a column of word-level embeddings.
    var_text : str
        Name of the text column (kept for compatibility; not used directly).
    var_word_embd : str
        Name of the column containing dictionaries {token: word_embedding}.
    name_var_sent : str
        Name of the output column that will store the sentence embeddings.
    remove_pc_nbr : int
        Number of principal components to remove (default = 1).
    a_arora : float
        Smoothing parameter for SIF weighting (default = 1e-3).

    Returns
    -------
    DataFrame
        The input DataFrame with an additional column `name_var_sent`
        containing one sentence embedding per document.
    """

    params = Params(rmpc=remove_pc_nbr)

    # List to store the computed sentence embeddings
    sentence_embeddings_list = []

    # Compute embeddings with progress bar
    for we in tqdm(df[var_word_embd], desc="Computing sentence embeddings..."):
        # Apply the full Arora pipeline on each document
        We, x, w = process_report(we, a_arora=a_arora)
        sentence_embedding = SIF_embedding(We, x, w, params)

        sentence_embeddings_list.append(sentence_embedding)

    # Remove extra dimensions if SIF returns shape (1, N)
    sentence_embeddings_list = [np.squeeze(e) for e in sentence_embeddings_list]

    # Store results in the DataFrame
    df[name_var_sent] = sentence_embeddings_list

    return df



class SpamClassificationPipeline:
    """
    A rigorous end-to-end pipeline that:
    - preprocesses the Arora embeddings
    - trains several models
    - computes prediction metrics
    - plots confusion matrices (optional)
    - consolidates results into a final DataFrame
    """

    def __init__(self, df_arora, target_col="target", embedding_col="sentence_embeddings"):
        self.df = df_arora
        self.target_col = target_col
        self.embedding_col = embedding_col

    # --------------------------------------------------------------
    # 1. PREPROCESSING
    # --------------------------------------------------------------
    def prepare_xy(self, test_size=0.2, random_state=177):
        """Convert embeddings to array and perform a train/test split."""
        X = np.vstack(self.df[self.embedding_col].values)
        y = self.df[self.target_col].values

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        return self.X_train, self.X_test, self.y_train, self.y_test

    # --------------------------------------------------------------
    # 2. FIT MODEL
    # --------------------------------------------------------------
    def fit_model(self, model):
        """Train a given model on the training data."""
        model.fit(self.X_train, self.y_train)
        return model

    # --------------------------------------------------------------
    # 3. PREDICT
    # --------------------------------------------------------------
    def predict_model(self, model):
        """Generate predictions and probabilities."""
        y_pred = model.predict(self.X_test)
        y_prob = model.predict_proba(self.X_test)[:, 1]
        return y_pred, y_prob

    # --------------------------------------------------------------
    # 4. METRICS
    # --------------------------------------------------------------
    def evaluate_model(self, model_name, y_pred, y_prob):
        """Compute standard classification metrics."""
        acc = accuracy_score(self.y_test, y_pred)
        prec = precision_score(self.y_test, y_pred)
        rec = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        brier = brier_score_loss(self.y_test, y_prob)

        return {
            "model": model_name,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "brier_score": brier
        }

    # --------------------------------------------------------------
    # 5. CONFUSION MATRIX
    # --------------------------------------------------------------
    def plot_confusion_matrix(self, y_pred, model_name="Model"):
        """Draw a well-formatted confusion matrix."""
        cm = confusion_matrix(self.y_test, y_pred)

        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Not Spam (0)", "Spam (1)"],
            yticklabels=["Not Spam (0)", "Spam (1)"],
            linewidths=0.5,
            cbar=False
        )
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(f"Confusion Matrix – {model_name}")
        plt.tight_layout()
        plt.show()

    # --------------------------------------------------------------
    # 6. FULL ORCHESTRATION
    # --------------------------------------------------------------
    def run_all_models(self, models_dict, plot_confusion=False, verbose=True):
        """
        Run the entire workflow for each model:
        - preprocessing
        - training
        - predictions
        - metrics
        - confusion matrix (optional)
        Returns a consolidated results DataFrame.
        """

        self.prepare_xy()
        results_list = []

        for model_name, model in models_dict.items():
            if verbose:
                print(f"\n=== Training {model_name} ===")

            # Train model
            model = self.fit_model(model)

            # Predict
            y_pred, y_prob = self.predict_model(model)

            # Evaluate
            metrics = self.evaluate_model(model_name, y_pred, y_prob)
            results_list.append(metrics)

            if verbose:
                print("\n--- Evaluation Metrics ---")
                print(f"Accuracy      : {metrics['accuracy']:.4f}")
                print(f"Precision     : {metrics['precision']:.4f}")
                print(f"Recall        : {metrics['recall']:.4f}")
                print(f"F1-score      : {metrics['f1_score']:.4f}")
                print(f"Brier Score   : {metrics['brier_score']:.4f}")

            # Confusion matrix
            if plot_confusion:
                self.plot_confusion_matrix(y_pred, model_name=model_name)

        # Consolidated DataFrame
        results_df = pd.DataFrame(results_list)
        
        results_df = results_df.sort_values(by="f1_score", ascending=False).reset_index(drop=True)
        
        return results_df
