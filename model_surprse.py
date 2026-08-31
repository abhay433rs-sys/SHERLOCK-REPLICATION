from minicons import scorer
import math

model = scorer.IncrementalLMScorer(
    "EleutherAI/pythia-160m",
    device="cpu"
)


# Your three sentences
sentences = [
    "The doctor is giving medicine to an old woman.",
    "The doctor is welding medicine to an old woman.",
    "The doctor is giving fire to an old woman."
]

# Get surprisal
scores = model.token_score(
    sentences,
    surprisal=True,
    base_two=True
)

# Print token, surprisal, and probability
for sentence, sentence_scores in zip(sentences, scores):

    print("\nSentence:")
    print(sentence)

    print("\nToken        Surprisal        Probability")
    print("-------------------------------------------")

    for token, surprisal in sentence_scores:

        # Convert surprisal to probability
        probability = 2 ** (-surprisal)

        print(
            f"{token:<12} "
            f"{surprisal:<15.4f} "
            f"{probability:.10f}"
        )


#output:

# Sentence:
# The doctor is giving medicine to an old woman.

# Token        Surprisal        Probability
# -------------------------------------------
# The          0.0000          1.0000000000
# Ġdoctor      12.2656         0.0002030855
# Ġis          5.7695          0.0183315013
# Ġgiving      8.6562          0.0024786152
# Ġmedicine    12.2656         0.0002030855
# Ġto          0.7212          0.6065962960
# Ġan          7.9336          0.0040902543
# Ġold         4.3281          0.0497856922
# Ġwoman       2.1641          0.2231270764
# .            3.6074          0.0820460748

# Sentence:
# The doctor is welding medicine to an old woman.

# Token        Surprisal        Probability
# -------------------------------------------
# The          0.0000          1.0000000000
# Ġdoctor      12.2656         0.0002030855
# Ġis          5.7695          0.0183315013
# Ġwelding     18.7500         0.0000022682
# Ġmedicine    13.7031         0.0000749805
# Ġto          4.3281          0.0497856922
# Ġan          7.9336          0.0040902543
# Ġold         4.3281          0.0497856922
# Ġwoman       5.0508          0.0301691686
# .            2.8848          0.1353938745

# Sentence:
# The doctor is giving fire to an old woman.

# Token        Surprisal        Probability
# -------------------------------------------
# The          0.0000          1.0000000000
# Ġdoctor      12.2656         0.0002030855
# Ġis          5.7695          0.0183315013
# Ġgiving      8.6562          0.0024786152
# Ġfire        15.8672         0.0000167302
# Ġto          4.3281          0.0497856922
# Ġan          7.9336          0.0040902543
# Ġold         2.8848          0.1353938745
# Ġwoman       3.6074          0.0820460748
# .            2.8848          0.1353938745
