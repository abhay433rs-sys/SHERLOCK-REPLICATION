import pandas as pd
from minicons import scorer



# 1. Load Model
model = scorer.IncrementalLMScorer(
    "Qwen/Qwen2-0.5B",
    device="cpu"
)


# 2. Load CSV
df = pd.read_csv(r"E:\Project LLM\Model_test\Dataset_Sentences.csv")

# 3. Remove rows where PV, IV or IO is empty
df = df.dropna(subset=["PV", "IV", "IO"])

# 4. Remove completely blank/whitespace sentences
df["PV"] = df["PV"].astype(str).str.strip()
df["IV"] = df["IV"].astype(str).str.strip()
df["IO"] = df["IO"].astype(str).str.strip()

df = df[
    (df["PV"] != "") &
    (df["IV"] != "") &
    (df["IO"] != "")
]

df = df.reset_index(drop=True)

# 5. Build sentence list INTERLEAVED by row/triplet:
sentences = []
meta = [] 

for row_id, row in df.iterrows():
    for stype in ["PV", "IV", "IO"]:
        sentences.append(row[stype])
        meta.append((row_id, stype))

print("Number of rows:", len(df))
print("Number of sentences:", len(sentences))

# 6. Get token-level surprisal
scores = model.token_score(
    sentences,
    surprisal=True,
    base_two=True
)



SPACE_MARKERS = ["Ġ", "▁"]

def strip_marker(token):
    for m in SPACE_MARKERS:
        if token.startswith(m):
            return token[len(m):]
    return token


# 7. Align tokens to whitespace-split words.
def tokens_to_words_aligned(sentence, sentence_scores):

    target_words = sentence.split()
    words_out = []
    idx = 0
    n_tokens = len(sentence_scores)

    for target in target_words:
        current = ""
        surprisal_sum = 0.0

        while idx < n_tokens:
            token, surprisal = sentence_scores[idx]
            clean = token.lstrip("Ġ")

            current += clean
            surprisal_sum += surprisal
            idx += 1

            if current == target:
                break
            if len(current) > len(target):
                break

        words_out.append((target, surprisal_sum))

    if idx < n_tokens:
        leftover = "".join(t.lstrip("Ġ") for t, _ in sentence_scores[idx:])
        print(f"  [WARNING] Unconsumed tokens for sentence: {sentence!r} -> {leftover!r}")

    return words_out


# 8. Print results
all_results = []
current_triplet = None

for (row_id, stype), sentence, sentence_scores in zip(meta, sentences, scores):

    if row_id != current_triplet:
        print("\n" + "#" * 70)
        print(f"# TRIPLET {row_id}")
        print("#" * 70)
        current_triplet = row_id

    word_scores = tokens_to_words_aligned(sentence, sentence_scores)

    print("=" * 60)
    print(f"[{stype}] Sentence: {sentence}")
    print("=" * 60)

    total_surprisal = sum(s for _, s in word_scores)
    avg_surprisal = total_surprisal / len(word_scores)

    print(f"{'Word':<20} {'Surprisal (bits)':<20}")
    for word, surprisal in word_scores:
        print(f"{word:<20} {surprisal:<20.4f}")
        all_results.append({
            "triplet_id": row_id,
            "type": stype,
            "word": word,
            "surprisal": surprisal
        })

    print(f"\nTotal Surprisal   : {total_surprisal:.4f} bits")
    print(f"Avg Surprisal/Word: {avg_surprisal:.4f} bits\n")

# 9. Save results
results_df = pd.DataFrame(all_results)

type_order = {"PV": 0, "IV": 1, "IO": 2}
results_df["type_order"] = results_df["type"].map(type_order)
results_df = results_df.sort_values(["triplet_id", "type_order"]).drop(columns="type_order")

results_df.to_csv(r"E:\Project LLM\Model_test\qwen2-0.5b_V0.1_SURPRSE.csv", index=False)
print("Saved word-level surprisal results (grouped by triplet) to qwen2-0.5b_V0.1_SURPRSE.csv")
