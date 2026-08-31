# import pandas as pd
# from tqdm import tqdm
# from minicons import scorer

# MODEL_NAME = "EleutherAI/pythia-160m"

# model = scorer.IncrementalLMScorer(
#     MODEL_NAME,
#     device="cpu"
# )




# def get_token_scores(sentence):

#     scores = model.token_score(
#         [sentence],
#         surprisal=True,
#         base_two=True
#     )

#     return scores[0]


# def get_target_word_score(sentence, target_word):

#     # Get token-level surprisal
#     token_scores = get_token_scores(sentence)

#     # How does this model tokenize the target word?
#     target_tokens = model.tokenizer.tokenize(target_word)

#     # Clean tokenizer markers for comparison
#     target_tokens_clean = [
#         token.replace("Ġ", "").lower()
#         for token in target_tokens
#     ]

#     # Clean the sentence tokens
#     sentence_tokens_clean = [
#         token.replace("Ġ", "").lower()
#         for token, _ in token_scores
#     ]

#     target_length = len(target_tokens_clean)

#     # Search for the target token sequence inside the sentence
#     for i in range(
#         len(sentence_tokens_clean) - target_length + 1
#     ):

#         window = sentence_tokens_clean[
#             i:i + target_length
#         ]

#         if window == target_tokens_clean:

#             selected_scores = token_scores[
#                 i:i + target_length
#             ]

#             # Get the individual subtoken surprisals
#             subtoken_surprisals = [
#                 surprisal
#                 for _, surprisal in selected_scores
#             ]

#             # Whole-word surprisal
#             total_surprisal = sum(
#                 subtoken_surprisals
#             )

#             # Whole-word probability
#             probability = 2 ** (-total_surprisal)

#             return {
#                 "target_word": target_word,
#                 "model_tokens": [
#                     token
#                     for token, _ in selected_scores
#                 ],
#                 "subtoken_surprisals":
#                     subtoken_surprisals,
#                 "whole_word_surprisal":
#                     total_surprisal,
#                 "whole_word_probability":
#                     probability
#             }

#     return None

# # sentence = "The doctor saw POOP."

# # result = get_target_word_score(
# #     sentence,
# #     "POOP"
# # )

# # print(result)

# pv = "The doctor is giving medicine to an old woman."
# iv_verb = "The doctor is welding medicine to an old woman."
# iv_object = "The doctor is giving fire to an old woman."

# pv_verb_result = get_target_word_score(
#     pv,
#     "giving"
# )

# iv_verb_result = get_target_word_score(
#     iv_verb,
#     "welding"
# )

# pv_object_result = get_target_word_score(
#     pv,
#     "medicine"
# )

# iv_object_result = get_target_word_score(
#     iv_object,
#     "fire"
# )

# print("\nPV verb:")
# print(pv_verb_result)

# print("\nIV-Verb:")
# print(iv_verb_result)

# print("\nPV object:")
# print(pv_object_result)

# print("\nIV-Object:")
# print(iv_object_result)

# verb_surprisal_difference = (
#     iv_verb_result["whole_word_surprisal"]
#     - pv_verb_result["whole_word_surprisal"]
# )

# object_surprisal_difference = (
#     iv_object_result["whole_word_surprisal"]
#     - pv_object_result["whole_word_surprisal"]
# )




from huggingface_hub import scan_cache_dir

cache_info = scan_cache_dir()
for repo in sorted(cache_info.repos, key=lambda r: -r.size_on_disk):
    print(f"{repo.repo_id:40s} {repo.size_on_disk / 1e9:8.2f} GB")

print(f"\nTotal cache size: {cache_info.size_on_disk / 1e9:.2f} GB")