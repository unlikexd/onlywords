from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model_name = "ai-forever/rugpt3large_based_on_gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

semantic_model_name = "DeepPavlov/rubert-base-cased"
semantic_model = SentenceTransformer(semantic_model_name)


def score_fluency(sentence: str) -> float:
    # Tokenize and compute model loss
    inputs = tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    perplexity = torch.exp(loss).item()

    fluency_score = max(0.0, 1 - (perplexity / 100))
    return fluency_score


def score_semantic(original: str, modified: str) -> float:
    emb_orig = semantic_model.encode(original)
    emb_mod = semantic_model.encode(modified)
    return cosine_similarity([emb_orig], [emb_mod])[0][0]


# main function for scoring
def evaluate_modified_sentence(
        original: str,
        modified: str,
) -> float:
    fluency_mod = score_fluency(modified)
    fluency_orig = score_fluency(original)
    fluency_curve = fluency_mod - fluency_orig
    semantic = score_semantic(original, modified)

    total_score = 0.7 * fluency_curve + 0.3 * semantic
    return round(total_score, 2)
