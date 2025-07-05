import os
from datasets import load_dataset, Dataset
import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from typing import List

os.environ['HF_HOME'] = '/scratch/' + \
    str(open('../tokens/HPC_ACCOUNT_ID.txt', 'r').read())
cache_dir = '/scratch/' + \
    str(open('../tokens/HPC_ACCOUNT_ID.txt', 'r').read()) + '/cache'


DATASET = "LeoZotos/bio_full"
WIKI = "en"  # or 'simple'


hf_api_key = ""
with open("../tokens/HF_TOKEN.txt", "r") as f:
    hf_api_key = f.read().strip()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


question_set = load_dataset(DATASET, split='train',
                            token=hf_api_key, cache_dir=cache_dir)


def retrieve_relevant_docs(
        question_set: Dataset,
        passages: Dataset,
        device: torch.device,
        top_k: int = 20,
        batch_size: int = 204800,
        num_workers: int = 48,) -> List[List[str]]:
    """
    !!!This function was written by Gemini 2.5!!!
    Retrieves the top_k relevant documents for each question using an optimized,
    asynchronous DataLoader to fix CPU bottlenecks.

    Args:
        question_set: A Hugging Face Dataset containing questions. Each row must have an 'emb' column
                      with the question embedding.
        passages: A Hugging Face Dataset of passages. Each row must have an 'emb' column
                  with the passage embedding, and 'title' and 'text' columns.
        device: The torch.device to run the computation on (e.g., torch.device('cuda:0')).
        top_k: The number of top documents to retrieve for each question.
        batch_size: The number of passages to process in a single batch. Adjust based on GPU memory.
                    A larger batch_size is faster but uses more memory.
        num_workers: The number of CPU processes to use for data loading.
                     A value > 0 enables asynchronous data loading.

    Returns:
        A list of lists, where each inner list contains the top_k relevant documents
        (formatted as "title text") for a corresponding question.
    """
    # Important: Set the format for the passages dataset *once* before creating the DataLoader.
    # We only need the 'emb' column for the computation loop.
    passages.set_format(
        "torch",
        columns=['emb'],
        dtype=torch.bfloat16
    )

    # 1. Load all question embeddings into GPU memory.
    num_questions = len(question_set)
    question_embs = torch.tensor(
        question_set['emb'], dtype=torch.bfloat16).to(device)

    # 2. Initialize tensors to store top scores and their corresponding passage indices.
    top_k_scores = torch.full(
        (num_questions, top_k), -torch.inf, device=device, dtype=torch.bfloat16)
    top_k_indices = torch.full(
        (num_questions, top_k), -1, device=device, dtype=torch.long)

    # 3. Set up the DataLoader for efficient, asynchronous data loading.
    #    - pin_memory=True speeds up CPU-to-GPU transfers.
    #    - num_workers > 0 uses background processes to load data, so the GPU doesn't wait.
    passage_loader = DataLoader(
        passages,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False  # Ensure all passages are processed
    )

    # 4. Iterate through the passages using the new efficient DataLoader.
    global_passage_idx = 0
    for batch in tqdm(passage_loader, desc="Finding top passages"):
        # `batch` is now a dictionary {'emb': tensor_on_cpu}.
        # The DataLoader has already prepared this batch in the background.
        # .to(device, non_blocking=True) starts the GPU transfer and returns immediately.
        passage_embs_batch = batch['emb'].to(device, non_blocking=True)

        # The GPU can start computing on the previous batch while this one is transferring.
        # (Though in this specific loop, the benefit is mainly that the CPU can
        # start preparing the *next* batch while this one transfers).
        batch_scores = torch.matmul(question_embs, passage_embs_batch.T)

        combined_scores = torch.cat([top_k_scores, batch_scores], dim=1)
        top_k_scores, relative_indices = torch.topk(
            combined_scores, top_k, dim=1)

        # Use the actual batch size, which might be smaller for the last batch
        current_batch_size = passage_embs_batch.shape[0]
        batch_global_indices = torch.arange(
            global_passage_idx, global_passage_idx + current_batch_size, device=device
        ).expand(num_questions, -1)

        combined_indices = torch.cat(
            [top_k_indices, batch_global_indices], dim=1)
        top_k_indices = torch.gather(combined_indices, 1, relative_indices)

        global_passage_idx += current_batch_size

    # Before the final step, reset the format to get 'title' and 'text'
    passages.reset_format()

    # 5. Retrieve the documents using the final top_k_indices.
    top_k_indices_cpu = top_k_indices.cpu().numpy()

    relevant_docs_combined = []
    print("Retrieving final documents...")
    for q_indices in tqdm(top_k_indices_cpu, desc="Formatting results"):
        valid_indices = [idx for idx in q_indices if idx != -1]
        # Fetching documents one by one can be slow, but it's correct.
        # For very large datasets, batch fetching could be another optimization.
        docs = passages[valid_indices]
        formatted_docs = [f"{title} {text}" for title,
                          text in zip(docs['title'], docs['text'])]
        relevant_docs_combined.append(formatted_docs)

    return relevant_docs_combined


passages = load_dataset("Cohere/wikipedia-2023-11-embed-multilingual-v3",
                        WIKI, split="train", cache_dir=cache_dir, token=hf_api_key)

relevant_docs = retrieve_relevant_docs(question_set, passages, device)


def replace_column_data(example, idx):
    example["Relevant_Docs_" + WIKI] = relevant_docs[idx]
    return example


question_set = question_set.map(replace_column_data, with_indices=True)

question_set.push_to_hub(
    repo_id=DATASET,
    commit_message="Added relevant documents from Wiki Simple",
    token=hf_api_key,
    private=True
)
