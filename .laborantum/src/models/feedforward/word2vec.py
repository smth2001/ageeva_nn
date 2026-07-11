import math
import torch
import torch.nn.functional as F


class BinaryIndexTree:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.depth = math.ceil(math.log2(vocab_size))
        self.max_path_length = self.depth
        self.num_internal_nodes = 2 ** self.depth - 1

    def path_and_targets(self, word_index):
        leaf_index = self.num_internal_nodes + int(word_index)

        path = []
        targets = []

        current = leaf_index
        while current > 0:
            parent = (current - 1) // 2
            target = 0.0 if current == 2 * parent + 1 else 1.0

            path.append(parent)
            targets.append(target)

            current = parent

        return path[::-1], targets[::-1]

    def __call__(self, context_word):
        device = context_word.device
        word_indices = context_word.detach().cpu().flatten().tolist()

        paths = []
        targets = []

        for word_index in word_indices:
            path, target = self.path_and_targets(word_index)
            paths.append(path)
            targets.append(target)

        path = torch.tensor(paths, dtype=torch.long, device=device)
        targets = torch.tensor(targets, dtype=torch.float32, device=device)

        mask = torch.ones(
            (len(word_indices), self.max_path_length),
            dtype=torch.float32,
            device=device,
        )

        return {
            "path": path,
            "targets": targets,
            "mask": mask,
        }


class HierarchicalSoftmax(torch.nn.Module):
    def __init__(self, embedding_dim, vocab_size):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size

        self.tree = BinaryIndexTree(vocab_size)
        self.num_internal_nodes = self.tree.num_internal_nodes
        self.max_path_length = self.tree.max_path_length

        self.decoder = torch.nn.Embedding(
            self.num_internal_nodes,
            embedding_dim,
        )

    def forward(self, embedding, target_word):
        tree_result = self.tree(target_word)

        path = tree_result["path"]
        targets = tree_result["targets"]
        mask = tree_result["mask"]

        node_embeddings = self.decoder(path)
        logits = (node_embeddings * embedding.unsqueeze(1)).sum(dim=-1)

        probabilities = torch.sigmoid(logits)

        target_probabilities = torch.where(
            targets.bool(),
            probabilities,
            1.0 - probabilities,
        )

        total_probability = target_probabilities.prod(dim=1)

        loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
        )
        loss = (loss * mask).sum(dim=1).mean()

        return {
            "path": path,
            "targets": targets,
            "mask": mask,
            "logits": logits,
            "probabilities": probabilities,
            "target_probabilities": target_probabilities,
            "total_probability": total_probability,
            "loss": loss,
        }


class Word2Vec(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.encoder = torch.nn.Embedding(vocab_size, embedding_dim)

        self.hierarchical_softmax = HierarchicalSoftmax(
            embedding_dim=embedding_dim,
            vocab_size=vocab_size,
        )

        self.decoder = self.hierarchical_softmax.decoder

    def forward(self, batch):
        data = batch["data"]

        center_word = data.get("center_word", data.get("center"))
        context_word = data.get("context_word", data.get("context"))

        if center_word is None:
            raise KeyError(f"center_word/center not found. Available keys: {list(data.keys())}")

        if context_word is None:
            raise KeyError(f"context_word/context not found. Available keys: {list(data.keys())}")

        embedding = self.encoder(center_word)

        hsoftmax_result = self.hierarchical_softmax(
            embedding,
            context_word,
        )

        batch.setdefault("signals", {})
        batch.setdefault("losses", {})

        batch["signals"]["embedding"] = embedding
        batch["signals"]["logits"] = hsoftmax_result["logits"]
        batch["signals"]["probabilities"] = hsoftmax_result["probabilities"]
        batch["signals"]["total_probability"] = hsoftmax_result["total_probability"]

        batch["data"]["path"] = hsoftmax_result["path"]
        batch["data"]["targets"] = hsoftmax_result["targets"]
        batch["data"]["mask"] = hsoftmax_result["mask"]

        batch["losses"]["loss"] = hsoftmax_result["loss"]

        return batch
