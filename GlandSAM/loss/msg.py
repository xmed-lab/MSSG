import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossClassPullLoss(nn.Module):
    def __init__(self, num_samples=1000, distance='cosine'):
        super().__init__()
        self.num_samples = num_samples
        self.distance = distance

    def forward(self, embeddings, labels):
        """
        embeddings: [1, C, H, W] - pixel embedding
        labels: [1, H, W] - pixel labels (0 or 1)
        """
        B, C, H, W = embeddings.shape
        assert B == 1, "Only batch size 1 is supported"

        # Flatten spatial dimensions
        embeddings = embeddings.view(C, -1).transpose(0, 1)  # [H*W, C]
        labels = labels.view(-1)  # [H*W]

        # Get indices of class 0 and class 1
        idx_0 = (labels == 0).nonzero(as_tuple=True)[0]
        idx_1 = (labels == 1).nonzero(as_tuple=True)[0]

        if len(idx_0) == 0 or len(idx_1) == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        # Sample pairs
        num_pairs = min(self.num_samples, len(idx_0), len(idx_1))
        idx_0_sample = idx_0[torch.randperm(len(idx_0))[:num_pairs]]
        idx_1_sample = idx_1[torch.randperm(len(idx_1))[:num_pairs]]

        emb_0 = embeddings[idx_0_sample]  # [num_pairs, C]
        emb_1 = embeddings[idx_1_sample]  # [num_pairs, C]

        # Compute distance
        if self.distance == 'cosine':
            loss = 1 - F.cosine_similarity(emb_0, emb_1).mean()
        elif self.distance == 'euclidean':
            loss = F.pairwise_distance(emb_0, emb_1).mean()
        else:
            raise ValueError("Unsupported distance type")

        return loss

class SemanticContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(SemanticContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, embeddings, annotations):
        """
        embeddings: Tensor of shape [1, C, H, W]
        annotations: Tensor of shape [1, H, W], values in {0, 1, 255}
        """
        B, C, H, W = embeddings.shape
        assert B == 1, "Only support batch size 1 for now"

        annotations = annotations.squeeze(0)  # shape: [H, W]
        embeddings = embeddings.squeeze(0).permute(1, 2, 0)  # shape: [H, W, C]

        # Flatten
        flat_embed = embeddings.view(-1, C)  # [N, C]
        flat_ann = annotations.view(-1)  # [N]

        # Masks
        mask_0 = flat_ann == 0
        mask_1 = flat_ann == 1
        mask_255 = flat_ann == 255

        emb_0 = flat_embed[mask_0]
        emb_1 = flat_embed[mask_1]
        emb_255 = flat_embed[mask_255]

        loss = 0.0
        eps = 1e-8
        count = 0

        # Pull 0 and 1 embeddings closer
        if emb_0.size(0) > 0 and emb_1.size(0) > 0:
            # Random sampling to reduce compute cost
            idx_0 = torch.randint(0, emb_0.size(0), (min(1000, emb_0.size(0)),))
            idx_1 = torch.randint(0, emb_1.size(0), (min(1000, emb_1.size(0)),))

            sim_pairs = F.pairwise_distance(emb_0[idx_0][:, None, :], emb_1[idx_1][None, :, :])
            pull_loss = sim_pairs.mean()
            loss += pull_loss
            count += 1

        # Push 0 vs 255
        if emb_0.size(0) > 0 and emb_255.size(0) > 0:
            idx_0 = torch.randint(0, emb_0.size(0), (min(1000, emb_0.size(0)),))
            idx_f = torch.randint(0, emb_255.size(0), (min(1000, emb_255.size(0)),))

            dist = F.pairwise_distance(emb_0[idx_0][:, None, :], emb_255[idx_f][None, :, :])
            push_loss_0 = F.relu(self.margin - dist).mean()
            loss += push_loss_0
            count += 1

        # Push 1 vs 255
        if emb_1.size(0) > 0 and emb_255.size(0) > 0:
            idx_1 = torch.randint(0, emb_1.size(0), (min(1000, emb_1.size(0)),))
            idx_f = torch.randint(0, emb_255.size(0), (min(1000, emb_255.size(0)),))

            dist = F.pairwise_distance(emb_1[idx_1][:, None, :], emb_255[idx_f][None, :, :])
            push_loss_1 = F.relu(self.margin - dist).mean()
            loss += push_loss_1
            count += 1

        if count > 0:
            return loss / count
        else:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)