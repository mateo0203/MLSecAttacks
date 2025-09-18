import torch
import torch.nn as nn

class ResnetPGDAttacker:
    def __init__(self, model, eps=8/255, alpha=2/255, steps=10, device="cpu"):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.device = device

        # Standard ImageNet normalization
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    def pgd_attack(self, images, labels):
        images = images.to(self.device)
        labels = labels.to(self.device)

        # Convert eps and alpha from pixel space to normalized space
        eps_norm = (self.eps / self.std)
        alpha_norm = (self.alpha / self.std)

        # === TODO 1: Random start within L_inf ball ===
        adv_images = images + torch.empty_like(images).uniform_(-1, 1) * eps_norm
        adv_images = torch.min(torch.max(adv_images, (0 - self.mean) / self.std), (1 - self.mean) / self.std)
        adv_images = adv_images.clone().detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.model(adv_images)
            loss = nn.CrossEntropyLoss()(outputs, labels)

            self.model.zero_grad()
            loss.backward()
            grad = adv_images.grad.data

            # === TODO 2: Gradient ascent step (sign of grad, scaled by alpha) ===
            adv_images = adv_images + alpha_norm * grad.sign()

            # === TODO 3: Projection step to L_inf ball and clamp ===
            delta = adv_images - images
            delta = torch.clamp(delta, -eps_norm, eps_norm)
            adv_images = images + delta
            adv_images = torch.min(torch.max(adv_images, (0 - self.mean) / self.std), (1 - self.mean) / self.std)
            adv_images = adv_images.detach()

        return adv_images

    def pgd_batch_attack(self, dataloader):
        adv_images_list = []
        labels_list = []
        for images, labels in dataloader:
            adv_images = self.pgd_attack(images, labels)
            adv_images_list.append(adv_images.cpu())
            labels_list.append(labels.cpu())
        adv_images = torch.cat(adv_images_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        return adv_images, labels
