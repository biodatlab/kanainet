from PIL import Image
from torchvision import transforms
import torch
from kan_acnet import KANACNet


class KANPredictor:
    def __init__(self, weights):
        self.model = KANACNet()
        self.model.load_state_dict(torch.load(weights, map_location="cpu"))
        self.model.eval()

    def __call__(self, image_path):
        img = transforms.ToTensor()(Image.open(image_path).convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            return (torch.sigmoid(self.model(img)) > 0.5).squeeze().numpy()


def visualize(image_path, mask):
    import matplotlib.pyplot as plt
    import numpy as np

    image   = np.array(Image.open(image_path).convert("RGB"))
    overlay = image.copy()
    overlay[mask > 0] = [255, 0, 0]

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1); plt.imshow(image);            plt.title("Original"); plt.axis("off")
    plt.subplot(1, 3, 2); plt.imshow(mask, cmap="gray"); plt.title("Mask");    plt.axis("off")
    plt.subplot(1, 3, 3); plt.imshow(overlay);           plt.title("Overlay"); plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import sys
    weights  = sys.argv[1]   # model.pth
    img_path = sys.argv[2]   # test.jpg

    kan  = KANPredictor(weights)
    mask = kan(img_path)
    visualize(img_path, mask)