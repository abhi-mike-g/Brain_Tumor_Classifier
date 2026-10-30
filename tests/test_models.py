import unittest

import torch

from btc.models import build_model


class ModelShapeTests(unittest.TestCase):
    def test_all_models_return_four_class_logits(self):
        common = {
            "classes": ["glioma", "meningioma", "notumor", "pituitary"],
            "image": {"size": 224},
            "hidden_dims": [128, 64],
            "downsample_size": 28,
            "lstm_hidden_size": 64,
            "patch_size": 16,
            "embed_dim": 96,
            "depth": 2,
            "num_heads": 3,
            "pretrained": False,
        }
        images = torch.randn(1, 3, 224, 224)
        for model_name in ("mlp", "resnet18", "cnn_lstm", "vit"):
            with self.subTest(model=model_name):
                model = build_model({**common, "model_name": model_name}).eval()
                with torch.inference_mode():
                    logits = model(images)
                self.assertEqual(tuple(logits.shape), (1, 4))


if __name__ == "__main__":
    unittest.main()
