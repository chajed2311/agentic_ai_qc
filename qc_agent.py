import cv2
import numpy as np
from skimage.measure import shannon_entropy
import sys
import json

class QualityControlAgent:
    def __init__(self, blur_threshold=100, contrast_threshold=20, entropy_threshold=3):
        self.blur_threshold = blur_threshold
        self.contrast_threshold = contrast_threshold
        self.entropy_threshold = entropy_threshold

    def evaluate(self, image_path):
        img = cv2.imread(image_path)

        if img is None:
            return {
                "qc_status": "fail",
                "reason": "corrupted_image",
                "confidence": 0.0
            }

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        contrast = gray.std()
        entropy = shannon_entropy(gray)

        checks = [
            blur > self.blur_threshold,
            contrast > self.contrast_threshold,
            entropy > self.entropy_threshold
        ]

        confidence = sum(checks) / len(checks)

        status = "pass" if confidence >= 0.67 else "fail"

        return {
            "qc_status": status,
            "confidence": round(confidence, 2),
            "metrics": {
                "blur": round(blur, 2),
                "contrast": round(contrast, 2),
                "entropy": round(entropy, 2)
            }
        }
if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python qc_agent.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    agent = QualityControlAgent()
    result = agent.evaluate(image_path)

    print(json.dumps(result, indent=2))
