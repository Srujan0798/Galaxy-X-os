# API Documentation

## Inference

```python
from src.inference import predict_image
result = predict_image("image.jpg")
print(result.class_name, result.confidence)
```

## Grad-CAM

```python
from src.gradcam import explain_image
from src.inference import ModelManager
manager = ModelManager()
result = explain_image(manager.model, "image.jpg", manager.device)
```
