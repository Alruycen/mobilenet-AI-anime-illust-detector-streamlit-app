Link:
https://ai-anime-illust-detector.streamlit.app/?embed_options=dark_theme

Not 100% accurate, only trained with 2000 AI & Non-AI illustration
Using MobileNet V3

Reasons:
- Have higher evaluation compared to MobileNetV2 (better) and Inception V3 (faster)
- Better, smaller, and faster in comparison

No copyright yet, so I hope very much for your wise and good cooperation

Update V1
Changes to model layers:
- Omit: Dropout layers
+ Add: Batch-Normalization layers

Update V2
Changes to augmentation:
+ Rescale 1./255. -> 1./127.5 + 2
+ Increased augmentation value in general (rotation 40 -> 60, zoom 0.2 -> 0.3, shear 0.2 -> 0.3)
+ Added augmentation (brightness 0.8...1.2, channel shift 30.
+ Added test-time augmentation (TTA), applied in evaluation

Changes to model layers:
+ Add: Dropout layer (0.2), Dense L2 regularizers (1e-4), Batch-Normalization (momentum=0.9)+
+ Activation (relu -> gelu)
+ 8 more trainable layers (Dense with 512 units - Batch-Normalization - Activation GeLU - Dropout) 2x

Changes to evaluation:
+ Test-time augmentation (TTA) with ImageDataGenerator
