# Computer-Vision
Driver fatigue and abnormal driving behaviors significantly contribute to road accidents worldwide. According to WHO, over 1.25 million fatalities occur annually due to vehicular accidents, with many caused by drowsiness and distractions. Weak enforcement of traffic laws, poor road infrastructure, and insufficient monitoring systems worsen the problem.

Facial fatigue detection systems leverage computer vision and deep learning to analyze yawning, eye closure, and head posture. Methods like DenseNet and Eye Aspect Ratio (EAR) provide real-time alerts. Hybrid systems integrating physiological signals (heart rate, skin conductance) improve robustness, overcoming limitations like occlusion and lighting variations.

Proposed Framework:
This study introduces a DenseNet-based anomaly detection and EAR-based fatigue analysis system, utilizing datasets like the DDD dataset from kaggle.

Architecture: DenseNet-121
Initial Layers: 7×7 convolution (64 filters) → 3×3 max pooling
Dense Blocks: Feature maps from earlier layers are reused for better learning
Transition Layers: 1×1 convolution + 2×2 pooling reduce feature maps
Final Layers: Fully connected layer for classification
