# 视觉模型封装
# 职责:
#   - 加载图像分类模型 (如 ResNet / ViT / 本地 GGUF 视觉模型)
#   - 对输入图像进行预处理 (resize, normalize)
#   - 推理并返回 top-1 类别名称和置信度
#
# 接口:
#   classify(image_bytes: bytes) -> dict
#     返回: {"category": "温湿度传感器", "confidence": 0.95}
#
#   或使用文件路径:
#   classify_from_path(image_path: str) -> dict
#
# 模型选择建议:
#   - 轻量: ResNet-50 / MobileNetV3 (CPU 也可)
#   - 中文标签: 使用自定义分类头 + 中文类别映射表
#   - 本地 GGUF: 如果有 llama.cpp 视觉模型，也可复用 bin/ 下的 llama 运行时
