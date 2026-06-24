# 视觉模型 HTTP 服务 (电脑B)
# 职责:
#   - 接收前端上传的物联网设备照片 (multipart/form-data)
#   - 调用视觉模型进行设备分类
#   - 返回分类结果及置信度
#
# 端点:
#   POST /api/vision/classify
#        form: image=<file>
#        → {"category": "温湿度传感器", "confidence": 0.95}
#   GET  /health
#        → {"status": "ok", "model_loaded": true}
#
# 依赖:
#   - src.vision.vision_model  (模型加载与推理)
#   - config.settings          (全局配置)
#
# 部署:
#   本服务运行在电脑B（带 GPU），电脑A 和手机通过局域网 IP 访问
