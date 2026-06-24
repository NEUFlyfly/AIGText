# 统一开发入口
# 职责:
#   - 同时启动电脑A 电脑B 两套服务 (开发调试用)
#   - 启动前端静态文件服务
#
# 用法 (需在项目根目录执行):
#   python run.py                     # 同时启动全部服务
#   python run.py --lang-only         # 仅启动电脑A
#   python run.py --vision-only       # 仅启动电脑B
#   python run.py --frontend-only     # 仅启动前端
#
# 部署方式 (生产环境、分机器部署):
#   - 电脑A: bash scripts/start_lang_server.sh
#   - 电脑B: bash scripts/start_vision_server.sh
#   - 前端: 手机浏览器打开 电脑B_IP:18081/frontend，
#           或用 nginx 托管 frontend/ 目录
