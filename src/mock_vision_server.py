#!/usr/bin/env python3
"""
模拟视觉后端服务器
- 接收图片请求后，随机从 iot_taxonomy.json 中选择一个子类别
- 返回符合 vision_client.py search() 契约的 fake visual_candidates
- 用于开发和测试环境，替代真实的视觉分类模型
"""

import json
import random
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs
import cgi
import os
import sys

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('MockVisionServer')

# 全局变量：加载的 taxonomy 数据
TAXONOMY = None
SUBCATEGORIES = []  # 扁平化的子类别列表 [{coarse_id, coarse_name, sub_id, sub_name, intro}, ...]


def load_taxonomy(taxonomy_path):
    """加载 iot_taxonomy.json 并构建扁平化的子类别列表"""
    global TAXONOMY, SUBCATEGORIES
    
    if not os.path.exists(taxonomy_path):
        logger.error(f"Taxonomy file not found: {taxonomy_path}")
        sys.exit(1)
    
    with open(taxonomy_path, 'r', encoding='utf-8') as f:
        TAXONOMY = json.load(f)
    
    SUBCATEGORIES = []
    for coarse in TAXONOMY['coarse_categories']:
        coarse_id = coarse['id']
        coarse_name = coarse['name']
        for sub in coarse['subclasses']:
            SUBCATEGORIES.append({
                'coarse_id': coarse_id,
                'coarse_name': coarse_name,
                'sub_id': sub['id'],
                'sub_name': sub['name'],
                'intro': sub['intro']
            })
    
    logger.info(f"Loaded {len(SUBCATEGORIES)} subcategories from {taxonomy_path}")


class MockVisionHandler(BaseHTTPRequestHandler):
    """HTTP 请求处理器"""
    
    def do_GET(self):
        """处理 GET 请求 - 只支持 /health"""
        if self.path == '/health':
            self._send_json_response({
                'status': 'ok',
                'mode': 'mock',
                'subcategories_count': len(SUBCATEGORIES)
            })
        else:
            self._send_error(404, 'Not Found')
    
    def do_POST(self):
        """处理 POST 请求 - 支持 /api/vision/search"""
        if self.path == '/api/vision/search':
            self._handle_search()
        else:
            self._send_error(404, 'Not Found')
    
    def _handle_search(self):
        """
        模拟视觉搜索：
        1. 解析 multipart/form-data（接收 image 字段）
        2. 随机选择一个子类别
        3. 返回符合 vision_client 契约的 visual_candidates
        """
        content_type = self.headers.get('Content-Type', '')
        
        # 读取请求体（只是为了消费数据，实际上不需要处理图片）
        content_length = int(self.headers.get('Content-Length', 0))
        request_body = self.rfile.read(content_length)
        
        # 随机选择一个子类别
        if not SUBCATEGORIES:
            self._send_error(500, 'No subcategories loaded')
            return
        
        selected = random.choice(SUBCATEGORIES)
        logger.info(f"Selected subcategory: {selected['coarse_id']}/{selected['sub_id']} - {selected['sub_name']}")
        
        # 构造 doc_id（格式：{coarse_id}/{sub_id}，与 iot_loader 生成的 document 对应）
        doc_id = f"{selected['coarse_id']}/{selected['sub_id']}"
        
        # 构造响应（符合 vision_client.py search() 的返回格式）
        response = {
            'status': 'ok',
            'coarse_category': selected['coarse_name'],
            'coarse_confidence': 0.95,  # 模拟高置信度
            'coarse_status': 'ok',
            'visual_candidates': [
                {
                    'doc_id': doc_id,
                    'sub_category': selected['sub_name'],
                    'coarse_category': selected['coarse_name'],
                    'score': 0.92,  # 模拟匹配分数
                    'evidence_image_id': f"mock_{selected['sub_id']}_001.jpg",
                    'matched_image_count': 1
                }
            ]
        }
        
        self._send_json_response(response)
    
    def _send_json_response(self, data, status_code=200):
        """发送 JSON 响应"""
        response_body = json.dumps(data, ensure_ascii=False)
        encoded_body = response_body.encode('utf-8')
        
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(encoded_body)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(encoded_body)
    
    def _send_error(self, status_code, message):
        """发送错误响应"""
        logger.error(f"Error {status_code}: {message}")
        self._send_json_response({
            'status': 'error',
            'error': message
        }, status_code)
    
    def log_message(self, format, *args):
        """覆盖默认的日志方法，使用 logger"""
        logger.info(f"{self.client_address[0]} - {format % args}")


def run_server(host='127.0.0.1', port=9101):
    """启动模拟视觉后端服务器"""
    server_address = (host, port)
    httpd = HTTPServer(server_address, MockVisionHandler)
    
    logger.info(f"Mock Vision Backend started on {host}:{port}")
    logger.info(f"  - Health check: http://{host}:{port}/health")
    logger.info(f"  - Search API:   http://{host}:{port}/api/vision/search")
    logger.info(f"  - Loaded subcategories: {len(SUBCATEGORIES)}")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down Mock Vision Backend...")
        httpd.shutdown()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Mock Vision Backend Server')
    parser.add_argument('--host', default='127.0.0.1', help='Server host (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=9101, help='Server port (default: 9101)')
    parser.add_argument('--taxonomy', default='data/iot_knowledge/iot_taxonomy.json',
                        help='Path to iot_taxonomy.json')
    
    args = parser.parse_args()
    
    # 加载 taxonomy
    load_taxonomy(args.taxonomy)
    
    # 启动服务器
    run_server(host=args.host, port=args.port)


if __name__ == '__main__':
    main()
