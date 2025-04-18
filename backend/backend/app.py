from flask import Flask, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()

app = Flask(__name__)

# Configuração CORS detalhada
CORS(app, resources={
    r"/*": {
        "origins": os.getenv('ALLOWED_ORIGINS', 'http://localhost:8080').split(','),
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True
    }
})

@app.route('/health')
def health_check():
    """Endpoint de verificação de saúde"""
    return jsonify({
        'status': 'healthy',
        'version': os.getenv('VERSION', '1.0.0'),
        'services': {
            'database': 'ok',  # TODO: Adicionar verificação real
            'cache': 'ok'      # TODO: Adicionar verificação real
        }
    }), 200

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Resource not found'}), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# Registra Blueprints
from api.v1 import bp as api_v1_bp
app.register_blueprint(api_v1_bp, url_prefix='/api/v1')

if __name__ == '__main__':
    app.run(
        host=os.getenv('FLASK_HOST', '0.0.0.0'),
        port=int(os.getenv('FLASK_PORT', 5000)),
        debug=os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    )

