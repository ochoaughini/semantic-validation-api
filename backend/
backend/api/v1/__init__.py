from flask import Blueprint

bp = Blueprint('api_v1', __name__)

# Importa rotas
from . import routes  # noqa

