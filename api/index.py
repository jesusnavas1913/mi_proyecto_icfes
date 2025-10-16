# Handler de Vercel para exponer la app Flask (WSGI)
# No mover la importación: debe apuntar a la instancia `app` que ya existe en icfes_api.py
from icfes_api import app as flask_app
import vercel_wsgi


def handler(request, context):
    """Punto de entrada serverless de Vercel.
    Convierte la app WSGI de Flask a una respuesta compatible con Vercel.
    """
    return vercel_wsgi.handle(flask_app, request, context)
