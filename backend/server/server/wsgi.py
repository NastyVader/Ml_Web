"""
WSGI config for server project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.0/howto/deployment/wsgi/
"""

from apps.ml.Linear_Regression.linearRegression import LinReg
from apps.ml.registry import MLRegistry
import inspect
import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'server.settings')

application = get_wsgi_application()

# ML registry

try:
    registry = MLRegistry()

    lr = LinReg()

    registry.add_algorithm(endpoint_name="RedWine Quality",
                           algorithm_object=lr,
                           algorithm_name="random forest",
                           algorithm_status="production",
                           algorithm_version="0.0.1",
                           owner="Karhik Ronad",
                           algorithm_description="Linear Regression with simple pre- and post-processing",
                           algorithm_code=inspect.getsource(LinReg))

except Exception as e:
    print("Exception while loading the algorithms to the registry,", str(e))
