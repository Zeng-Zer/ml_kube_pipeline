from flask import Flask, Blueprint
from ressource import api, house_pricing

app = Flask(__name__)

blueprint = Blueprint('api', __name__, url_prefix='/api')

api.init_app(blueprint)

app.register_blueprint(blueprint)
api.add_namespace(house_pricing)


if __name__ == '__main__':
    print("starting on 0.0.0.0:5000")
    app.run(port=5000, host='0.0.0.0')
