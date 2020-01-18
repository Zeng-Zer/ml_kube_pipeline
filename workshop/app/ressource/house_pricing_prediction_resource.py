from flask_restplus import Resource
from models import HousePricingPrediction
from flask import request
from .api import api

ns = api.namespace('house_pricing')


@ns.route('/predict')
class HousePricingPredictionResource(Resource):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def get():
        bedrooms = float(request.args.get('bedrooms', 0))
        bathrooms = float(request.args.get('bathrooms', 0))
        sqft_living = float(request.args.get('sqft_living', 0))
        sqft_lot = float(request.args.get('sqft_lot', 0))
        floors = float(request.args.get('floors', 0))
        sqft_above = float(request.args.get('sqft_above', 0))
        sqft_basement = float(request.args.get('sqft_basement', 0))
        yr_built = float(request.args.get('yr_built', 1900))

        params = [bedrooms, bathrooms, sqft_living, sqft_lot, floors, sqft_above, sqft_basement, yr_built]

        model = HousePricingPrediction()
        result = model.predict(params)

        return {
            'input': params,
            'output': result,
        }
