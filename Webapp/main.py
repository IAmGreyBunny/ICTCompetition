from flask import Flask
from flask_restful import Api, Resource,reqparse

import BERTriage.detect
from BERTriage import detect

#Loads model
#nlp_model = BERTriage.detect.load_model("MODEL PATH")

app = Flask(__name__)
api = Api(app)

bertriage_api_args = reqparse.RequestParser()
bertriage_api_args.add_argument("data",type=str,help="Sentence Data is missing",required=True)

class bertriage_api(Resource):
    def post(self):
        args = bertriage_api_args.parse_args()
        data = args['data']
        print(data)
        #BERTriage.detect.make_prediction(nlp_model,args["data"])
        return


api.add_resource(bertriage_api,"/bertriage_api")

if __name__ == "__main__":
    app.run(debug=True)
