from flask_restful import Resource,reqparse
from webapp import api
import BERTriage.detect

# Loads NLP model
nlp_model = BERTriage.detect.load_model(r"D:\ICT Competition\Model\model.hdf5")

# Configuring bertriage api call
bertriage_api_args = reqparse.RequestParser()
bertriage_api_args.add_argument("data",type=str,help="Sentence Data is missing",required=True)

# Bertriage api resource object
class bertriage_api(Resource):
    def post(self):
        args = bertriage_api_args.parse_args()
        data = args['data']
        print(data)
        print(BERTriage.detect.make_prediction(nlp_model,args["data"]))
        return


# Add bertriage api
api.add_resource(bertriage_api,"/bertriage_api")