from flask import Flask,render_template,url_for
from flask_restful import Api, Resource,reqparse

import BERTriage.detect

# Loads NLP model
nlp_model = BERTriage.detect.load_model(r"D:\ICT Competition\Model\model.hdf5")

# Initializing flask
app = Flask(__name__)
api = Api(app)

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


@app.route("/")
def authentication_page():
    return render_template('authentication_page.html')
    #return "Hello World"


if __name__ == "__main__":
    app.run(debug=True)