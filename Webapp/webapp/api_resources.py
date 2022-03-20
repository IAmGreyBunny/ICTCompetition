from flask_restful import Resource, reqparse
from webapp import api
from webapp import db
from webapp.models import Patient, Bed
import BERTriage.detect
from BERTriage.model_config import label_map

# Loads NLP model
nlp_model = BERTriage.detect.load_model(r"D:\ICT Competition\Model\6_class_model.hdf5")

# Configuring bertriage api call
bertriage_api_args = reqparse.RequestParser()
bertriage_api_args.add_argument("data", type=str, help="Sentence Data is missing", required=True)


# Bertriage api resource object
class bertriage_api(Resource):
    def post(self):
        args = bertriage_api_args.parse_args()
        medical_request = args["data"]
        triage_category = BERTriage.detect.make_prediction(nlp_model, medical_request)
        patient = Patient(triage_category=triage_category, request=medical_request, status="awaiting treatment")
        if triage_category == label_map[0]:
            db.session.add(patient)
            db.session.commit()
            ward = "Intensive Care Unit"
            bed = Bed(ward=ward)
            patient.patient_bed = bed
            db.session.add(bed)
            db.session.commit()
        elif triage_category == label_map[1]:
            db.session.add(patient)
            db.session.commit()
            ward = "High Dependency Unit"
            bed = Bed(ward=ward)
            patient.patient_bed = bed
            db.session.add(bed)
            db.session.commit()
        else:
            db.session.add(patient)
            db.session.commit()
        return


# Add bertriage api
api.add_resource(bertriage_api, "/bertriage_api")
