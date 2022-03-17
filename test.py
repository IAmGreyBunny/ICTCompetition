# import requests
#
# json_data = {"data": "Hi this is a test"}
#
# response = requests.post("http://127.0.0.1:5000/bertriage_api", json=json_data)
# print(response.text)

import BERTriage.detect

#Loads model
nlp_model = BERTriage.detect.load_model(r"D:\ICT Competition\Model\model.hdf5")
print(BERTriage.detect.make_prediction(nlp_model,"he is suffering from some slight burns"))
