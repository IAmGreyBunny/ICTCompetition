import requests

json_data = {"data": "he is having from heart attack"}

response = requests.post("http://127.0.0.1:5000/bertriage_api", json=json_data)
print(response.text)

# import BERTriage.detect
#
# #Loads model
# data = "My head pain"
# nlp_model = BERTriage.detect.load_model(r"D:\ICT Competition\Model\model.hdf5")
# print(BERTriage.detect.make_prediction(nlp_model,data))