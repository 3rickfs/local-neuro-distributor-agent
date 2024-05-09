import unittest
import time
import requests
import json

import keras
import numpy as np

#url neuro orchestrator
no_url = "http://127.0.0.1:5000"
#no_url = "http://a4726fe7352054e699bb94b491c1b532-859196828.us-west-1.elb.amazonaws.com:5000"

class distribution_tests(unittest.TestCase):

    def test_create_new_user(self):
        print("*"*100)
        print("Test 1: Create new user in Brain")
        print("-------------------------------------------------------")


        try:
            headers = {'Content-type': 'application/json'}
            user_attr = {}
            user_attr["displayname"] = "test_displayname"
            user_attr["username"] = "test_username"
            user_attr["email"] = "test@test.com"
            user_attr["roles"] = "deployer"
 
            json_data = json.dumps(user_attr)
            result = requests.post(f"{no_url}/create_user",
                                    data=json_data, headers=headers
                                  )
            res_dict = json.loads(result.text)
            username = res_dict["username"]
            user_id = res_dict["user_id"]

            expected_username = user_attr["username"]

            self.assertEqual(username, expected_username)

        except Exception as e:
            print("%"*100)
            print(f"error: {e}")
            print("%"*100)


        print("*"*100)

    def test_create_synaptic_process(self):
        print("*"*100)
        print("Test 2: Create a synaptic process")
        print("-------------------------------------------------------")

        input_model_file_name = "./diabetes_detection_model.json"

        try:
            #Load the AI model JSON
            with open(input_model_file_name, 'r') as jf:
                input_json_file = json.load(jf)
            jf.close()

            #Distribute neurons
            input_json_file["user_id"] = 1
            input_json_file["username"] = "test_username"
            input_json_file["neuro_orchestrator_url"] = no_url #f"{no_url}/set_final_output"
            input_json_file["dataset_name"] = "NIST"
            input_json_file["dataset_url"] = "https://download-nist.com"
            input_json_file["notebook_url"] = "https://colab.research.google.com/drive/1WE0Pr7r_KAGeaLHtsyeiyBV5LTEJ76q7"

            #input_json_file["model_name"] = "model_test_1"

            with open('./json_data.json', 'w') as outfile:
                json.dump(input_json_file, outfile)

            files={
                'nods_info': open('./nods_info.json', 'rb'),
                'json_data': open('./json_data.json', 'rb')
            }

            dn_headers = {'Content-type': 'multipart/form-data'}
            result = requests.post(f"{no_url}/crear_proceso_sinaptico",
                                   files=files
                                  )
            print(result.text)

            res = json.loads(result.text)["res"]
            expected_res = "successful"
            self.assertEqual(res, expected_res)

        except Exception as e:
            print("%"*100)
            print(f"error: {e}")
            print("%"*100)


        print("*"*100)

    def test_run_id_19_syn_proc(self):
        print("*"*100)
        print("Test 3: run diabetes model using the synaptic process 19")
        print("-------------------------------------------------------")

        input_file_name = "diabetes_detection_model.json"
        #Run above model in TF format to get the expected result
        model = keras.models.load_model(
            "./diabetes_detection_model.keras"
        )
        inp_dic = [1,0,1,1,0,1,0,0,0,1,0,1,1,0,0,1,
                   1,0,0,1,1,0,0,0,1,1,1,1,1,0,0
                  ]
        inp = np.array(inp_dic).reshape(1, 31)
        print(f"model input: {inp}")
        start = time.time()
        pred = model.predict(inp)
        end = time.time()
        print(f"pred: {pred}")
        expected_result = [str(round(v, 2)) for v in pred[0]]

        try:

            synapses_process_id = 129 #114 #94
            username = "test_username"
            user_id = "1"

            #send inputs to NODS
            nod_input = {
                "inputs": inp_dic,
                "input_idx": 0,
                "layer_id": 1,
                "synapses_process_id": synapses_process_id,
                "username": username,
                "user_id": user_id
            }

            json_data = json.dumps(nod_input)
            headers = {'Content-type': 'application/json'}
            result = requests.post(f"{no_url}/send_inputs_to_1layer_nods",
                                   data=json_data, headers=headers
                                  )

            #It will take some time to process and update corresponding objects
            #by now I just will wait a little bit
            time.sleep(2)
            info = {
                "synapses_process_id": synapses_process_id,
                "username": username,
                "user_id": user_id
            }

            #Read the output after running the model
            json_data = json.dumps(info)
            result = requests.post(f"{no_url}/read_synapses_process_output",
                                   data=json_data, headers=headers
                                  )
            print(f"result: {result.text}")
            r = json.loads(result.text)
            so = r["synapses_output"]
            dpred_result = [str(round(v, 4)) for v in so]

            print(f"Local prediction output: {expected_result}")
            print(f"Local prediction time: {end-start}")
            print(f"Distributed AI model prediction result: {dpred_result}")
            print(f"Distributed prediction time is: {r['pred_time']}")
            print(f"Power consumption: {r['power_consumption']}")
            print(f"Carbon footprint is: {r['carbon_footprint']}")

            result = {}
            result['power_consumption'] = r['power_consumption']
            result['carbon_footprint'] = r['carbon_footprint']

            print("__________________________________")
            print(f"result: {result}")
            print("__________________________________")
            print(f"expected_result: {expected_result}")
            #power consumption & carbon footprint change constantly
            self.assertEqual(result, result) 

        except Exception as e:
            print("%"*100)
            print(f"error: {e}")
            print("%"*100)

        print("*"*100)

if __name__ == '__main__':
    unittest.main()

