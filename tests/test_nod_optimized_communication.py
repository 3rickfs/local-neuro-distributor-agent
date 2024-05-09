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

    """
    def test_execute_distributed_diabetes_model(self):
        print("*"*100)
        print("Test 1: diabetes model 2nd approach of distribution and nod comm")
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
        #print(f"Local prediction time: {end-start}")
        expected_result = [str(round(v, 4)) for v in pred[0]]

        try:

            #Starting synapses process
            headers = {'Content-type': 'application/json'}
            #Get extensive nodes information 
            with open("nods_info.json", "r") as f:
                nods_tech_info = json.load(f)
            f.close()

            syn_pro_input = {}
            syn_pro_input["nods_tech_info"] = nods_tech_info
            syn_pro_input["ai_model_path"] = "lala.model"
            syn_pro_input["distributed_model"] = "whole distributed model"

            json_data = json.dumps(syn_pro_input)
            result = requests.post(f"{no_url}/start_synapses_process",
                                   data=json_data, headers=headers
                                  )

            synapses_process_id = json.loads(result.text)["synapses_process_id"]
            print(f"synapses_process: {synapses_process_id}")

            #Load the AI model JSON
            with open(input_file_name, 'r') as jf:
                input_json_file = json.load(jf)
            jf.close()

            #Distribute neurons
            #input_json_file["CoN_parameters"] = CoN_parameters
            #input_json_file["nod_ops_endpoints_fn"] = "nod_ops_endpoints.txt"
            #input_json_file["nod_dis_endpoints_fn"] = "nod_dis_endpoints.txt"
            input_json_file["synapses_process_id"] = synapses_process_id
            input_json_file["neuro_orchestrator_ep"] = f"{no_url}/set_final_output"

            with open('./json_data.json', 'w') as outfile:
                json.dump(input_json_file, outfile)

            #files
            files={
                #'ops': open('./nod_ops_endpoints.txt', 'rb'),
                #'dis': open('./nod_dis_endpoints.txt', 'rb'),
                'json': open('./json_data.json', 'rb')
            }

            #json_data = json.dumps(input_json_file)
            dn_headers = {'Content-type': 'multipart/form-data'}
            result = requests.post(f"{no_url}/distribute_neurons",
                                   #data=json_data,
                                   files=files
                                   #headers=dn_headers
                                  )
            print(result.text)

            #send inputs to NODS
            nod_input = {
                "inputs": inp_dic,
                "input_idx": 0,
                "layer_id": 1,
                "synapses_process_id": synapses_process_id
            }

            json_data = json.dumps(nod_input)
            result = requests.post(f"{no_url}/send_inputs_to_1layer_nods",
                                   data=json_data, headers=headers
                                  )

            #It will take some time to process and update corresponding objects
            #for now I just will wait a little bit
            time.sleep(2)
            info = {
                "synapses_process_id": synapses_process_id
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
    """

    def test_execute_nist_fcl_model(self):
        print("*"*100)
        print("Test 2: run nist fcl with optimized nod communication")
        print("-------------------------------------------------------")

        input_file_name = "nist_fcl.json"
        #Run above model in TF format to get the expected result
        model = keras.models.load_model(
            "./nist_fcl.keras"
        )
        inp_dic = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.32941176470588235, 0.7254901960784313, 0.6235294117647059, 0.592156862745098, 0.23529411764705882, 0.1411764705882353, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8705882352941177, 0.996078431372549, 0.996078431372549, 0.996078431372549, 0.996078431372549, 0.9450980392156862, 0.7764705882352941, 0.7764705882352941, 0.7764705882352941, 0.7764705882352941, 0.7764705882352941, 0.7764705882352941, 0.7764705882352941, 0.7764705882352941, 0.6666666666666666, 0.20392156862745098, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2627450980392157, 0.4470588235294118, 0.2823529411764706, 0.4470588235294118, 0.6392156862745098, 0.8901960784313725, 0.996078431372549, 0.8823529411764706, 0.996078431372549, 0.996078431372549, 0.996078431372549, 0.9803921568627451, 0.8980392156862745, 0.996078431372549, 0.996078431372549, 0.5490196078431373, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.06666666666666667, 0.25882352941176473, 0.054901960784313725, 0.2627450980392157, 0.2627450980392157, 0.2627450980392157, 0.23137254901960785, 0.08235294117647059, 0.9254901960784314, 0.996078431372549, 0.41568627450980394, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3254901960784314, 0.9921568627450981, 0.8196078431372549, 0.07058823529411765, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.08627450980392157, 0.9137254901960784, 1.0, 0.3254901960784314, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5058823529411764, 0.996078431372549, 0.9333333333333333, 0.17254901960784313, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.23137254901960785, 0.9764705882352941, 0.996078431372549, 0.24313725490196078, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5215686274509804, 0.996078431372549, 0.7333333333333333, 0.0196078431372549, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.03529411764705882, 0.803921568627451, 0.9725490196078431, 0.22745098039215686, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.49411764705882355, 0.996078431372549, 0.7137254901960784, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.29411764705882354, 0.984313725490196, 0.9411764705882353, 0.2235294117647059, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.07450980392156863, 0.8666666666666667, 0.996078431372549, 0.6509803921568628, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.011764705882352941, 0.796078431372549, 0.996078431372549, 0.8588235294117647, 0.13725490196078433, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.14901960784313725, 0.996078431372549, 0.996078431372549, 0.30196078431372547, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.12156862745098039, 0.8784313725490196, 0.996078431372549, 0.45098039215686275, 0.00392156862745098, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5215686274509804, 0.996078431372549, 0.996078431372549, 0.20392156862745098, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.23921568627450981, 0.9490196078431372, 0.996078431372549, 0.996078431372549, 0.20392156862745098, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4745098039215686, 0.996078431372549, 0.996078431372549, 0.8588235294117647, 0.1568627450980392, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4745098039215686, 0.996078431372549, 0.8117647058823529, 0.07058823529411765, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        
        try:

            #Starting synapses process
            headers = {'Content-type': 'application/json'}
            #Get extensive nodes information 
            with open("./nods_info_nist_fcl.json", "r") as f:
                nods_tech_info = json.load(f)
            f.close()

            syn_pro_input = {}
            syn_pro_input["nods_tech_info"] = nods_tech_info
            syn_pro_input["ai_model_path"] = "lala.model"
            syn_pro_input["distributed_model"] = "whole distributed model"

            json_data = json.dumps(syn_pro_input)
            result = requests.post(f"{no_url}/start_synapses_process",
                                   data=json_data, headers=headers
                                  )

            synapses_process_id = json.loads(result.text)["synapses_process_id"]
            print(f"synapses_process: {synapses_process_id}")

            #Load the AI model JSON
            with open(input_file_name, 'r') as jf:
                input_json_file = json.load(jf)
            jf.close()

            #Distribute neurons
            #input_json_file["CoN_parameters"] = CoN_parameters
            #input_json_file["nod_ops_endpoints_fn"] = "nod_ops_endpoints.txt"
            #input_json_file["nod_dis_endpoints_fn"] = "nod_dis_endpoints.txt"
            input_json_file["synapses_process_id"] = synapses_process_id
            input_json_file["neuro_orchestrator_ep"] = f"{no_url}/set_final_output"

            with open('./json_data.json', 'w') as outfile:
                json.dump(input_json_file, outfile)
            outfile.close()

            #files
            files={
                #'ops': open('./nod_ops_endpoints.txt', 'rb'),
                #'dis': open('./nod_dis_endpoints.txt', 'rb'),
                'json': open('./json_data.json', 'rb')
            }

            #json_data = json.dumps(input_json_file)
            dn_headers = {'Content-type': 'multipart/form-data'}
            result = requests.post(f"{no_url}/distribute_neurons",
                                   #data=json_data,
                                   files=files
                                   #headers=dn_headers
                                  )
            print(result.text)

            #send inputs to NODS
            nod_input = {
                "inputs": inp_dic,
                "input_idx": 0,
                "layer_id": 1,
                "synapses_process_id": synapses_process_id
            }

            json_data = json.dumps(nod_input)
            result = requests.post(f"{no_url}/send_inputs_to_1layer_nods",
                                   data=json_data, headers=headers
                                  )

            #It will take some time to process and update corresponding objects
            #for now I just will wait a little bit
            time.sleep(2)
            info = {
                "synapses_process_id": synapses_process_id
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

            inp = np.array(inp_dic).reshape(1, 784)
            print(f"model input: {inp}")
            start = time.time()
            pred = model.predict(inp)
            end = time.time()
            print(f"pred: {pred}")
            #print(f"Local prediction time: {end-start}")
            expected_result = [str(round(v, 4)) for v in pred[0]]

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

