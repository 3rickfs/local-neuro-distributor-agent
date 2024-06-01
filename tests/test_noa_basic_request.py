import os
import sys
import time
import json
import unittest
import requests

pd = os.path.abspath('..')
sys.path.append(pd)
from noa import crear_proceso_sinaptico, getJSONfromKerasModel

#url neuro orchestrator
#no_url = "http://127.0.0.1:5000"
#no_url = "http://a9d4da522d410433185d883825694e22-1486249831.us-west-1.elb.amazonaws.com:5000"
no_url = "http://a850b7340f8cb4840ab5ed5165b87da9-809592125.us-west-1.elb.amazonaws.com:5000"

class synaptic_process_creations(unittest.TestCase):
    def test_connect_to_noa(self):
        print("*"*100)
        print("Test 1: is noa alive?")
        print("-------------------------------------------------------")


        try:
            data = {}
            data["uno"] = 1
            data["dos"] = 2
            data["tres"] = 3
            json_data = json.dumps(data)
            headers = {'Content-type': 'application/json'} 
            #result = requests.get(f"{no_url}/")
            result = requests.post(f"{no_url}/dummy_endpoint",
                                   data=json_data,
                                   headers=headers
                                  )
            print(result.text)
            result = result.text

            expected_res = '{"uno": 1, "dos": 2, "tres": 3}'
            self.assertEqual(result, expected_res)

        except Exception as e:
            print("%"*100)
            print(f"error: {e}")
            print("%"*100)


        print("*"*100)

if __name__ == '__main__':
    unittest.main()

