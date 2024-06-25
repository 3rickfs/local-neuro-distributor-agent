import os
import json
import requests
from abc import ABC, abstractmethod
from datetime import datetime

from orchestration_planner import (#read_endpoints,
                                   OrchPlannerOps
                                  )

class model_onboarding_ops(ABC):
    """ Operations to perform to onboard a AI model
    """

    @abstractmethod
    def run_operation(**kwargs):
        #interface to child classes
        pass

class create_synaptic_process(model_onboarding_ops):
    """ Create synaptic process calling the corresponding NOA to do that
    """

    def run_operation(**kwargs):
        print("Creating synaptic process")
        mj = {"model_info":{
            "nombre": kwargs["mj"]["model_info"]["nombre"],
            "model_version": kwargs["mj"]["model_info"]["model_version"],
            "neurons_num": kwargs["mj"]["model_info"]["neurons_num"],
            "layers_num": kwargs["mj"]["model_info"]["layers_num"],
            "params_num": kwargs["mj"]["model_info"]["params_num"]
            }
        }
        ni = kwargs["nods_info"]
        json_data = {
            "user_id": kwargs["user_id"],
            "username": kwargs["username"],
            "neuro_orchestrator_url": kwargs["neuro_orchestrator_url"],
            "dataset_name": kwargs["dataset_name"],
            "dataset_url": kwargs["dataset_url"],
            "notebook_url": kwargs["notebook_url"],
            "nombre": mj["model_info"]["nombre"],
            "model_version": mj["model_info"]["model_version"],
            "neurons_num": mj["model_info"]["neurons_num"],
            "layers_num": mj["model_info"]["layers_num"],
            "params_num": mj["model_info"]["params_num"],
            "nods_info": ni
            #"mj": mj
        }
        print("-----------------------------------------------------")

        #headers = {'Content-type': 'application/json'}
        #noa_url = kwargs["neuro_orchestrator_url"]
        #url = noa_url + "/crear_no_distribution_synaptic_process"
        #print(f"URL for no distribution sp creation: {url}")
        #result = requests.post(url, headers=headers, files=files)
        #json_data = json.dumps(json_data) #kwargs["nods_info"]) #json_data)
        #json_data = json.dumps({"uno":1, "dos":2})
        #print(json_data)
        #result = requests.post(url, data=json_data, headers=headers)
        #result = requests.post(url, files=files)


        #headers = {'Content-type': 'application/json'} 
        #result = requests.get(f"{no_url}/")
        #json_data = json.dumps(json_data)
        #result = requests.post(f"{noa_url}/dummy_endpoint",
        #                       data=json_data,
        #                       headers=headers
        #                      )
        #result = requests.post(f"{noa_url}/dummy_endpoint",
        #                       files=files,
        #                       headers=headers
        #                      )

        #res = json.loads(result.text)
        res["proc_sinap_id"] = 0
        print(res)

        #json_data.close()
        #nods_info.close()

        if res["res"] == "successful":
            spid = res["proc_sinap_id"]
            kwargs["proc_sinap_id"] = spid
            print(f"Correctly created synaptic process: {spid}")
        else:
            raise Exception(f"Issues with user: {kwargs['username']} for" + \
                            "creating the synaptic process")

        return kwargs

class update_syn_proc_info(model_onboarding_ops):
    """ update info in corresponing syn_proc obj
    """

    def run_operation(**kwargs):
        print("Updating syn_proc obj info")
        #synaptic process
        kwargs["sp"].synapses_process_code = kwargs["spcode"] #node synaptic process
        kwargs["sp"].owner_id = kwargs["user_id"]
        kwargs["sp"].model_details = kwargs["mj"]
        kwargs["sp"].num_nods = len(kwargs["sp"].nods_tech_info)
        kwargs["sp"].dataset_name = kwargs["dataset_name"]
        kwargs["sp"].dataset_url = kwargs["dataset_url"]
        kwargs["sp"].notebook_url = kwargs["notebook_url"]
        kwargs["sp"].no_output_ep = kwargs["noep"]

        return kwargs

class update_sp_and_model_paths(model_onboarding_ops):
    """ Update syn proc and model local and cloud paths using
    """

    def run_operation(**kwargs):
        print("Updating syn proc and model paths")
        objfn = str(kwargs["user_id"]) + \
                "-" + \
                str(kwargs["spcode"]) + \
                "-spobj.json"
        kwargs["sp"].obj_cloud_path = kwargs["sc_fpath"] + "/" + objfn
        # following path should be set up in the corresponding NOA
        kwargs["sp"].obj_local_path = ""
        mfname =  kwargs["sp"].model_details["model_info"]["nombre"]
        mfname = mfname.replace(" ", "_") # replacing blank spaces by _
        mfname += "-" + str(kwargs["spcode"]) + "-aimodel.json"
        kwargs["mfname"] = mfname
        kwargs["sp"].aimodel_file_name = mfname

        kwargs["sp"].aimodel_cloud_path = ""
        kwargs["sp"].aimodel_local_path = ""

        return kwargs

class plan_orchestration(model_onboarding_ops):
    """ Run the orchestration planner to know how to distribute neurons
    """

    def run_operation(**kwargs):
        print("Planning neuron orchestration")
        try:
            nod_dict = OrchPlannerOps.run(
                #nod_ep = nod_ops_ep,
                nods_tech_info = kwargs["sp"].nods_tech_info,
                neuro_orchestrator_ep = kwargs["sp"].no_output_ep,
                json_data = kwargs["mj"]
            )["nod_dict"]
        except Exception as e:
            raise Exception(f"Error during orchestration planning: {e}")

        kwargs["nod_dict"] = nod_dict

        return kwargs

class output_msg(model_onboarding_ops):
    """ create an output message as a successful result
    """

    def run_operation(**kwargs):
        print("Create output message")
        kwargs["output_msg"] = {
            "proc_sinap_id": kwargs["proc_sinap_id"],
            "res": "successful",
            "nod_dict": kwargs["nod_dict"]
        }
        print(f"Output message: {kwargs['output_msg']}")

        return kwargs

class LocalModelOnboardingOps:

    @staticmethod
    def run(**kwargs):
        for operation in model_onboarding_ops.__subclasses__():
            kwargs = operation.run_operation(**kwargs)

        return kwargs["output_msg"] #, kwargs["sp"].read_fleps()

