import os
import json
import requests
from abc import ABC, abstractmethod
from datetime import datetime

from orchestration_planner import (#read_endpoints,
                                   OrchPlannerOps,
                                   save_files,
                                   read_json_data
                                  )
from neuron_distributor import (start_distribution,
                                start_first_layer_input_distribution
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
        json_data = {
            "user_id": kwargs["user_id"],
            "username": kwargs["username"],
            "neuro_orchestrator_url": kwargs["neuro_orchestrator_url"],
            "dataset_name": kwargs["dataset_name"],
            "dataset_url": kwargs["dataset_url"],
            "notebook_url": kwargs["notebook_url"],
            "mj": mj
        }

        with open("json_data.json", "w") as jf:
            json.dump(json_data, jf)
        jf.close()
        with open("nods_info.json", "w") as jf:
            json.dump(kwargs["nods_info"], jf)
        jf.close()
        #with open("json_data.json", "rb") as jf:
        #    json_data = jf
        #jf.close()
        #with open("nods_info.json", "rb") as jf:
        #    nods_info= jf
        #jf.close()
        json_data = open("json_data.json", "rb")
        nods_info = open("nods_info.json", "rb")

        files = {
            'nods_info': nods_info,
            'json_data': json_data
        }

        noa_url = kwargs["neuro_orchestrator_url"]
        url = noa_url + "/crear_no_distribution_synaptic_process"
        result = requests.post(url, files=files)

        res = json.loads(result.text)
        print(res)

        json_data.close()
        nods_info.close()

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

class save_nod_dis_urls(model_onboarding_ops):
    """ Save nod distribution endpoints 
    """

    def run_operation(**kwargs):
        print("Saving nod distribution URLs")

        #nd_urls = [
        #    "http://" + e \
        #    for e in kwargs["nod_dict"]["dis_ep"].split("/")[2] \
        #    if kwargs["nod_dict"]["neuron_dist"] != [[0,0,0]]
        #]

        nd_urls = []
        for nod_idx in range(1, len(kwargs["nod_dict"]) + 1):
            ni = kwargs["nod_dict"]["nod_" + str(nod_idx)]
        #   if ni["neuron_dist"] != [[0,0,0]]:
            url = "http://" + ni["dis_ep"].split("/")[2]
            nd_urls.append(url)
        print(f"nd_urls: {nd_urls}")

        kwargs["sp"].nd_urls = nd_urls

        return kwargs


class distribute_neurons_to_nods(model_onboarding_ops):
    """ Distribute neurons to corresponding NODs according to the nod info file
    """

    def run_operation(**kwargs):
        print("Distributing neurons to NODs")
        #Getting first layer endpoints and save them into synapses process obj
        fl_eps = kwargs["sp"].get_fleps(kwargs["nod_dict"])
        kwargs["sp"].save_fleps(fl_eps)
        #Distribution of neurons
        try:
            nod_res = start_distribution(kwargs["nod_dict"],
                                         kwargs["proc_sinap_id"], #kwargs["spid"]
                                         kwargs["mfname"]
                                        )
        except Exception as e:
            raise Exception(f"Error during distribution neurons to NODs: {e}")

        kwargs["dist_nod_response"] = nod_res

        return kwargs

class save_fleps_in_noa_sp(model_onboarding_ops):
    """ save fleps in sp record in NOA
    """

    def run_operation(**kwargs):
        print("Save fleps in sp record in corresponding NOA")
        input_data = {
            "fleps": kwargs["sp"].read_fleps(),
            "user_id": kwargs["user_id"],
            "username": kwargs["username"],
            "synapses_process_id": kwargs["proc_sinap_id"]
        }

        headers = {'Content-type': 'application/json'}
        json_data = json.dumps(input_data)
        no_url = kwargs["neuro_orchestrator_url"]
        print(f"no url: {no_url}")
        result = requests.post(f"{no_url}/save_sp_fleps",
                               data=json_data, headers=headers
                              )

        result = json.loads(result.text)
        print(result)
        if result["result"] != "ok":
            raise Exception(f"Error when saving fleps: {result}")

        return kwargs

class save_files_local_and_cloud(model_onboarding_ops):
    """ save synproc obj and the ai model in the local machine and cloud 
    """

    def run_operation(**kwargs):
        print("Saving in local and remote the syn proc obj and ai model")
        #kwargs["sp"].export_obj_as_json() #save the obj as json in local machine
        #kwargs["sp"].upload_obj_json_to_cloud()
        #kwargs["sp"].save_aimodel_local() # as json
        #kwargs["sp"].upload_aimodel_json_to_cloud()

        return kwargs

class output_msg(model_onboarding_ops):
    """ create an output message as a successful result
    """

    def run_operation(**kwargs):
        print("Create output message")
        kwargs["output_msg"] = {
            "proc_sinap_id": kwargs["proc_sinap_id"],
            "res": "successful",
            "fleps_eps": kwargs["sp"].read_fleps()
        }
        print(f"Output message: {kwargs['output_msg']}")

        return kwargs

class ModelOnboardingOps:

    @staticmethod
    def run(**kwargs):
        for operation in model_onboarding_ops.__subclasses__():
            kwargs = operation.run_operation(**kwargs)

        return kwargs["output_msg"]#, kwargs["sp"].read_fleps()

