import os
import json
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
from noaDBmanager import noaDBmanager

class model_onboarding_ops(ABC):
    """ Operations to perform to onboard a AI model
    """

    @abstractmethod
    def run_operation(**kwargs):
        #interface to child classes
        pass

class verify_user_availability(model_onboarding_ops):
    """ see if the user exists or is available to onboard models
    """

    def run_operation(**kwargs):
        print("Veriying user availability")
        if noaDBmanager.verify_user_compliance(
            kwargs["user_id"],
            kwargs["username"]
        ):
            #if yes then it exists
            print("User exists and meets the compliance")
        else:
            raise Exception("User does not exist or not"
                            "meet the compliance"
                           )

        return kwargs

class get_model_json_ready(model_onboarding_ops):
    """ Get the model in json ready for being distributed
    """ 

    def run_operation(**kwargs):
        print("Getting the model json ready")
        #print(kwargs["mj"])

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
        kwargs["sp"].obj_local_path = kwargs["upload_folder_path"] + \
                                      "/sps/" + \
                                      objfn
        mfname =  kwargs["sp"].model_details["model_info"]["nombre"]
        mfname = mfname.replace(" ", "_") # replacing blank spaces by _
        mfname += "-" + str(kwargs["spcode"]) + "-aimodel.json"
        kwargs["mfname"] = mfname
        kwargs["sp"].aimodel_file_name = mfname

        kwargs["sp"].aimodel_cloud_path = kwargs["mfpc"] + "/" + mfname
        kwargs["sp"].aimodel_local_path = kwargs["upload_folder_path"] + \
                                          "/models/" + mfname

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


class save_distribution_info_to_db(model_onboarding_ops):
    """ Being ok previous steps it's time to save in db what was done
    """

    def run_operation(**kwargs):
        print("Saving distribution info to db")
        # Adding an ai model to db
        aimodel_data = {}
        aimodel_data["owner_id"] = kwargs["user_id"]
        aimodel_data["model_path_cloud"] = kwargs["sp"].aimodel_cloud_path
        aimodel_data["model_bucket_name"] = kwargs["model_bucket_name"]
        aimodel_data["model_path_local"] = kwargs["sp"].aimodel_local_path
        aimodel_data["creation_datetime"] = datetime.now()
        aimodel_data["lastmodification_datetime"] = datetime.now()
        aimodel_data["model_name"] = kwargs["sp"].\
                                     model_details["model_info"]["nombre"]
        aimodel_data["model_file_name"] = kwargs["mfname"]
        aimodel_data["model_version"] = kwargs["sp"].\
                                        model_details["model_info"]\
                                        ["model_version"]
        aimodel_data["neurons_number"] = kwargs["sp"].\
                                         model_details["model_info"]\
                                         ["neurons_num"]
        aimodel_data["layers_number"] = kwargs["sp"].\
                                        model_details["model_info"]\
                                        ["layers_num"]
        aimodel_data["parameters_number"] = kwargs["sp"].\
                                            model_details["model_info"]\
                                            ["params_num"]

        aimodel_id = noaDBmanager.insert_aimodel_data(aimodel_data)
        kwargs["aimodel_id"] = aimodel_id
        print(f"New ai model id: {aimodel_id}")

        #if res["res"] != "ok"
        #    raise Exception("Issues to insert new row in aimodel table")

        # Get the last created model
        #aimodelid = res["aimodel_id"] #noaDBmanager.get_last_aimodel_id()
        # Create synapses process
        sp_data = {}
        sp_data["user_id"] = kwargs["user_id"]
        sp_data["aimodel_id"] = aimodel_id
        sp_data["proc_sinap_code"] = kwargs["spcode"]
        sp_data["creation_datetime"] = datetime.now()
        sp_data["last_modification_datetime"] = sp_data["creation_datetime"]
        sp_data["nods_num"] = kwargs["sp"].get_nods_number()
        sp_data["dataset_name"] = kwargs["dataset_name"]
        sp_data["dataset_url"] = kwargs["dataset_url"]
        sp_data["obj_cloud_path"] = kwargs["sp"].obj_cloud_path
        sp_data["obj_local_path"] = kwargs["sp"].obj_local_path
        sp_data["notebook_url"] = kwargs["notebook_url"]

        cloud_proc_sinap_id = noaDBmanager.insert_synapses_process(sp_data)
        kwargs["proc_sinap_id"] = cloud_proc_sinap_id
        print(f"New synaptic process created: {cloud_proc_sinap_id}")
        #Update synaptic process parameters
        kwargs["sp"].save_object_memory_address(
            cloud_proc_sinap_id, #cloud synaptic process id
            kwargs["spcode"], #local synaptic process id
            kwargs["upload_folder_path"]
        )
        #if res["res"] != "ok"
        #    raise Exception("Issues to insert new row in synproc table")

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

class save_files_local_and_cloud(model_onboarding_ops):
    """ save synproc obj and the ai model in the local machine and cloud 
    """

    def run_operation(**kwargs):
        print("Saving in local and remote the syn proc obj and ai model")
        kwargs["sp"].export_obj_as_json() #save the obj as json in local machine
        kwargs["sp"].upload_obj_json_to_cloud()
        kwargs["sp"].save_aimodel_local() # as json
        kwargs["sp"].upload_aimodel_json_to_cloud()

        return kwargs

class output_msg(model_onboarding_ops):
    """ create an output message as a successful result
    """

    def run_operation(**kwargs):
        print("Create output message")
        kwargs["output_msg"] = {
            "proc_sinap_id": kwargs["proc_sinap_id"],
            "aimodel_id": kwargs["aimodel_id"],
            "res": "successful"
        }

        return kwargs

class ModelOnboardingOps:

    @staticmethod
    def run(**kwargs):
        for operation in model_onboarding_ops.__subclasses__():
            kwargs = operation.run_operation(**kwargs)

        return kwargs["output_msg"]

