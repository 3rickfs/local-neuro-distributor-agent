import json

import boto3
from orchestration_planner import OrchPlannerOps
from neuron_distributor import (start_distribution,
                                start_first_layer_input_distribution
                               )
from ModelOnboarder import ModelOnboardingOps

class synapses_process():
    def __init__(self, nods_tech_info={}):
        self.mem_adr = 0
        self.synapses_output = []
        self.synapses_processes = {}
        self.synapses_process_id = 0 #cloud id
        self.synapses_process_code = 0 #local id
        self.fleps = []
        self.pred_start_time = 0
        self.pred_end_time = 0
        self.prediction_time = 0
        self.nods_tech_info = nods_tech_info
        self.nd_urls = []
        self.carbon_footprint = 0
        self.model_power_consumption = 0

        self.owner_id = {}
        self.model_details = {}
        self.num_nods = 0
        self.dataset_name = ""
        self.dataset_url = ""
        self.notebook_url = ""
        self.obj_cloud_path = ""
        self.obj_local_path = ""
        self.aimodel_cloud_path = ""
        self.aimodel_local_path = ""
        self.aimodel_file_name = ""
        self.no_output_ep = ""

    def delete_synproc_aimodel_s3_objs(self):
        try:
            s3_resource = boto3.resource(
                's3',
                region_name = self.region,
                aws_access_key_id = self.access_key_id,
                aws_secret_access_key = self.secret_access_key
            )
            s3_resource.Object(self.bucket_name, self.obj_cloud_path).delete()
            s3_resource.Object(self.bucket_name, self.aimodel_cloud_path).delete()
        except Exception as e:
            raise Exception(f"Failed to delete synproc and or aimodel objs: {e}")


    def reload_synaptic_process(self, sp_data):
        print("Reloading synpatic process")
        self.fleps = sp_data["fleps"]
        self.nods_tech_info = sp_data["nods_tech_info"]
        self.owner_id = sp_data["owner_id"]
        self.model_details = sp_data["model_details"]
        self.num_nods = sp_data["num_nods"]
        self.dataset_name = sp_data["dataset_name"]
        self.dataset_url = sp_data["dataset_url"]
        self.notebook_url = sp_data["notebook_url"]
        self.obj_cloud_path = sp_data["obj_cloud_path"]
        self.obj_local_path = sp_data["obj_local_path"]
        self.aimodel_cloud_path = sp_data["aimodel_cloud_path"]
        self.aimodel_local_path = sp_data["aimodel_local_path"]
        self.no_output_ep = sp_data["no_output_ep"]
        self.synapses_output = sp_data["synapses_output"]
        self.pred_start_time = sp_data["pred_start_time"]
        self.pred_end_time = sp_data["pred_end_time"]
        self.prediction_time = sp_data["prediction_time"]
        self.nods_tech_info = sp_data["nods_tech_info"]
        self.model_power_consumption = sp_data["model_power_consumption"]
        self.carbon_footprint = sp_data["carbon_footprint"]

    def upload_aimodel_json_to_cloud(self):
        try:
            self.upload_file_to_cloud(self.aimodel_cloud_path, self.aimodel_local_path)
        except Exception as e:
            raise Exception(f"Failed to upload ai model to cloud: {e}")

    def upload_obj_json_to_cloud(self):
        try:
            self.upload_file_to_cloud(self.obj_cloud_path, self.obj_local_path)
        except Exception as e:
            raise Exception(f"Failed to upload obj to cloud: {e}")

    def upload_file_to_cloud(self, cloud_file_path, local_file_path):
        s3_resource = boto3.resource(
            's3',
            region_name = self.region,
            aws_access_key_id = self.access_key_id,
            aws_secret_access_key = self.secret_access_key
        )

        with open(local_file_path, 'rb') as of:
            s3_resource.Bucket(self.bucket_name).put_object(
                #Bucket = self.bucket_name,
                Key = cloud_file_path,
                Body = of
            )
        of.close()

    def save_aimodel_local(self):
        with open(self.aimodel_local_path, "w") as jf:
            json.dump(self.model_details, jf)
        jf.close()

    def export_obj_as_json(self):
        sp_data = {}
        sp_data["fleps"] = self.fleps
        sp_data["nods_tech_info"] = self.nods_tech_info
        sp_data["owner_id"] = self.owner_id
        # to get the obj lighter in local
        sp_data["model_details"] = {} #self.model_details
        sp_data["num_nods"] = self.num_nods
        sp_data["dataset_name"] = self.dataset_name
        sp_data["dataset_url"] = self.dataset_url
        sp_data["notebook_url"] = self.notebook_url
        sp_data["obj_cloud_path"] = self.obj_cloud_path
        sp_data["obj_local_path"] = self.obj_local_path
        sp_data["aimodel_cloud_path"] = self.aimodel_cloud_path
        sp_data["aimodel_local_path"] = self.aimodel_local_path
        sp_data["no_output_ep"] = self.no_output_ep
        sp_data["synapses_output"] = self.synapses_output
        sp_data["pred_start_time"] = self.pred_start_time
        sp_data["pred_end_time"] = self.pred_end_time
        sp_data["prediction_time"] = self.prediction_time
        sp_data["nods_tech_info"] = self.nods_tech_info
        sp_data["model_power_consumption"] = self.model_power_consumption
        sp_data["carbon_footprint"] = self.carbon_footprint

        with open(self.obj_local_path, "w") as of:
            json.dump(sp_data, of)
        of.close()

    def get_nods_number(self):
        nods_num = 0
        try:
            for n in self.nods_tech_info:
                null_nod_dist = [[0,0,0]]
                if null_nod_dist != self.nods_tech_info[n]["neuron_dist"]:
                    nods_num += 1
        except Exception as e:
            raise Exception(f"Bad nods tech info format: {e}")

        return nods_num

    def get_fleps(self, nod_info):
        print("Getting first layer endpoints")
        fleps = []
        #print(f"nod info: {nod_info}")
        for nodi in nod_info:
            cid = nod_info[nodi]["capa_ids"][0]
            if cid == 1: #first layer only
                print(f"nod first layer endpoint: {(nod_info[nodi]['ops_ep'])}")
                fleps.append(nod_info[nodi]["ops_ep"])
            else:
                break

        return fleps

    #def onboard_model(self, spid, noep, user_id, username, mj, spc_path,
    #                  spl_path, ds_name, ds_url, nb_url):
    #    res = ModelOnboardingOps.run(sp = self,
    #                                 spid = spid,
    #                                 no_ep = noep,
    #                                 user_id = user_id,
    #                                 username = username,
    #                                 model_json = mj,
    #                                 synproc_local_path = spl_path,
    #                                 synproc_cloud_path = spc_path,
    #                                 dataset_name = ds_name,
    #                                 dataset_url = ds_url,
    #                                 notebook_url = nb_url
    #                                )
    def onboard_model(self, **kwargs):
        kwargs["sp"] = self
        res = ModelOnboardingOps.run(**kwargs)

        return res

    def calculate_pred_power_consumption(self):
        nods_num = self.get_nods_number() #len(self.nods_tech_info)
        if nods_num != 0:
            self.model_power_consumption = 0
            for nod_idx in range(1, nods_num + 1):
                nodti_d = self.nods_tech_info["nod_"+str(nod_idx)]
                pc = nodti_d["power_consumption"]
                pt = self.prediction_time
                mpc = (pc / 1000.0) * pt / 3600.0
                self.model_power_consumption += mpc
        else:
            return 0

        return self.model_power_consumption

    def calculate_carbon_footprint(self):
        nods_num = self.get_nods_number() #len(self.nods_tech_info)
        if nods_num != 0:
            self.carbon_footprint = 0
            pt = self.prediction_time / nods_num #just to simplify
            for nod_idx in range(1, nods_num + 1):
                nodti_d = self.nods_tech_info["nod_"+str(nod_idx)]
                pc = nodti_d["power_consumption"]
                pue = nodti_d["pue"]
                gco2pkwh = nodti_d["gco2pkwh"]
                manufacco2 = nodti_d["manufacco2"]
                cf = ((pc * pue * gco2pkwh / 1000.0) + manufacco2) * pt / 3600.0
                self.carbon_footprint += cf
        else:
            return 0

        return self.carbon_footprint

    def get_nod_tech_info(self):
        return self.nods_tech_info

    def calculate_prediction_time(self):
        print(self.pred_end_time)
        print(self.pred_start_time)
        self.prediction_time = self.pred_end_time - self.pred_start_time

    def get_prediction_time(self):
        return self.prediction_time

    def set_pred_start_time(self, st):
        print(f"setting pred start time: {st}")
        self.pred_start_time = st

    def set_pred_end_time(self, et):
        print(f"setting pred end time: {et}")
        self.pred_end_time = et

    def get_pred_start_time(self):
        return self.pred_start_time

    def get_pred_end_time(self):
        return self.pred_end_time

    def save_fleps(self, fleps):
        self.fleps = fleps

    def read_fleps(self):
        return self.fleps

    def read_synapses_output(self):
        print("Reading synpases output")
        return self.synapses_output

    def save_object_memory_address(self, cloud_sp_id, local_sp_id, jfp):
        #jfp: json file path
        print("Saving object memory address into a dictionary")
        self.synapses_processes = self.read_sp_file(jfp, "synapses_processes.json")
        self.synapses_processes[str(cloud_sp_id)] = local_sp_id
        self.synapses_process_code = local_sp_id
        self.write_sp_file(jfp, "synapses_processes.json", self.synapses_processes)

        try:
            psps = self.read_sp_file(jfp, "persistent_synapses_processes.json")
        except:
            print("Persistent synapses processes json file is going to be created")
            psps = {}
        psps[str(cloud_sp_id)] = local_sp_id
        self.write_sp_file(jfp, "persistent_synapses_processes.json", psps)

    def write_sp_file(self, jfp, filename, sp):
        with open(jfp + "/" + filename, "w") as jf:
            json.dump(sp, jf)
        jf.close()

    def read_sp_file(self, jfp, filename):
        with open(jfp + "/" + filename, "r") as jf:
            synapses_processes = json.load(jf)
        jf.close()

        return synapses_processes

    def set_mem_adr(self, mem_adr):
        self.mem_adr = mem_adr

    def set_synapses_output(self, syn_output):
        self.synapses_output = syn_output

