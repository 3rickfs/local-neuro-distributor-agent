import json

model_details_dict = {
    "nombre_modelo": "ERP almacenes EMTRAFESA",
    "numero_neuronas": 340,
    "numero_parametros": 1000,
    "numero_nods": 4,
    "numero_capas": 20,
    "id_proceso_sinaptico": 123456,
    "version_modelo": "2.3.1",
    "huella_carbono_gco2eq_total": 0.01,
    "eficiencia_neuronas_per_w": 200,
    "consumo_energia_kw_per_prediccion": 0.0001,
    "fecha_creacion_proceso_sinaptico": "15/10/24",
    "id_usuario": 4531343,
    "id_owner": 123451,
    "id_proceso_sinaptico": 123456,
    "url_inputs_proceso_sinaptico": "http://123.321.21.1:5000/send_inputs",
    "notebook_url": "https://colab.research.google.com/drive/1WE0Pr7r_KAGeaLHtsyeiyBV5LTEJ76q7?usp=sharing",
    "dataset_name": "Objetos de almacen",
    "dataset_url": "https://universe.roboflow.com/project-xpy6q/plant_disease-db7ns",
    "historial": {
        "pred_1": {
            "fecha": "28/09/2024",
            "hora": "13:25",
            "huella_carbono_gco2eq": 0.001,
            "consumo_energia_w_per_prediccion": 0.000001,
            "tiempo_prediccion": 0.01,
            "tiempo_delay": 0.004
        },
        "pred_2": {
            "fecha": "28/09/2024",
            "hora": "17:11",
            "huella_carbono_gco2eq": 0.0019,
            "consumo_energia_w_per_prediccion": 0.000004,
            "tiempo_prediccion": 0.01,
            "tiempo_delay": 0.004
        },
        "pred_3": {
            "fecha": "28/09/2024",
            "hora": "18:34",
            "huella_carbono_gco2eq": 0.002,
            "consumo_energia_w_per_prediccion": 0.000011,
            "tiempo_prediccion": 0.01,
            "tiempo_delay": 0.004
        },
        "pred_4": {
            "fecha": "28/09/2024",
            "hora": "18:55",
            "huella_carbono_gco2eq": 0.0013,
            "consumo_energia_w_per_prediccion": 0.000006,
            "tiempo_prediccion": 0.01,
            "tiempo_delay": 0.004
        },
        "pred_5": {
            "fecha": "28/09/2024",
            "hora": "19:13",
            "huella_carbono_gco2eq": 0.0016,
            "consumo_energia_w_per_prediccion": 0.000003,
            "tiempo_prediccion": 0.01,
            "tiempo_delay": 0.004
        },
        "pred_6": {
            "fecha": "28/09/2024",
            "hora": "20:30",
            "huella_carbono_gco2eq": 0.003,
            "consumo_energia_w_per_prediccion": 0.000004,
            "tiempo_prediccion": 0.01,
            "tiempo_delay": 0.004
        },
        "pred_7": {
            "fecha": "28/09/2024",
            "hora": "21:14",
            "huella_carbono_gco2eq": 0.0011,
            "consumo_energia_w_per_prediccion": 0.000007,
            "tiempo_prediccion": 0.01,
            "tiempo_delay": 0.004
        },
        "pred_8": {
            "fecha": "28/09/2024",
            "hora": "22:25",
            "huella_carbono_gco2eq": 0.001,
            "consumo_energia_w_per_prediccion": 0.000008,
            "tiempo_prediccion": 0.01,
            "tiempo_delay": 0.004
        }
    }
}

with open("model_details.json", "w") as f:
    json.dump(model_details_dict, f)
f.close()

json_obj = json.dumps(model_details_dict, indent=4)
print(json_obj)
