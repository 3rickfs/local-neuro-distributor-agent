import json

user_home_dict = {}
user_home_dict["modelo_1"] = {
    "nombre_modelo": "ERP almacenes EMTRAFESA",
    "numero_neuronas": 340,
    "numero_parametros": 1000,
    "numero_nods": 4,
    "numero_capas": 20,
    "id_proceso_sinaptico": 123456,
    "version_modelo": "2.3.1",
    "huella_carbono_gco2eq_total": 0.01,
    "eficiencia_neuronas_per_w": 200,
    "consumo_energia_w_per_prediccion": 0.0001
}

user_home_dict["modelo_2"] = {
    "nombre_modelo": "Predicciones contables 2024",
    "numero_neuronas": 23940,
    "numero_parametros": 1402000,
    "numero_nods": 5,
    "numero_capas": 25,
    "id_proceso_sinaptico": 136456,
    "version_modelo": "8.4.2",
    "huella_carbono_gco2eq_total": 0.03,
    "eficiencia_neuronas_per_w": 400,
    "consumo_energia_w_per_prediccion": 0.0002
}

user_home_dict["modelo_3"] = {
    "nombre_modelo": "Prueba chatbot",
    "numero_neuronas": 2533940,
    "numero_parametros": 19402000,
    "numero_nods": 7,
    "numero_capas": 45,
    "id_proceso_sinaptico": 536456,
    "version_modelo": "1.1.2",
    "huella_carbono_gco2eq_total": 0.05,
    "eficiencia_neuronas_per_w": 700,
    "consumo_energia_w_per_prediccion": 0.0002
}

user_home_dict["modelo_4"] = {
    "nombre_modelo": "Pronostico de ventas",
    "numero_neuronas": 53940,
    "numero_parametros": 402000,
    "numero_nods": 2,
    "numero_capas": 5,
    "id_proceso_sinaptico": 736456,
    "version_modelo": "7.1.2",
    "huella_carbono_gco2eq_total": 0.01,
    "eficiencia_neuronas_per_w": 200,
    "consumo_energia_w_per_prediccion": 0.0001
}

user_home_dict["modelo_5"] = {
    "nombre_modelo": "deteccion facial",
    "numero_neuronas": 23499,
    "numero_parametros": 4191473,
    "numero_nods": 4,
    "numero_capas": 6,
    "id_proceso_sinaptico": 936456,
    "version_modelo": "6.2.2",
    "huella_carbono_gco2eq_total": 0.05,
    "eficiencia_neuronas_per_w": 400,
    "consumo_energia_w_per_prediccion": 0.0005
}

user_home_dict["modelo_6"] = {
    "nombre_modelo": "test modelo satisfaccion del cliente",
    "numero_neuronas": 5499,
    "numero_parametros": 51473,
    "numero_nods": 2,
    "numero_capas": 4,
    "id_proceso_sinaptico": 116456,
    "version_modelo": "2.2.2",
    "huella_carbono_gco2eq_total": 0.01,
    "eficiencia_neuronas_per_w": 200,
    "consumo_energia_w_per_prediccion": 0.0002
}

with open("user_home.json", "w") as f:
    json.dump(user_home_dict, f)
f.close()
json_object = json.dumps(user_home_dict, indent = 4) 
print(json_object)





