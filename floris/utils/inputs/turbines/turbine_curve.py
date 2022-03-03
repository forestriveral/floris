import os, sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Horns(object):
    
    wind_pdf = np.array([[0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360],
                            [8.89, 9.27, 8.23, 9.78, 11.64, 11.03, 11.50,
                                11.92, 11.49, 11.08, 11.34, 10.76, 8.89],
                            [2.09, 2.13, 2.29, 2.30, 2.67, 2.45,
                                2.51, 2.40, 2.35, 2.27, 2.24, 2.19, 2.09],
                            [4.82, 4.06, 3.59, 5.27, 9.12, 6.97, 9.17,
                            11.84, 12.41, 11.34, 11.70, 9.69, 4.82]])
    
    ref_pdf = {'single': np.array([[1.90, 1.90, 1.90, 1.90, 1.90, 1.90, 1.90,
                                    1.90, 79.10, 1.90, 1.90, 1.90, 1.90]]),
               'average': np.array([[8.33, 8.33, 8.33, 8.33, 8.33, 8.33, 8.33,
                                     8.33, 8.33, 8.33, 8.33, 8.33, 8.33]])}
    
    @classmethod
    def layout(cls):
        # Wind turbines labelling
        c_n, r_n = 8, 10
        labels = []
        for i in range(1, r_n + 1):
            for j in range(1, c_n + 1):
                l = "c{}_r{}".format(j, i)
                labels.append(l)
        # Wind turbines location generating  wt_c1_r1 = (0., 4500.)
        locations = np.zeros((c_n * r_n, 2))
        num = 0
        for i in range(r_n):
            for j in range(c_n):
                loc_x = 0. + 68.589 * j + 7 * 80. * i
                loc_y = 3911. - j * 558.616
                locations[num, :] = [loc_x, loc_y]
                num += 1
        return np.array(locations)
    
    @classmethod
    def params(cls):
        params = dict()
        
        params["D_r"] = [80.]
        params["z_hub"] = [70.]
        params["v_in"] = [4.]
        params["v_rated"] = [15.]
        params["v_out"] = [25.]
        params["P_rated"] = [2.]  # 2WM
        params["power_curve"] = ["horns"]
        params["ct_curve"] = ["horns"]
        
        return pd.DataFrame(params)
    
    @classmethod
    def pow_curve(cls, vel):
        if vel <= 4.:
            return 0.
        elif vel >= 15.:
            return 2.
        else:
            return 1.45096246e-07 * vel**8 - 1.34886923e-05 * vel**7 + \
                5.23407966e-04 * vel**6 - 1.09843946e-02 * vel**5 + \
                    1.35266234e-01 * vel**4 - 9.95826651e-01 * vel**3 + \
                        4.29176920e+00 * vel**2 - 9.84035534e+00 * vel + \
                            9.14526132e+00

    @classmethod
    def ct_curve(cls, vel):
        if vel <= 10.:
            vel = 10.
        elif vel >= 20.:
            vel = 20.
        return np.array([-2.98723724e-11, 5.03056185e-09, -3.78603307e-07,  1.68050026e-05,
                            -4.88921388e-04,  9.80076811e-03, -1.38497930e-01,  1.38736280e+00,
                            -9.76054549e+00,  4.69713775e+01, -1.46641177e+02,  2.66548591e+02,
                            -2.12536408e+02]).dot(np.array([vel**12, vel**11, vel**10, vel**9,
                                                            vel**8, vel**7, vel**6, vel**5,
                                                            vel**4, vel**3, vel**2, vel, 1.]))


def power_to_cpct(curves, temp='Vesta_2MW'):
    pow_curve, ct_curve = curves
    air_density = 1.225
    generator_efficiency = 1.0
    input_json = f"./{temp}.json"
    with open(input_json, 'r+') as jsonfile:
        turbine_data = json.load(jsonfile)
    radius = turbine_data["turbine"]["properties"]["rotor_diameter"] / 2
    wind_speed = np.array(
        turbine_data["turbine"]["properties"]["power_thrust_table"]["wind_speed"])
    power = np.vectorize(pow_curve)(wind_speed) * 1e6   # change units of MW to W
    cp = 2 * power / (air_density * np.pi * radius ** 2 * generator_efficiency * wind_speed ** 3)
    ct = np.vectorize(ct_curve)(wind_speed)
    
    turbine_data["turbine"]["properties"]["power_thrust_table"]["power"] = list(np.round(cp, 7))
    turbine_data["turbine"]["properties"]["power_thrust_table"]["thrust"] = list(np.round(ct, 7))
    
    export_json = f"./{temp}_new.json"
    with open(export_json,'w+') as dump_f:
        json.dump(turbine_data, dump_f, indent=2)


if __name__ == "__main__":
    
    power_to_cpct((Horns.pow_curve, Horns.ct_curve))