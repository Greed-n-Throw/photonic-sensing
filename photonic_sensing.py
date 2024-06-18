import matplotlib.pyplot as plt
import numpy as np
import csv
import math
import cmath
import customtkinter
import colorsys
from typing import List
from typing import Tuple


#  _v . _val : value / _l : list / _ll : list of list / _t : tuple / _lt : list of tuples / _b : bool / s : str
#  / nda : ndarray

class ConstVal:
    @staticmethod
    def set_list() -> Tuple[List[float], List[float], List[float], List[float], List[float]]:
        f_i_l = list(np.linspace(0.000001, 0.0001, 11))
        f_r_l = list(np.linspace(0.01, 0.09, 9))
        d_r_l = list(np.linspace(0.020, 0.140, 13))
        d_dr_l = list(np.linspace(0.020, 0.140, 121))
        n_3_l = list(np.linspace(1.0, 1.33, 12))
        return f_i_l, f_r_l, d_r_l, d_dr_l, n_3_l


class Light:
    @staticmethod
    def wave_length_to_rgb(wavelength_val: float) -> Tuple[float, float, float]:
        gamma_v: float = 0.8
        intensity_max_v: int = 255
        factor_v: float = 0.0
        r_v: float = 0.0
        g_v: float = 0.0
        b_v: float = 0.0

        if (wavelength_val >= 0.380) and (wavelength_val < 0.440):
            r_v = -(wavelength_val - 0.440) / (0.440 - 0.380)
            g_v = 0.0
            b_v = 1.0
        elif (wavelength_val >= 0.440) and (wavelength_val < 0.490):
            r_v = 0.0
            g_v = (wavelength_val - 0.440) / (0.490 - 0.440)
            b_v = 1.0
        elif (wavelength_val >= 0.490) and (wavelength_val < 0.510):
            r_v = 0.0
            g_v = 1.0
            b_v = -(wavelength_val - 0.510) / (0.510 - 0.490)
        elif (wavelength_val >= 0.510) and (wavelength_val < 0.580):
            r_v = (wavelength_val - 0.510) / (0.580 - 0.510)
            g_v = 1.0
            b_v = 0.0
        elif (wavelength_val >= 0.580) and (wavelength_val < 0.645):
            r_v = 1.0
            g_v = -(wavelength_val - 0.645) / (0.645 - 0.580)
            b_v = 0.0
        elif (wavelength_val >= 0.645) and (wavelength_val <= 0.750):
            r_v = 1.0
            g_v = 0.0
            b_v = 0.0

        if (wavelength_val >= 0.380) and (wavelength_val < 0.420):
            factor_v = 0.3 + 0.7 * (wavelength_val - 0.380) / (0.420 - 0.380)
        elif (wavelength_val >= 0.420) and (wavelength_val < 0.645):
            factor_v = 1.0
        elif (wavelength_val >= 0.645) and (wavelength_val <= 0.750):
            factor_v = 0.3 + 0.7 * (0.750 - wavelength_val) / (0.750 - 0.645)

        r_v = round(intensity_max_v * (r_v * factor_v) ** gamma_v)
        g_v = round(intensity_max_v * (g_v * factor_v) ** gamma_v)
        b_v = round(intensity_max_v * (b_v * factor_v) ** gamma_v)

        return r_v, g_v, b_v

    @staticmethod
    def rgb_I(f_l: list, I_l: list, n_v: int, lambda_l: List[float]) -> List[Tuple[float, float, float]]:
        r_l: List[float] = []
        g_l: List[float] = []
        b_l: List[float] = []
        for wavelength in lambda_l:
            r_v, g_v, b_v = Light.wave_length_to_rgb(wavelength)
            r_l.append(r_v)
            g_l.append(g_v)
            b_l.append(b_v)
        r_l_normalized: np.ndarray = np.array(r_l) / 255.0
        g_l_normalized: np.ndarray = np.array(g_l) / 255.0
        b_l_normalized: np.ndarray = np.array(b_l) / 255.0

        r_v: float = 0
        g_v: float = 0
        b_v: float = 0
        color_list: list = []
        for k in range(len(f_l)):
            for i in range(n_v):
                r_v = r_v + I_l[k][i] * r_l_normalized[i]
                g_v = g_v + I_l[k][i] * g_l_normalized[i]
                b_v = b_v + I_l[k][i] * b_l_normalized[i]
            r_v = r_v / n_v
            g_v = g_v / n_v
            b_v = b_v / n_v
            color_list.append((r_v, g_v, b_v))
        return color_list

    @staticmethod
    def hsv_circle(target_rgb_lt: List[Tuple[float, float, float]]) -> None:
        #  + opti si on reduit n en se rapprochant du centre
        n_v: int = 360
        e_v: int = 25
        s_v: int = 2000 // e_v
        t_v: int = 0
        target_hsv_lt: List[Tuple[float, float, float]] = []
        prev_x_v = None
        prev_y_v = None

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')

        n2_v: int = len(target_rgb_lt)
        for m in range(n2_v):
            target_hsv_lt.append(colorsys.rgb_to_hsv(target_rgb_lt[m][0], target_rgb_lt[m][1], target_rgb_lt[m][2]))

        for k in range(e_v + 1):
            u: float = int(k / (e_v + 1) * 1000) / 10
            print(str(str(u) + '%'))
            hsv_colors_lt: List[Tuple[float, float, int]] = [(i / n_v, 1 - k / e_v, 1) for i in range(n_v)]
            rgb_colors_lt: List[tuple[float, float, float]] = [colorsys.hsv_to_rgb(*hsv) for hsv in hsv_colors_lt]

            for i, (rgb_color_t, hsv_color_t) in enumerate(zip(rgb_colors_lt, hsv_colors_lt)):
                x_v: float = np.cos(2 * np.pi * i / n_v) * (1 - k / e_v)
                y_v: float = np.sin(2 * np.pi * i / n_v) * (1 - k / e_v)
                ax.scatter(x_v, y_v, c=[rgb_color_t], s=s_v)

                for j in range(n2_v):
                    #  + opti si on .pop les éléments de target_hsv_lt qd trouvé
                    if (hsv_color_t[0] < target_hsv_lt[j][0] + 1 / ((n_v - 1) * 2)) & (
                            hsv_color_t[0] > target_hsv_lt[j][0] - 1 / ((n_v - 1) * 2)) & (
                            hsv_color_t[1] < target_hsv_lt[j][1] + 1 / (e_v * 2)) & (
                            hsv_color_t[1] > target_hsv_lt[j][1] - 1 / (e_v * 2)):
                        if t_v == 0:
                            ax.plot(x_v, y_v, 'k+', markersize=5)
                        elif prev_x_v is not None and prev_y_v is not None:
                            ax.plot([prev_x_v, x_v], [prev_y_v, y_v], 'k-')
                        prev_x_v, prev_y_v = x_v, y_v
                        t_v = 1
                        break


class Material:
    @staticmethod
    def read_csv_on_list(file_name: str, start_v: float, stop_v: float) -> Tuple[List[List[str]], List[List[str]]]:
        with (open(file_name, 'r', encoding='utf-8') as file_csv):
            reader_csv = csv.reader(file_csv)
            data_list: List[List[str]] = []
            data2_list: List[List[str]] = []
            letters_count: int = 0
            for line in reader_csv:
                if letters_count == 0:
                    if line:
                        if any(char.isalpha() for char in line[0]):
                            letters_count += 1
                elif letters_count == 1:
                    if line:
                        if all(char.isdigit() or char == '.' or char == 'e' or char == '-' or char == '+'
                               for char in line[0]):
                            first_column_value: float = float(line[0])
                            if start_v <= first_column_value <= stop_v:
                                data_list.append(line)
                        else:
                            letters_count += 1
                elif letters_count == 2:
                    if line:
                        if all(char.isdigit() or char == '.' or char == 'e' or char == '-' or char == '+'
                               for char in line[0]):
                            first_column_value: float = float(line[0])
                            if start_v <= first_column_value <= stop_v:
                                data2_list.append(line)
        return data_list, data2_list

    @staticmethod
    def make_smooth_dataset(start_v: float, stop_v: float, lambda_l: List[float]) -> Tuple[dict, dict]:
        smooth_dataset_l: list = []
        smooth_dataset_square_l: list = []

        n_pmma = "array/n_pmma.csv"
        n_glass = "array/n_fused_silica.csv"
        n_k_gold = "array/n_k_gold.csv"
        n_k_iron = "array/n_k_iron.csv"
        n_k_copper = "array/n_k_copper.csv"
        n_k_plat = "array/n_k_plat.csv"
        n_k_silver = "array/n_k_silver.csv"

        n_data_pmma_ll, empty0 = Material.read_csv_on_list(n_pmma, start_v, stop_v)
        n_data_glass_ll, empty0 = Material.read_csv_on_list(n_glass, start_v, stop_v)
        n_data_gold_ll, k_data_gold_ll = Material.read_csv_on_list(n_k_gold, start_v, stop_v)
        n_data_iron_ll, k_data_iron_ll = Material.read_csv_on_list(n_k_iron, start_v, stop_v)
        n_data_copper_ll, k_data_copper_ll = Material.read_csv_on_list(n_k_copper, start_v, stop_v)
        n_data_plat_ll, k_data_plat_ll = Material.read_csv_on_list(n_k_plat, start_v, stop_v)
        n_data_silver_ll, k_data_silver_ll = Material.read_csv_on_list(n_k_silver, start_v, stop_v)

        n_square_data_pmma_ll: List[list] = [[sublist[0], float(sublist[1]) ** 2] for sublist in n_data_pmma_ll]

        n_square_data_glass_ll: List[list] = [[sublist[0], float(sublist[1]) ** 2] for sublist in n_data_glass_ll]

        n_square_data_gold_ll: List[list] = [[sublist[0], float(sublist[1]) ** 2] for sublist in n_data_gold_ll]
        k_square_data_gold_ll: List[list] = [[sublist[0], float(sublist[1]) ** 2] for sublist in k_data_gold_ll]

        n_square_data_iron_ll: List[list] = [[sublist[0], float(sublist[1]) ** 2] for sublist in n_data_iron_ll]
        k_square_data_iron_ll: List[list] = [[sublist[0], float(sublist[1]) ** 2] for sublist in k_data_iron_ll]

        n_square_data_copper_ll: List[list] = [[sublist[0], float(sublist[1]) ** 2] for sublist in n_data_copper_ll]
        k_square_data_copper_ll: List[list] = [[sublist[0], float(sublist[1]) ** 2] for sublist in k_data_copper_ll]

        n_square_data_plat_ll: List[list] = [[sublist[0], float(sublist[1]) ** 2] for sublist in n_data_plat_ll]
        k_square_data_plat_ll: List[list] = [[sublist[0], float(sublist[1]) ** 2] for sublist in k_data_plat_ll]

        n_square_data_silver_ll: List[list] = [[sublist[0], float(sublist[1]) ** 2] for sublist in n_data_silver_ll]
        k_square_data_silver_ll: List[list] = [[sublist[0], float(sublist[1]) ** 2] for sublist in k_data_silver_ll]

        dataset = [(n_data_pmma_ll, n_square_data_pmma_ll),
                   (n_data_glass_ll, n_square_data_glass_ll),
                   (n_data_gold_ll, n_square_data_gold_ll),
                   (k_data_gold_ll, k_square_data_gold_ll),
                   (n_data_iron_ll, n_square_data_iron_ll),
                   (k_data_iron_ll, k_square_data_iron_ll),
                   (n_data_copper_ll, n_square_data_copper_ll),
                   (k_data_copper_ll, k_square_data_copper_ll),
                   (n_data_plat_ll, n_square_data_plat_ll),
                   (k_data_plat_ll, k_square_data_plat_ll),
                   (n_data_silver_ll, n_square_data_silver_ll),
                   (k_data_silver_ll, k_square_data_silver_ll)]

        for (data_ll, square_data_ll) in dataset:
            array_data_nda: np.ndarray = np.array(data_ll)
            array_square_data_nda: np.ndarray = np.array(square_data_ll)

            x_data_nda: np.ndarray = array_data_nda[:, 0].astype(float)
            y_data_nda: np.ndarray = array_data_nda[:, 1].astype(float)
            x_square_data_nda: np.ndarray = array_square_data_nda[:, 0].astype(float)
            y_square_data_nda: np.ndarray = array_square_data_nda[:, 1].astype(float)

            data_nda: np.ndarray = np.interp(lambda_l, x_data_nda, y_data_nda)
            square_data_nda: np.ndarray = np.interp(lambda_l, x_square_data_nda, y_square_data_nda)

            smooth_dataset_l.extend([data_nda])
            smooth_dataset_square_l.extend([square_data_nda])

        return ({
                    "pmma_n": smooth_dataset_l[0],
                    "glass_n": smooth_dataset_l[1],
                    "gold_n": smooth_dataset_l[2],
                    "gold_k": smooth_dataset_l[3],
                    "iron_n": smooth_dataset_l[4],
                    "iron_k": smooth_dataset_l[5],
                    "copper_n": smooth_dataset_l[6],
                    "copper_k": smooth_dataset_l[7],
                    "plat_n": smooth_dataset_l[8],
                    "plat_k": smooth_dataset_l[9],
                    "silver_n": smooth_dataset_l[10],
                    "silver_k": smooth_dataset_l[11]},
                {
                    "pmma_n_square": smooth_dataset_square_l[0],
                    "glass_n_square": smooth_dataset_square_l[1],
                    "gold_n_square": smooth_dataset_square_l[2],
                    "gold_k_square": smooth_dataset_square_l[3],
                    "iron_n_square": smooth_dataset_square_l[4],
                    "iron_k_square": smooth_dataset_square_l[5],
                    "copper_n_square": smooth_dataset_square_l[6],
                    "copper_k_square": smooth_dataset_square_l[7],
                    "plat_n_square": smooth_dataset_square_l[8],
                    "plat_k_square": smooth_dataset_square_l[9],
                    "silver_n_square": smooth_dataset_square_l[10],
                    "silver_k_square": smooth_dataset_square_l[11]})


class Calculation:
    @staticmethod
    def n_eff_calc(f_l: List[float], epsilon_h_l: List[float], eta_l: List[float]) -> List[List[float]]:
        n_eff_ll: List[list] = []
        for f_val in f_l:
            n_eff_square_l: List[float] = [(x * (1 + 2 * f_val * y) / (1 - f_val * y)) for x, y in
                                           zip(epsilon_h_l, eta_l)]
            n_eff_l: List[complex] = [cmath.sqrt(x) for x in n_eff_square_l]
            n_eff_ll.append(n_eff_l)
        return n_eff_ll

    @staticmethod
    def k_calc(f_l: List[float], n_eff_ll: List[List[float]]) -> List[List[float]]:
        k_ll: List[List[float]] = []
        for i in range(len(f_l)):
            k_l: List[float] = [np.imag(n) for n in n_eff_ll[i]]
            k_ll.append(k_l)
        return k_ll

    @staticmethod
    def I_calc(f_l: List[float], k0_l: List[float], d_val: float, epsilon_h_l: List[float], eta_l: List[float]) \
            -> List[List[float]]:
        I_ll: List[List[float]] = []
        n_eff_ll: List[List[float]] = Calculation.n_eff_calc(f_l, epsilon_h_l, eta_l)
        k_ll = Calculation.k_calc(f_l, n_eff_ll)
        for i in range(len(f_l)):
            I_l = [np.exp(-2 * x * y * d_val) for x, y in zip(k0_l, k_ll[i])]
            I_ll.append(I_l)
        return I_ll

    @staticmethod
    def reflexion_f(n_3_val: float, f_l: List[float], epsilon_h_l: List[float], eta_l: List[float],
                    dataset_l: List[float], k0_l: List[float], d_val: float) \
            -> Tuple[List[List[float]], List[List[float]]]:
        R_f_ll: List[List[float]] = []
        T_f_ll: List[List[float]] = []
        n_eff_ll: List[List[float]] = Calculation.n_eff_calc(f_l, epsilon_h_l, eta_l)
        for n_l in n_eff_ll:
            r_21: List[float] = [(x - y) / (x + y) for x, y in zip(n_l, dataset_l)]
            t_21: List[float] = [(1 + x) for x in r_21]

            r_12: List[float] = [(y - x) / (x + y) for x, y in zip(n_l, dataset_l)]
            t_12: List[float] = [(1 + x) for x in r_12]

            E: List[complex] = [cmath.exp(2j * x * y * d_val) for x, y in zip(k0_l, n_l)]
            E_1: List[complex] = [cmath.exp(1j * x * y * d_val) for x, y in zip(k0_l, n_l)]

            r_23: List[float] = [(x - n_3_val) / (x + n_3_val) for x in n_l]
            t_23: List[float] = [(1 + x) for x in r_23]

            r: List[float] = [((r1 + ((t1 * r2 * t2) * e2)) / (1 - (r3 * r2) * e2)) for r1, r2, r3, t1, t2, e2 in zip(
                r_12, r_23, r_21, t_12, t_21, E)]

            t: List[float] = [((t1 * t2) * e1) / (1 - (r3 * r2) * e2) for r2, r3, t1, t2, e1, e2 in zip(
                r_23, r_21, t_21, t_23, E_1, E)]

            R_l: List[float] = [abs(x) * abs(x) for x in r]
            R_f_ll.append(R_l)

            T_l: List[float] = [abs(x) * abs(x) for x in t]
            T_f_ll.append(T_l)
        return R_f_ll, T_f_ll

    @staticmethod
    def reflexion_n_3(n_3_l, f_val, epsilon_h_l, eta_l, dataset_l, k0_l, d_val) -> List[List[float]]:
        R_n_3_ll: List[List[float]] = []
        n_eff_ll: List[List[float]] = Calculation.n_eff_calc(f_val, epsilon_h_l, eta_l)
        n_eff_l: List[float] = n_eff_ll[0]

        r_12: List[float] = [(y - x) / (x + y) for x, y in zip(n_eff_l, dataset_l)]
        t_12: List[float] = [(1 + x) for x in r_12]

        r_21: List[float] = [(x - y) / (x + y) for x, y in zip(n_eff_l, dataset_l)]
        t_21: List[float] = [(1 + x) for x in r_21]

        E: List[complex] = [cmath.exp(2j * x * y * d_val) for x, y in zip(k0_l, n_eff_l)]

        for n_3_val in n_3_l:
            r_23: List[float] = [(x + (-n_3_val)) / (x + n_3_val) for x in n_eff_l]

            r: List[float] = [((r1 + ((t1 * r2 * t2) * e2)) / (1 - (r3 * r2) * e2)) for r1, r2, r3, t1, t2, e2 in zip(
                r_12, r_23, r_21, t_12, t_21, E)]

            R_l: List[float] = [abs(x) * abs(x) for x in r]
            R_n_3_ll.append(R_l)

        return R_n_3_ll

    @staticmethod
    def reflexion_d(n_3_val: float, f_val: float, epsilon_h_l: List[float], eta_l: List[float], dataset_l: List[float],
                    k0_l: List[float], d_l: List[float]) -> List[List[float]]:
        R_d_ll: List[List[float]] = []
        n_eff_ll: List[List[float]] = Calculation.n_eff_calc([f_val], epsilon_h_l, eta_l)
        n_eff_l: List[float] = n_eff_ll[0]

        r_12: List[float] = [(y - x) / (x + y) for x, y in zip(n_eff_l, dataset_l)]
        t_12: List[float] = [(1 + x) for x in r_12]

        r_21: List[float] = [(x - y) / (x + y) for x, y in zip(n_eff_l, dataset_l)]
        t_21: List[float] = [(1 + x) for x in r_21]

        for d_val in d_l:
            E: List[complex] = [cmath.exp(2j * x * y * d_val) for x, y in zip(k0_l, n_eff_l)]

            r_23: List[float] = [(x + (-n_3_val)) / (x + n_3_val) for x in n_eff_l]

            r: List[float] = [((r1 + ((t1 * r2 * t2) * e2)) / (1 - (r3 * r2) * e2)) for r1, r2, r3, t1, t2, e2 in
                 zip(r_12, r_23, r_21,
                     t_12, t_21, E)]

            R_l: List[float] = [abs(x) * abs(x) for x in r]
            R_d_ll.append(R_l)

        return R_d_ll

    @staticmethod
    def dynamic_range(f_l: List[float], epsilon_h_l: List[float], eta_l: List[float], dataset_l: List[float],
                      k0_l: List[float], d_l: List[float]) -> Tuple[List[List[float]], List[List[float]]]:
        n_eff_ll: List[List[float]] = Calculation.n_eff_calc(f_l, epsilon_h_l, eta_l)
        crb_i_ll: List[List[float]] = []
        crb_w_ll: List[List[float]] = []

        for i in range(len(f_l)):
            r_21: List[float] = [(x - y) / (x + y) for x, y in zip(n_eff_ll[i], dataset_l)]
            r_12: List[float] = [(y - x) / (x + y) for x, y in zip(n_eff_ll[i], dataset_l)]
            t_12: List[float] = [(1 + x) for x in r_12]
            t_21: List[float] = [(1 + x) for x in r_21]
            r_23_133: List[float] = [(x - 1.33) / (x + 1.33) for x in n_eff_ll[i]]
            r_23_100: List[float] = [(x - 1) / (x + 1) for x in n_eff_ll[i]]

            crb_i_l: List[float] = []
            crb_w_l: List[float] = []
            for d_val in d_l:
                E: List[complex] = [cmath.exp(2j * x * y * d_val) for x, y in zip(k0_l, n_eff_ll[i])]

                r_133: List[float] = [((x1 + ((y1 * x2 * y2) * z2)) / (1 - (x3 * x2) * z2)) for
                                        x1, x2, x3, y1, y2, z2 in zip(r_12, r_23_133, r_21, t_12, t_21, E)]
                R_133_l: List[float] = [abs(x) * abs(x) for x in r_133]
                maximum_133: float = max(R_133_l)

                r_100: List[float] = [((x1 + ((y1 * x2 * y2) * z2)) / (1 - (x3 * x2) * z2)) for
                                        x1, x2, x3, y1, y2, z2 in zip(r_12, r_23_100, r_21, t_12, t_21, E)]
                R_100_l: List[float] = [abs(x) * abs(x) for x in r_100]
                maximum_100: float = max(R_100_l)

                lambda_max_100 = 400 + 0.1 * R_100_l.index(maximum_100)
                lambda_max_133 = 400 + 0.1 * R_133_l.index(maximum_133)

                intensity_dynamic_range = 100 * (maximum_100 - maximum_133)
                crb_i_l.append(intensity_dynamic_range)

                wavelength_dynamic_range = lambda_max_100 - lambda_max_133
                crb_w_l.append(wavelength_dynamic_range)

            crb_i_ll.append(crb_i_l)
            crb_w_ll.append(crb_w_l)

        return crb_i_ll, crb_w_ll

    @staticmethod
    def do_calculation() -> None:
        f_i_list, f_r_list, d_r_list, d_dr_list, n_3_list = ConstVal.set_list()

        (entry_f_s, entry_d_s, entry_n3_s, calcul_type_s, entry_start_s, entry_stop_s, entry_metal_s, entry_host_s,
         entry_substrate_s, color_check_box_s) = (Graphic.create_custom_window())
        print("calcul start")
        start_v: float = float(entry_start_s)
        stop_v: float = float(entry_stop_s)
        N_v: int = int((stop_v - start_v) * 10000 + 1)
        vector_lambda_l: List[float] = list(np.linspace(start_v, stop_v, N_v).astype(float))
        k0_list: List[float] = [2 * math.pi / v for v in vector_lambda_l]

        sd_ll, sds_ll = Material.make_smooth_dataset(start_v, stop_v, list(vector_lambda_l))
        epsilon_h_list: List[float] = []
        n_s_l: List[float] = []
        n_l: List[float] = []
        k_s_l: List[float] = []
        k_l: List[float] = []
        n_substrate_list: List[float] = []

        if entry_host_s == "pmma":
            epsilon_h_list = sds_ll["pmma_n_square"]
        elif entry_host_s == "glass":
            epsilon_h_list = sds_ll["glass_n_square"]

        if entry_metal_s == "gold":
            n_l = sd_ll["gold_n"]
            n_s_l = sds_ll["gold_n_square"]
            k_l = sd_ll["gold_k"]
            k_s_l = sds_ll["gold_k_square"]
        elif entry_metal_s == "iron":
            n_l = sd_ll["iron_n"]
            n_s_l = sds_ll["iron_n_square"]
            k_l = sd_ll["iron_k"]
            k_s_l = sds_ll["iron_k_square"]
        elif entry_metal_s == "copper":
            n_l = sd_ll["copper_n"]
            n_s_l = sds_ll["copper_n_square"]
            k_l = sd_ll["copper_k"]
            k_s_l = sds_ll["copper_k_square"]
        elif entry_metal_s == "plat":
            n_l = sd_ll["plat_n"]
            n_s_l = sds_ll["plat_n_square"]
            k_l = sd_ll["plat_k"]
            k_s_l = sds_ll["plat_k_square"]
        elif entry_metal_s == "silver":
            n_l = sd_ll["silver_n"]
            n_s_l = sds_ll["silver_n_square"]
            k_l = sd_ll["silver_k"]
            k_s_l = sds_ll["silver_k_square"]

        if entry_substrate_s == "glass":
            n_substrate_list = sd_ll["glass_n"]

        epsilon_m_list: List[float] = [X - Y + 2j * x * y for X, Y, x, y in zip(n_s_l, k_s_l, n_l, k_l)]
        eta_list: List[float] = [(x - y) / (x + 2 * y) for x, y in zip(epsilon_m_list, epsilon_h_list)]

        calcul_type: str = calcul_type_s
        if calcul_type == 'I':
            Intensity_ll = Calculation.I_calc(f_i_list, k0_list, float(entry_d_s), epsilon_h_list, eta_list)
            label_l = ["f = " + str(f_i_list[i]) for i in range(len(f_i_list))]
            tlt_str, xlab_str, y_lab_str = "I", "wavelength (nm)", "intensity"
            Graphic.graphic(vector_lambda_l, Intensity_ll, lab_l=label_l, tlt_s=tlt_str, xlab_s=xlab_str,
                            yla_s=y_lab_str)
            if color_check_box_s == "on":
                rgb_clr_lt = Light.rgb_I(f_i_list, Intensity_ll, N_v, vector_lambda_l)
                Light.hsv_circle(rgb_clr_lt)

        elif calcul_type == "R/lambda f var":
            r_f_ll, t_f_ll = Calculation.reflexion_f(float(entry_n3_s), f_r_list, epsilon_h_list, eta_list,
                                                     n_substrate_list, k0_list, float(entry_d_s))
            label_l = ["f = " + str(f_r_list[i]) for i in range(len(f_r_list))]
            tlt_str, xlab_str, y_lab_str = "Reflexion in term of lambda with f variations", "wavelength (nm)", "R"
            tlt2_str, y2_lab_str = "Transmission in term of lambda with f variations", 'T'
            Graphic.graphic(vector_lambda_l, r_f_ll, lab_l=label_l, tlt_s=tlt_str, xlab_s=xlab_str, yla_s=y_lab_str)
            Graphic.graphic(vector_lambda_l, t_f_ll, lab_l=label_l, tlt_s=tlt2_str, xlab_s=xlab_str, yla_s=y2_lab_str)
            if color_check_box_s == "on":
                rgb_clr_lt = Light.rgb_I(f_r_list, t_f_ll, N_v, vector_lambda_l)
                Light.hsv_circle(rgb_clr_lt)

        elif calcul_type == "R/lambda n3 var":
            r_n3_ll = Calculation.reflexion_n_3(n_3_list, [float(entry_f_s)], epsilon_h_list, eta_list,
                                                sd_ll["glass_n"], k0_list, float(entry_d_s))
            label_l = ["n3 = " + str(n_3_list[i]) for i in range(len(n_3_list))]
            tlt_str, xlab_str, y_lab_str = "Reflexion in term of lambda with n3 variations", "wavelength (nm)", "R"
            Graphic.graphic(vector_lambda_l, r_n3_ll, lab_l=label_l, tlt_s=tlt_str, xlab_s=xlab_str, yla_s=y_lab_str)

        elif calcul_type == "R/lambda d var":
            r_d_ll = Calculation.reflexion_d(float(entry_n3_s), float(entry_f_s), epsilon_h_list, eta_list,
                                             sd_ll["glass_n"], k0_list, d_r_list)
            label_l = ["d = " + str(d_r_list[i]) for i in range(len(d_r_list))]
            tlt_str, xlab_str, y_lab_str = "Reflexion in term of lambda with d variations", "wavelength (nm)", "R"
            Graphic.graphic(vector_lambda_l, r_d_ll, lab_l=label_l, tlt_s=tlt_str, xlab_s=xlab_str, yla_s=y_lab_str)

        elif calcul_type == "DRi/d" or calcul_type == "DRw/d":
            Dynamic_i, Dynamic_w = Calculation.dynamic_range(f_r_list, epsilon_h_list, eta_list, n_substrate_list,
                                                             k0_list, d_dr_list)
            label_l = ["f = " + str(f_r_list[i]) for i in range(len(f_r_list))]
            if calcul_type == "DRi/d":
                tlt_str, xlab_str, y_lab_str = "Intensity Dynamic Range", "thickness d", "Intensity"
                Graphic.graphic(d_dr_list, Dynamic_i, lab_l=label_l, tlt_s=tlt_str, xlab_s=xlab_str, yla_s=y_lab_str)
            else:
                tlt_str, xlab_str, y_lab_str = "Wavelength Dynamic Range", "thickness d (μm)", "Wavelength (nm)"
                Graphic.graphic(d_dr_list, Dynamic_w, lab_l=label_l, tlt_s=tlt_str, xlab_s=xlab_str, yla_s=y_lab_str)
        print("calcul end")


class Graphic:
    @staticmethod
    def graphic(x_l: List[float], y_ll: List[List[float]], lab_l: List[str] = None, tlt_s: str = None,
                xlab_s: str = None, yla_s: str = None) -> None:
        plt.figure()
        n_v: int = len(lab_l)
        x_l = [x*1000 for x in x_l]
        for i, y_l in enumerate(y_ll):
            label: str = lab_l[i] if lab_l and i < n_v else None
            plt.plot(x_l, np.real(y_l), label=label)
        plt.title(tlt_s)
        plt.xlabel(xlab_s)
        plt.ylabel(yla_s)
        plt.legend()
        plt.tight_layout()

    @staticmethod
    def create_custom_window() -> Tuple[str, str, str, str, str, str, str, str, str, str]:
        customtkinter.set_appearance_mode("dark")  # Modes: "System" (standard), "Dark", "Light"
        customtkinter.set_default_color_theme("dark-blue")  # Themes: "blue" (standard), "green", "dark-blue"

        root = customtkinter.CTk()
        root.geometry("630x400")

        frame = customtkinter.CTkFrame(master=root)
        frame.pack(pady=30, padx=30, fill="both", expand=True)

        label = customtkinter.CTkLabel(master=frame, text="Choose values", font=("Roboto", 24))
        label.grid(row=0, column=0, pady=12, padx=20, sticky="w")

        entryStart = customtkinter.CTkOptionMenu(master=frame, values=["0.400"])
        entryStart.grid(row=1, column=0, pady=12, padx=20)
        entryStart.set("0.400")

        entryStop = customtkinter.CTkOptionMenu(master=frame, values=["0.700", "1.000"])
        entryStop.grid(row=1, column=1, pady=12, padx=20)
        entryStop.set("0.700")

        entryHost = customtkinter.CTkOptionMenu(master=frame, values=["pmma", "glass"])
        entryHost.grid(row=2, column=0, pady=12, padx=20)
        entryHost.set("pmma")

        entryMetal = customtkinter.CTkOptionMenu(master=frame, values=["gold", "iron", "copper", "plat", "silver"])
        entryMetal.grid(row=2, column=1, pady=12, padx=20)
        entryMetal.set("gold")

        entrySubstrate = customtkinter.CTkOptionMenu(master=frame, values=["glass"])
        entrySubstrate.grid(row=2, column=2, pady=12, padx=20)
        entrySubstrate.set("glass")

        CalculType = customtkinter.CTkOptionMenu(master=frame, values=["I",
                                                                       "R/lambda f var",
                                                                       "R/lambda n3 var",
                                                                       "R/lambda d var",
                                                                       "DRi/d",
                                                                       "DRw/d"])
        CalculType.grid(row=3, column=0, pady=12, padx=20)
        CalculType.set("I")

        entryFv = customtkinter.CTkOptionMenu(master=frame,
                                              values=["0.01", "0.02", "0.03", "0.04", "0.05", "0.06", "0.07",
                                                      "0.08", "0.09", "0.10"])
        entryFv.grid(row=4, column=0, pady=12, padx=20)
        entryFv.set("0.10")

        entryDv = customtkinter.CTkOptionMenu(master=frame,
                                              values=["1000", "10000", "100000", "0.020", "0.080", "0.140"])
        entryDv.grid(row=4, column=1, pady=12, padx=20)
        entryDv.set("0.020")

        entryN3v = customtkinter.CTkOptionMenu(master=frame,
                                               values=["1.00", "1.05", "1.10", "1.15", "1.20", "1.25", "1.30",
                                                       "1.33"])
        entryN3v.grid(row=4, column=2, pady=12, padx=20)
        entryN3v.set("1.00")

        def on_button_click():
            print("Start button clicked")
            root.quit()

        button = customtkinter.CTkButton(master=frame, text="Start", command=on_button_click)
        button.grid(row=5, column=0, pady=12, padx=20)

        def checkbox_event():
            print("checkbox toggled, current value:", check_var.get())

        check_var = customtkinter.StringVar(value="off")
        colorCheckbox = customtkinter.CTkCheckBox(master=frame, text="Circle Color", command=checkbox_event,
                                                  variable=check_var, onvalue="on", offvalue="off")
        colorCheckbox.grid(row=5, column=1, pady=12, padx=20)

        root.mainloop()

        return (entryFv.get(), entryDv.get(), entryN3v.get(), CalculType.get(), entryStart.get(), entryStop.get(),
                entryMetal.get(), entryHost.get(), entrySubstrate.get(), check_var.get())
