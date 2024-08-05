import matplotlib.pyplot as plt
import numpy as np
import csv
import math
import cmath
import colorsys
import os
from typing import List
from typing import Tuple
import customtkinter as ctk
from PIL import Image, ImageTk

# variables globales :
alpha: float = 0.0
alpha_degrees: float = 0.0
data: list = []
name: str = ""

#  _v . _val : value / _l : list / _ll : list of list / _t : tuple / _lt : list of tuples / _b : bool / s : str
#  / nda : array


class ConstVal:
    @staticmethod
    def set_list() -> Tuple[List[float], List[float], List[float], List[float], List[float], List[float]]:
        f_i_l = [
                 0.0000001, 0.0000002, 0.0000003, 0.0000004, 0.0000005, 0.0000006, 0.0000007, 0.0000008, 0.0000009,
                 0.000001, 0.000002, 0.000003, 0.000004, 0.000005, 0.000006, 0.000007, 0.000008, 0.000009,
                 0.00001, 0.00002, 0.00003, 0.00004, 0.00005, 0.00006, 0.00007, 0.00008, 0.00009,
                 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009,
                 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009,
                 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1
                 ]
        d_i_l = list(np.linspace(100, 10000, 21))
        f_r_l = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
        d_r_l = list(np.linspace(0.020, 0.140, 13))
        for i in range(len(d_r_l)):
            d_r_l[i] = int(d_r_l[i] * 1000) / 1000
        d_dr_l = list(np.linspace(0.020, 0.140, 61))
        for i in range(len(d_dr_l)):
            d_dr_l[i] = int(d_dr_l[i] * 1000) / 1000
        n_3_l = list(np.linspace(1.0, 1.33, 12))
        for i in range(len(n_3_l)):
            n_3_l[i] = int(n_3_l[i]*100)/100
        return f_i_l, d_i_l, f_r_l, d_r_l, d_dr_l, n_3_l

    @staticmethod
    def set_hsv_list(gold_s, iron_s, copper_s, plat_s, silver_s) -> List[str]:
        hsv_list: List[str] = []
        if gold_s == "on":
            hsv_list.append("gold")
        if iron_s == "on":
            hsv_list.append("iron")
        if copper_s == "on":
            hsv_list.append("copper")
        if plat_s == "on":
            hsv_list.append("plat")
        if silver_s == "on":
            hsv_list.append("silver")
        return hsv_list


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
    def rgb_i(f_or_d_l: list, i_lll: List[List[List[float]]], n_v: int, lambda_l: List[float]) \
            -> List[List[Tuple[float, float, float]]]:
        r_l: List[float] = []
        g_l: List[float] = []
        b_l: List[float] = []
        for wavelength in lambda_l:
            r_v0, g_v0, b_v0 = Light.wave_length_to_rgb(wavelength)
            r_l.append(r_v0)
            g_l.append(g_v0)
            b_l.append(b_v0)
        r_l_normalized: np.ndarray = np.array(r_l) / 255.0
        g_l_normalized: np.ndarray = np.array(g_l) / 255.0
        b_l_normalized: np.ndarray = np.array(b_l) / 255.0

        color_list_list: List[list] = []
        for m in range(len(i_lll)):
            color_list: list = []
            for i in range(len(f_or_d_l)):
                r_v = 0.0
                g_v = 0.0
                b_v = 0.0
                for j in range(n_v):
                    r_v = r_v + i_lll[m][i][j] * r_l_normalized[j]
                    g_v = g_v + i_lll[m][i][j] * g_l_normalized[j]
                    b_v = b_v + i_lll[m][i][j] * b_l_normalized[j]
                r_v = r_v / n_v
                g_v = g_v / n_v
                b_v = b_v / n_v
                color_list.append((r_v, g_v, b_v))
            color_list_list.append(color_list)
        return color_list_list

    @staticmethod
    def hsv_circle(target_rgb_llt: List[List[Tuple[float, float, float]]], hsv_l: List[str], title_s: str) -> None:
        n_v: int = 180
        e_v: int = 20
        s_v: int = 4000 // e_v
        first_element: list = [0, 0, 0, 0, 0]

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')

        hsv_color_l: List = ['k', 'gray', 'brown', 'blue', 'darkblue']

        n1_v: int = len(target_rgb_llt)
        n2_v: int = len(target_rgb_llt[0])

        list_plot: List[List[Tuple[float, float]]] = [[] for _ in range(n1_v)]
        target_hsv_llt: List[List[Tuple[float, float, float]]] = []
        for p in range(n1_v):
            target_hsv_lt: List[Tuple[float, float, float]] = []
            for m in range(n2_v):
                target_hsv_lt.append(colorsys.rgb_to_hsv(np.real(target_rgb_llt[p][m][0]),
                                                         np.real(target_rgb_llt[p][m][1]),
                                                         np.real(target_rgb_llt[p][m][2])))
            target_hsv_llt.append(target_hsv_lt)

        for k in range(e_v + 1):
            u: float = int(k / (e_v + 1) * 1000) / 10
            print(f"{u}%")
            hsv_colors_lt: List[Tuple[float, float, int]] = [(i / n_v, 1 - k / e_v, 1) for i in range(n_v)]
            rgb_colors_lt: List[tuple[float, float, float]] = [colorsys.hsv_to_rgb(*hsv) for hsv in hsv_colors_lt]

            for i, (rgb_color_t, hsv_color_t) in enumerate(zip(rgb_colors_lt, hsv_colors_lt)):
                x_v: float = np.cos(2 * np.pi * i / n_v) * (1 - k / e_v)
                y_v: float = np.sin(2 * np.pi * i / n_v) * (1 - k / e_v)
                ax.scatter(x_v, y_v, c=[rgb_color_t], s=s_v)

                for o in range(n1_v):
                    if all(abs(hsv_color_t[i] - target_hsv_llt[o][0][i]) <= [1 / ((n_v - 1) * 2), 1 / (e_v * 2)][i]
                           for i in range(2)):
                        list_plot[o].append((x_v, y_v))
                        first_element[o] = len(list_plot[o])-1

                    for j in range(1, n2_v):
                        if all(abs(hsv_color_t[i] - target_hsv_llt[o][j][i]) <= [1 / ((n_v - 1) * 2), 1 / (e_v * 2)][i]
                               for i in range(2)):
                            list_plot[o].append((x_v, y_v))
                            break

        for o in range(n1_v):
            for j in range(0, len(list_plot[o])):
                x, y = list_plot[o][j]
                ax.scatter(x, y, color=hsv_color_l[o], marker='o', s=15)

            x, y = list_plot[o][first_element[o]]
            ax.scatter(x, y, color=hsv_color_l[o], marker='+', label=hsv_l[o], s=150)

        ax.set_title(str(title_s))
        ax.legend()


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

        n_pm_ma = "res/array/n_pm_ma.csv"
        n_glass = "res/array/n_fused_silica.csv"
        n_k_gold = "res/array/n_k_gold.csv"
        n_k_iron = "res/array/n_k_iron.csv"
        n_k_copper = "res/array/n_k_copper.csv"
        n_k_plat = "res/array/n_k_plat.csv"
        n_k_silver = "res/array/n_k_silver.csv"

        n_data_pm_ma_ll, empty0 = Material.read_csv_on_list(n_pm_ma, start_v, stop_v)
        n_data_glass_ll, empty0 = Material.read_csv_on_list(n_glass, start_v, stop_v)
        n_data_gold_ll, k_data_gold_ll = Material.read_csv_on_list(n_k_gold, start_v, stop_v)
        n_data_iron_ll, k_data_iron_ll = Material.read_csv_on_list(n_k_iron, start_v, stop_v)
        n_data_copper_ll, k_data_copper_ll = Material.read_csv_on_list(n_k_copper, start_v, stop_v)
        n_data_plat_ll, k_data_plat_ll = Material.read_csv_on_list(n_k_plat, start_v, stop_v)
        n_data_silver_ll, k_data_silver_ll = Material.read_csv_on_list(n_k_silver, start_v, stop_v)

        n_square_data_pm_ma_ll: List[list] = [[sublist[0], float(sublist[1]) ** 2] for sublist in n_data_pm_ma_ll]

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

        dataset = [(n_data_pm_ma_ll, n_square_data_pm_ma_ll),
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
                    "pm_ma_n": smooth_dataset_l[0],
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
                    "pm_ma_n_square": smooth_dataset_square_l[0],
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

    @staticmethod
    def save_list_to_csv():
        directory = 'export_data'
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_path = os.path.join(directory, name)

        with open(file_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerows(data)


class Calculation:
    @staticmethod
    def n_eff_calc(f_l: List[float], epsilon_h_l: List[float], eta_l: List[float]) -> List[List[float]]:
        n_eff_ll: List[list] = []
        for f_val in f_l:
            n_eff_square_l: List[float] = [(eps * (1 + 2 * f_val * eta) / (1 - f_val * eta)) for eps, eta
                                           in zip(epsilon_h_l, eta_l)]
            n_eff_l: List[complex] = [cmath.sqrt(n) for n in n_eff_square_l]
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
    def i_calc_f(f_l: List[float], k0_l: List[float], d_val: float, epsilon_h_l: List[float], eta_ll: List[list[float]],
                 t_val: str, alpha_1_val: float) -> List[List[List[float]]]:
        i_lll: List[List[List[float]]] = []
        for j in range(len(eta_ll)):
            n_eff_ll: List[List[float]] = Calculation.n_eff_calc(f_l, epsilon_h_l, eta_ll[j])
            k_ll = Calculation.k_calc(f_l, n_eff_ll)
            i_ll: List[List[float]] = []
            if t_val == "TE":
                for i in range(len(f_l)):
                    i_l = [np.exp(-2 * k0 * k * d_val) for k0, k in zip(k0_l, k_ll[i])]
                    i_ll.append(i_l)
                i_lll.append(i_ll)
            if t_val == "TM":
                for i in range(len(f_l)):
                    i_l = [np.exp(-2 * k0 * k * d_val * math.cos(alpha_1_val)) for k0, k in zip(k0_l, k_ll[i])]
                    i_ll.append(i_l)
                i_lll.append(i_ll)
        return i_lll

    @staticmethod
    def i_calc_d(f_s: float, k0_l: List[float], d_l: List[float], epsilon_h_l: List[float], eta_ll: List[list[float]],
                 t_val: str, alpha_1_val: float) -> List[List[List[float]]]:
        i_lll: List[List[List[float]]] = []
        for j in range(len(eta_ll)):
            n_eff_ll: List[List[float]] = Calculation.n_eff_calc([f_s], epsilon_h_l, eta_ll[j])
            k_ll = Calculation.k_calc([f_s], n_eff_ll)
            i_ll: List[List[float]] = []

            if t_val == "TE":
                for d in d_l:
                    i_l = [np.exp(-2 * k0 * k * d) for k0, k in zip(k0_l, k_ll[0])]
                    i_ll.append(i_l)
                i_lll.append(i_ll)
            if t_val == "TM":
                for d in d_l:
                    i_l = [np.exp(-2 * k0 * k * d * math.cos(alpha_1_val)) for k0, k in zip(k0_l, k_ll[0])]
                    i_ll.append(i_l)
                i_lll.append(i_ll)
        return i_lll

    @staticmethod
    def reflection_f(n_3_val: float, f_l: List[float], epsilon_h_l: List[float], eta_l: List[float], n_1_l: List[float],
                     k0_l: List[float], d_val: float, t_val: str, alpha_1_val: float) \
            -> Tuple[List[List[float]], List[List[float]]]:
        rr_f_ll: List[List[float]] = []
        tt_f_ll: List[List[float]] = []
        n_eff_ll: List[List[float]] = Calculation.n_eff_calc(f_l, epsilon_h_l, eta_l)

        for n_2_l in n_eff_ll:
            if t_val == "TE":
                r_21: List[float] = [(n2 - n1) / (n2 + n1) for n1, n2 in zip(n_1_l, n_2_l)]
                r_12: List[float] = [(n1 - n2) / (n2 + n1) for n1, n2 in zip(n_1_l, n_2_l)]
                r_23: List[float] = [(n2 - n_3_val) / (n2 + n_3_val) for n2 in n_2_l]
                t_12: List[float] = [(1 + r1) for r1 in r_12]
                t_21: List[float] = [(1 + r2) for r2 in r_21]
                t_23: List[float] = [(1 + r3) for r3 in r_23]
                e: List[complex] = [cmath.exp(2j * k0 * n2 * d_val) for k0, n2 in zip(k0_l, n_2_l)]
                e_1: List[complex] = [cmath.exp(1j * k0 * n2 * d_val) for k0, n2 in zip(k0_l, n_2_l)]

                r_l: List[float] = [((r1 + ((t1 * r3 * t2) * e2)) / (1 - (r2 * r3) * e2))
                                    for r1, r2, r3, t1, t2, e2 in zip(r_12, r_21, r_23, t_12, t_21, e)]
                rr_l: List[float] = [abs(r) * abs(r) for r in r_l]
                rr_f_ll.append(rr_l)

                t_l: List[float] = [((t1 * t2) * e1) / (1 - (r2 * r3) * e2)
                                    for r2, r3, t1, t2, e1, e2 in zip(r_21, r_23, t_21, t_23, e_1, e)]
                tt_l: List[float] = [abs(t) * abs(t) * n_3_val * n_3_val / (n2 * n2) for t, n2 in zip(t_l, n_2_l)]
                tt_f_ll.append(tt_l)

            elif t_val == "TM":
                alpha_2_l = [cmath.asin((n1/n2) * cmath.sin(alpha_1_val)) for n1, n2 in zip(n_1_l, n_2_l)]
                alpha_3_l = [cmath.asin((n2/n_3_val) * cmath.sin(a2)) for n2, a2 in zip(n_2_l, alpha_2_l)]
                alpha_4_l = [cmath.asin((n2/n1) * cmath.sin(a2)) for n1, n2, a2 in zip(n_1_l, n_2_l, alpha_2_l)]

                r_12: List[float] = [(math.cos(alpha_1_val) / n1 - (cmath.cos(a2) / n2)) /
                                     (math.cos(alpha_1_val) / n1 + (cmath.cos(a2) / n2))
                                     for n1, n2, a2 in zip(n_1_l, n_2_l, alpha_2_l)]
                r_21: List[float] = [(cmath.cos(a2)/n2 - cmath.cos(alpha_1_val)/n1) /
                                     (cmath.cos(alpha_1_val)/n1 + cmath.cos(a2)/n2)
                                     for n1, n2, a2 in zip(n_1_l, n_2_l, alpha_4_l)]
                r_23: List[float] = [(cmath.cos(a2)/n2 - cmath.cos(a3)/n_3_val) /
                                     (cmath.cos(a2)/n2 + cmath.cos(a3)/n_3_val)
                                     for n2, a2, a3 in zip(n_2_l, alpha_2_l, alpha_3_l)]
                t_12: List[float] = [(1 + r) for r in r_12]
                t_21: List[float] = [(1 + r) for r in r_21]
                t_23: List[float] = [(1 + r) for r in r_23]
                e: List[complex] = [cmath.exp(2j * k0 * n2 * d_val * cmath.cos(a2)) for k0, n2, a2
                                    in zip(k0_l, n_2_l, alpha_2_l)]
                e_1: List[complex] = [cmath.exp(1j * k0 * n2 * d_val * cmath.cos(a2)) for k0, n2, a2
                                      in zip(k0_l, n_2_l, alpha_2_l)]

                r_l: List[float] = [((r1 + ((t1 * r2 * t2) * e2)) / (1 - (r3 * r2) * e2)) for r1, r2, r3, t1, t2, e2
                                    in zip(r_12, r_23, r_21, t_12, t_21, e)]
                rr_l: List[float] = [abs(r) * abs(r) for r in r_l]
                rr_f_ll.append(rr_l)

                t_l: List[float] = [((t1 * t2) * e1) / (1 - (r2 * r3) * e2) for r2, r3, t1, t2, e1, e2
                                    in zip(r_21, r_23, t_21, t_23, e_1, e)]
                tt_l: List[float] = [abs(t) * abs(t) * n_3_val * n_3_val / (n2 * n2) for t, n2 in zip(t_l, n_2_l)]
                tt_f_ll.append(tt_l)
        return rr_f_ll, tt_f_ll

    @staticmethod
    def reflection_n_3(n_3_l: list, f_val: float, epsilon_h_l: list, eta_l: list, n_1_l: list, k0_l: list,
                       d_val: float, t_val: str, alpha_1_val: float) \
            -> (Tuple)[List[List[float]], List[List[float]]]:
        rr_n_3_ll: List[List[float]] = []
        tt_n3_ll: List[List[float]] = []
        n_eff_ll: List[List[float]] = Calculation.n_eff_calc([f_val], epsilon_h_l, eta_l)
        n_2_l: List[float] = n_eff_ll[0]

        if t_val == "TE":
            r_12: List[float] = [(n1 - n2) / (n2 + n1) for n1, n2 in zip(n_1_l, n_2_l)]
            t_12: List[float] = [(1 + r1) for r1 in r_12]

            r_21: List[float] = [(n2 - n1) / (n2 + n1) for n1, n2 in zip(n_1_l, n_2_l)]
            t_21: List[float] = [(1 + r2) for r2 in r_21]

            e: List[complex] = [cmath.exp(2j * k0 * n2 * d_val) for k0, n2 in zip(k0_l, n_2_l)]
            e_1: List[complex] = [cmath.exp(1j * k0 * n2 * d_val) for k0, n2 in zip(k0_l, n_2_l)]

            for n_3_val in n_3_l:
                r_23: List[float] = [(n2 + (-n_3_val)) / (n2 + n_3_val) for n2 in n_2_l]
                t_23: List[float] = [(1 + r3) for r3 in r_23]

                r_l: List[float] = [((r1 + ((t1 * r3 * t2) * e2)) / (1 - (r2 * r3) * e2)) for r1, r2, r3, t1, t2, e2
                                    in zip(r_12, r_21, r_23, t_12, t_21, e)]
                rr_l: List[float] = [abs(r) * abs(r) for r in r_l]
                rr_n_3_ll.append(rr_l)

                t: List[float] = [((t1 * t2) * e1) / (1 - (r2 * r3) * e2) for r2, r3, t1, t2, e1, e2
                                  in zip(r_21, r_23, t_21, t_23, e_1, e)]
                tt_l: List[float] = [abs(x) * abs(x) * n_3_val * n_3_val / (y * y) for x, y in zip(t, n_2_l)]
                tt_n3_ll.append(tt_l)

        elif t_val == "TM":
            alpha_2_l = [cmath.asin((n1 / n2) * cmath.sin(alpha_1_val)) for n1, n2 in zip(n_1_l, n_2_l)]
            alpha_4_l = [cmath.asin((n2 / n1) * cmath.sin(a2)) for n1, n2, a2 in zip(n_1_l, n_2_l, alpha_2_l)]

            r_12: List[float] = [(math.cos(alpha_1_val) / n1 - (cmath.cos(a2) / n2)) /
                                 (math.cos(alpha_1_val) / n1 + (cmath.cos(a2) / n2))
                                 for n1, n2, a2 in zip(n_1_l, n_2_l, alpha_2_l)]
            r_21: List[float] = [(cmath.cos(a2) / n2 - math.cos(alpha_1_val) / n1) /
                                 (math.cos(alpha_1_val) / n1 + cmath.cos(a2) / n2)
                                 for n1, n2, a2 in zip(n_1_l, n_2_l, alpha_4_l)]

            t_12: List[float] = [(1 + r1) for r1 in r_12]
            t_21: List[float] = [(1 + r2) for r2 in r_21]

            e: List[complex] = [cmath.exp(2j * k0 * n2 * d_val * cmath.cos(a2)) for k0, n2, a2
                                in zip(k0_l, n_2_l, alpha_2_l)]
            e_1: List[complex] = [cmath.exp(1j * k0 * n2 * d_val * cmath.cos(a2)) for k0, n2, a2
                                  in zip(k0_l, n_2_l, alpha_2_l)]

            for n_3_val in n_3_l:
                alpha_3_l = [cmath.asin((n2 / n_3_val) * cmath.sin(a2)) for n2, a2 in zip(n_2_l, alpha_2_l)]
                r_23: List[float] = [(cmath.cos(a2) / n2 - cmath.cos(a3) / n_3_val) /
                                     (cmath.cos(a2) / n2 + cmath.cos(a3) / n_3_val)
                                     for n2, a2, a3 in zip(n_2_l, alpha_2_l, alpha_3_l)]
                t_23: List[float] = [(1 + r3) for r3 in r_23]

                r_l: List[float] = [((r1 + ((t1 * r3 * t2) * e2)) / (1 - (r2 * r3) * e2)) for r1, r2, r3, t1, t2, e2
                                    in zip(r_12, r_21, r_23, t_12, t_21, e)]
                rr_l: List[float] = [abs(r) * abs(r) for r in r_l]
                rr_n_3_ll.append(rr_l)

                t_l: List[float] = [((t1 * t2) * e1) / (1 - (r2 * r3) * e2) for r2, r3, t1, t2, e1, e2
                                    in zip(r_21, r_23, t_21, t_23, e_1, e)]
                tt_l: List[float] = [abs(t) * abs(t) * n_3_val * n_3_val / (n2 * n2) for t, n2 in zip(t_l, n_2_l)]
                tt_n3_ll.append(tt_l)

        return rr_n_3_ll, tt_n3_ll

    @staticmethod
    def reflection_d(n_3_val: float, f_val: float, epsilon_h_l: List[float], eta_l: List[float], n_1_l: List[float],
                     k0_l: List[float], d_l: List[float], t_val: str, alpha_1_val: float) \
            -> Tuple[List[List[float]], List[List[float]]]:
        rr_d_ll: List[List[float]] = []
        tt_d_ll: List[List[float]] = []
        n_eff_ll: List[List[float]] = Calculation.n_eff_calc([f_val], epsilon_h_l, eta_l)
        n_2_l: List[float] = n_eff_ll[0]

        if t_val == "TE":
            r_12: List[float] = [(n1 - n2) / (n2 + n1) for n1, n2 in zip(n_1_l, n_2_l)]
            t_12: List[float] = [(1 + r1) for r1 in r_12]

            r_21: List[float] = [(n2 - n1) / (n2 + n1) for n1, n2 in zip(n_1_l, n_2_l)]
            t_21: List[float] = [(1 + r2) for r2 in r_21]

            r_23: List[float] = [(n2 + (-n_3_val)) / (n2 + n_3_val) for n2 in n_2_l]
            t_23: List[float] = [(1 + r3) for r3 in r_23]

            for d_val in d_l:
                e: List[complex] = [cmath.exp(2j * k0 * n2 * d_val) for k0, n2 in zip(k0_l, n_2_l)]
                e_1: List[complex] = [cmath.exp(1j * k0 * n2 * d_val) for k0, n2 in zip(k0_l, n_2_l)]

                r_l: List[float] = [((r1 + ((t1 * r3 * t2) * e2)) / (1 - (r2 * r3) * e2))
                                    for r1, r2, r3, t1, t2, e2 in zip(r_12, r_21, r_23, t_12, t_21, e)]
                rr_l: List[float] = [abs(r) * abs(r) for r in r_l]
                rr_d_ll.append(rr_l)

                t_l: List[float] = [((t1 * t2) * e1) / (1 - (r2 * r3) * e2) for r2, r3, t1, t2, e1, e2
                                    in zip(r_21, r_23, t_21, t_23, e_1, e)]
                tt_l: List[float] = [abs(t) * abs(t) * n_3_val * n_3_val / (n2 * n2) for t, n2 in zip(t_l, n_2_l)]
                tt_d_ll.append(tt_l)

        elif t_val == "TM":
            alpha_2_l = [cmath.asin((n1 / n2) * cmath.sin(alpha_1_val)) for n1, n2 in zip(n_1_l, n_2_l)]
            alpha_3_l = [cmath.asin((n2 / n_3_val) * cmath.sin(a2)) for n2, a2 in zip(n_2_l, alpha_2_l)]
            alpha_4_l = [cmath.asin((n2 / n1) * cmath.sin(a2)) for n1, n2, a2 in zip(n_1_l, n_2_l, alpha_2_l)]

            r_12: List[float] = [(math.cos(alpha_1_val) / n1 - (cmath.cos(a2) / n2)) /
                                 (math.cos(alpha_1_val) / n1 + (cmath.cos(a2) / n2))
                                 for n1, n2, a2 in zip(n_1_l, n_2_l, alpha_2_l)]
            t_12: List[float] = [(1 + r1) for r1 in r_12]

            r_21: List[float] = [(cmath.cos(a2) / n2 - math.cos(alpha_1_val) / n1) /
                                 (math.cos(alpha_1_val) / n1 + cmath.cos(a2) / n2)
                                 for n1, n2, a2 in zip(n_1_l, n_2_l, alpha_4_l)]
            t_21: List[float] = [(1 + r2) for r2 in r_21]

            r_23: List[float] = [(cmath.cos(a2) / n2 - cmath.cos(a3) / n_3_val) /
                                 (cmath.cos(a2) / n2 + cmath.cos(a3) / n_3_val)
                                 for n2, a2, a3 in zip(n_2_l, alpha_2_l, alpha_3_l)]
            t_23: List[float] = [(1 + r3) for r3 in r_23]

            for d_val in d_l:
                e: List[complex] = [cmath.exp(2j * k0 * n2 * d_val * cmath.cos(a2))
                                    for k0, n2, a2 in zip(k0_l, n_2_l, alpha_2_l)]
                e_1: List[complex] = [cmath.exp(1j * k0 * n2 * d_val * cmath.cos(a2))
                                      for k0, n2, a2 in zip(k0_l, n_2_l, alpha_2_l)]

                r_l: List[float] = [((r1 + ((t1 * r3 * t2) * e2)) / (1 - (r2 * r3) * e2)) for r1, r2, r3, t1, t2, e2
                                    in zip(r_12, r_21, r_23, t_12, t_21, e)]
                rr_l: List[float] = [abs(r) * abs(r) for r in r_l]
                rr_d_ll.append(rr_l)

                t_l: List[float] = [((t1 * t2) * e1) / (1 - (r2 * r3) * e2) for r2, r3, t1, t2, e1, e2
                                    in zip(r_21, r_23, t_21, t_23, e_1, e)]
                tt_l: List[float] = [abs(t) * abs(t) * n_3_val * n_3_val / (n2 * n2) for t, n2 in zip(t_l, n_2_l)]
                tt_d_ll.append(tt_l)

        return rr_d_ll, tt_d_ll

    @staticmethod
    def dynamic_range(f_l: List[float], epsilon_h_l: List[float], eta_l: List[float], n_1_l: List[float],
                      k0_l: List[float], d_l: List[float], t_val: str, alpha_1_val: float) \
            -> (Tuple)[List[List[float]], List[List[float]]]:
        n_eff_ll: List[List[float]] = Calculation.n_eff_calc(f_l, epsilon_h_l, eta_l)
        crb_i_ll: List[List[float]] = []
        crb_w_ll: List[List[float]] = []

        if t_val == "TE":
            for i in range(len(f_l)):
                u: float = int(i / (len(f_l)) * 1000) / 10
                print(str(str(u) + '%'))
                r_21: List[float] = [(n2 - n1) / (n2 + n1) for n1, n2 in zip(n_1_l, n_eff_ll[i])]
                r_12: List[float] = [(n1 - n2) / (n2 + n1) for n1, n2 in zip(n_1_l, n_eff_ll[i])]
                t_12: List[float] = [(1 + r1) for r1 in r_12]
                t_21: List[float] = [(1 + r2) for r2 in r_21]
                r_23_133: List[float] = [(n2 - 1.33) / (n2 + 1.33) for n2 in n_eff_ll[i]]
                r_23_100: List[float] = [(n2 - 1) / (n2 + 1) for n2 in n_eff_ll[i]]

                crb_i_l: List[float] = []
                crb_w_l: List[float] = []
                for d_val in d_l:
                    e: List[complex] = [cmath.exp(2j * k0 * n2 * d_val) for k0, n2 in zip(k0_l, n_eff_ll[i])]

                    r_133: List[float] = [((r1 + ((t1 * r4 * t2) * e2)) / (1 - (r2 * r4) * e2))
                                          for r1, r4, r2, t1, t2, e2 in zip(r_12, r_23_133, r_21, t_12, t_21, e)]
                    rr_133_l: List[float] = [abs(r) * abs(r) for r in r_133]
                    maximum_133: float = max(rr_133_l)

                    r_100: List[float] = [((r1 + ((t1 * r5 * t2) * e2)) / (1 - (r2 * r5) * e2))
                                          for r1, r5, r2, t1, t2, e2 in zip(r_12, r_23_100, r_21, t_12, t_21, e)]
                    rr_100_l: List[float] = [abs(r) * abs(r) for r in r_100]
                    maximum_100: float = max(rr_100_l)

                    lambda_max_100 = 400 + 0.1 * rr_100_l.index(maximum_100)
                    lambda_max_133 = 400 + 0.1 * rr_133_l.index(maximum_133)

                    intensity_dynamic_range = 100 * (maximum_100 - maximum_133)
                    crb_i_l.append(intensity_dynamic_range)

                    wavelength_dynamic_range = lambda_max_100 - lambda_max_133
                    crb_w_l.append(wavelength_dynamic_range)

                crb_i_ll.append(crb_i_l)
                crb_w_ll.append(crb_w_l)

        elif t_val == "TM":
            for i in range(len(f_l)):
                u: float = int(i / (len(f_l)) * 1000) / 10
                print(str(str(u) + '%'))

                alpha_2_l = [cmath.asin((n1 / n2) * cmath.sin(alpha_1_val)) for n1, n2 in zip(n_1_l, n_eff_ll[i])]
                alpha_3_133_l = [cmath.asin((n2 / 1.33) * cmath.sin(a2)) for n2, a2 in zip(n_eff_ll[i], alpha_2_l)]
                alpha_3_100_l = [cmath.asin((n2 / 1.00) * cmath.sin(a2)) for n2, a2 in zip(n_eff_ll[i], alpha_2_l)]
                alpha_4_l = [cmath.asin((n2 / n1) * cmath.sin(a2)) for n1, n2, a2 in zip(n_1_l, n_eff_ll[i], alpha_2_l)]

                r_12: List[float] = [(math.cos(alpha_1_val) / n1 - (cmath.cos(a2) / n2)) /
                                     (math.cos(alpha_1_val) / n1 + (cmath.cos(a2) / n2))
                                     for n1, n2, a2 in zip(n_1_l, n_eff_ll[i], alpha_2_l)]
                r_21: List[float] = [(cmath.cos(a2) / n2 - cmath.cos(alpha_1_val) / n1) /
                                     (cmath.cos(alpha_1_val) / n1 + cmath.cos(a2) / n2)
                                     for n1, n2, a2 in zip(n_1_l, n_eff_ll[i], alpha_4_l)]

                t_12: List[float] = [(1 + r1) for r1 in r_12]
                t_21: List[float] = [(1 + r2) for r2 in r_21]
                r_23_133: List[float] = [(cmath.cos(a2) / n2 - cmath.cos(a3) / 1.33) /
                                     (cmath.cos(a2) / n2 + cmath.cos(a3) / 1.33)
                                     for n2, a2, a3 in zip(n_eff_ll[i], alpha_2_l, alpha_3_133_l)]

                r_23_100: List[float] = [(cmath.cos(a2) / n2 - cmath.cos(a3) / 1.00) /
                                           (cmath.cos(a2) / n2 + cmath.cos(a3) / 1.00)
                                           for n2, a2, a3 in zip(n_eff_ll[i], alpha_2_l, alpha_3_100_l)]

                crb_i_l: List[float] = []
                crb_w_l: List[float] = []
                for d_val in d_l:
                    e: List[complex] = [cmath.exp(2j * k0 * n2 * d_val) for k0, n2 in zip(k0_l, n_eff_ll[i])]

                    r_133: List[float] = [((r1 + ((t1 * r4 * t2) * e2)) / (1 - (r2 * r4) * e2))
                                          for r1, r4, r2, t1, t2, e2 in zip(r_12, r_23_133, r_21, t_12, t_21, e)]
                    rr_133_l: List[float] = [abs(r) * abs(r) for r in r_133]
                    maximum_133: float = max(rr_133_l)

                    r_100: List[float] = [((r1 + ((t1 * r5 * t2) * e2)) / (1 - (r2 * r5) * e2))
                                          for r1, r5, r2, t1, t2, e2 in zip(r_12, r_23_100, r_21, t_12, t_21, e)]
                    rr_100_l: List[float] = [abs(x) * abs(x) for x in r_100]
                    maximum_100: float = max(rr_100_l)

                    lambda_max_100 = 400 + 0.1 * rr_100_l.index(maximum_100)
                    lambda_max_133 = 400 + 0.1 * rr_133_l.index(maximum_133)

                    intensity_dynamic_range = 100 * (maximum_100 - maximum_133)
                    crb_i_l.append(intensity_dynamic_range)

                    wavelength_dynamic_range = lambda_max_100 - lambda_max_133
                    crb_w_l.append(wavelength_dynamic_range)

                crb_i_ll.append(crb_i_l)
                crb_w_ll.append(crb_w_l)

        return crb_i_ll, crb_w_ll

    @staticmethod
    def do_calculation(entry_start_s, entry_stop_s, entry_substrate_s, entry_host_s, entry_gold_s, entry_iron_s,
                       entry_copper_s, entry_plat_s, entry_silver_s, entry_f_s, entry_d_s, entry_n3_s,
                       entry_reflection_s, entry_transmission_s, entry_circle_s, calcul_type_s, entry_dri_s,
                       entry_drw_s, entry_tm_te_s) -> None:

        f_i_list, d_i_list, f_r_list, d_r_list, d_dr_list, n_3_list = ConstVal.set_list()
        hsv_list = ConstVal.set_hsv_list(entry_gold_s, entry_iron_s, entry_copper_s, entry_plat_s, entry_silver_s)

        global data
        data = []

        global name
        name = ""

        print("calcul start")
        start_v: float = float(entry_start_s)
        stop_v: float = float(entry_stop_s)
        n_v: int = int((stop_v - start_v) * 10000 + 1)
        vector_lambda_l: List[float] = list(np.linspace(start_v, stop_v, n_v).astype(float))
        k0_list: List[float] = [2 * math.pi / v for v in vector_lambda_l]

        sd_ll, sds_ll = Material.make_smooth_dataset(start_v, stop_v, list(vector_lambda_l))
        epsilon_h_list: List[float] = []
        n_s_ll: List[list[float]] = []
        n_ll: List[list[float]] = []
        k_s_ll: List[list[float]] = []
        k_ll: List[list[float]] = []
        n_1_list: List[float] = []

        if entry_host_s == "pmma":
            epsilon_h_list = sds_ll["pm_ma_n_square"]
        elif entry_host_s == "glass":
            epsilon_h_list = sds_ll["glass_n_square"]

        if entry_gold_s == "on":
            n_ll.append(sd_ll["gold_n"])
            n_s_ll.append(sds_ll["gold_n_square"])
            k_ll.append(sd_ll["gold_k"])
            k_s_ll.append(sds_ll["gold_k_square"])
        if entry_iron_s == "on":
            n_ll.append(sd_ll["iron_n"])
            n_s_ll.append(sds_ll["iron_n_square"])
            k_ll.append(sd_ll["iron_k"])
            k_s_ll.append(sds_ll["iron_k_square"])
        if entry_copper_s == "on":
            n_ll.append(sd_ll["copper_n"])
            n_s_ll.append(sds_ll["copper_n_square"])
            k_ll.append(sd_ll["copper_k"])
            k_s_ll.append(sds_ll["copper_k_square"])
        if entry_plat_s == "on":
            n_ll.append(sd_ll["plat_n"])
            n_s_ll.append(sds_ll["plat_n_square"])
            k_ll.append(sd_ll["plat_k"])
            k_s_ll.append(sds_ll["plat_k_square"])
        if entry_silver_s == "on":
            n_ll.append(sd_ll["silver_n"])
            n_s_ll.append(sds_ll["silver_n_square"])
            k_ll.append(sd_ll["silver_k"])
            k_s_ll.append(sds_ll["silver_k_square"])

        if entry_substrate_s == "glass":
            n_1_list = sd_ll["glass_n"]
        elif entry_substrate_s == "pmma":
            n_1_list = sd_ll["pm_ma_n"]

        eta_list: List[List[float]] = []
        for i in range(len(n_ll)):
            epsilon_m_list: List[float] = ([ns - ks + 2j * n * k for ns, ks, n, k in zip(
                n_s_ll[i], k_s_ll[i], n_ll[i], k_ll[i])])
            eta_list.append([(eps_m - eps_h) / (eps_m + 2 * eps_h) for eps_m, eps_h in zip(
                epsilon_m_list, epsilon_h_list)])

        calc_type: str = calcul_type_s

        if calc_type == 'I f var':
            intensity_lll = Calculation.i_calc_f(f_i_list, k0_list, float(entry_d_s), epsilon_h_list, eta_list,
                                                 entry_tm_te_s, alpha)
            data = intensity_lll + []
            rgb_clr_llt = Light.rgb_i(f_i_list, intensity_lll, n_v, vector_lambda_l)
            if entry_tm_te_s == "TE":
                title_ = "Color Transmission in function of f which scale linearly from 0.0000001 to 0.1 for TE"
                name = title_+".csv"
                Light.hsv_circle(rgb_clr_llt, hsv_list, title_)
            elif entry_tm_te_s == "TM":
                title_ = ("Color Transmission in function of f which scale linearly from 0.0000001 to 0.1 for TM "
                          "with alpha = " + str(alpha_degrees))
                name = title_+".csv"
                Light.hsv_circle(rgb_clr_llt, hsv_list, title_)

        elif calc_type == 'I d var':
            intensity_lll = Calculation.i_calc_d(float(entry_f_s), k0_list, d_i_list, epsilon_h_list, eta_list,
                                                 entry_tm_te_s, alpha)
            data = intensity_lll + []
            rgb_clr_llt = Light.rgb_i(d_i_list, intensity_lll, n_v, vector_lambda_l)
            if entry_tm_te_s == "TE":
                title_ = "Color Transmission in function of d which scale linearly from 0.1 mm to 1 cm for TE"
                name = title_+".csv"
                Light.hsv_circle(rgb_clr_llt, hsv_list, title_)
            elif entry_tm_te_s == "TM":
                title_ = ("Color Transmission in function of d which scale linearly from 0.1 mm to 1 cm for TM "
                          "with alpha = " + str(alpha_degrees))
                name = title_+".csv"
                Light.hsv_circle(rgb_clr_llt, hsv_list, title_)

        elif calc_type == "R/lambda f var":
            r_f_ll, t_f_ll = Calculation.reflection_f(float(entry_n3_s), f_r_list, epsilon_h_list, eta_list[0],
                                                      n_1_list, k0_list, float(entry_d_s), entry_tm_te_s, alpha)
            label_l = ["f = " + str(f_r_list[i]) for i in range(len(f_r_list))]
            tlt_str_te = ("reflection with f variations for d = " + str(entry_d_s) + " and n3 = " + str(entry_n3_s)
                          + "for TE")
            tlt_str_tm = ("reflection with f variations for d = " + str(entry_d_s) + " and n3 = " + str(entry_n3_s)
                          + "for TM with alpha = " + str(alpha_degrees))
            xla_str = "wavelength (nm)"
            y_lab_str = "R"
            tlt2_str_te = ("Transmission with f variations for d = " + str(entry_d_s) + " and n3 = " + str(entry_n3_s)
                           + "for TE")
            tlt2_str_tm = ("Transmission with f variations for d = " + str(entry_d_s) + " and n3 = " + str(entry_n3_s)
                           + " for TM with alpha = " + str(alpha_degrees))
            y2_lab_str = 'T'
            if entry_reflection_s == "on":
                data = r_f_ll + []
                if entry_tm_te_s == "TE":
                    name = tlt_str_te+".csv"
                    Graphic.graphic(vector_lambda_l, r_f_ll, lab_l=label_l, tlt_s=tlt_str_te, xla_s=xla_str,
                                    yla_s=y_lab_str)
                elif entry_tm_te_s == "TM":
                    name = tlt_str_tm+".csv"
                    Graphic.graphic(vector_lambda_l, r_f_ll, lab_l=label_l, tlt_s=tlt_str_tm, xla_s=xla_str,
                                    yla_s=y_lab_str)
            if entry_transmission_s == "on":
                data = data + t_f_ll
                if entry_tm_te_s == "TE":
                    if entry_reflection_s == "on":
                        name = name + " and " + tlt2_str_te+".csv"
                    else:
                        name = tlt2_str_te+".csv"
                    Graphic.graphic(vector_lambda_l, t_f_ll, lab_l=label_l, tlt_s=tlt2_str_te, xla_s=xla_str,
                                    yla_s=y2_lab_str)
                elif entry_tm_te_s == "TM":
                    if entry_reflection_s == "on":
                        name = name + " and " + tlt2_str_tm+".csv"
                    else:
                        name = tlt2_str_tm+".csv"
                    Graphic.graphic(vector_lambda_l, t_f_ll, lab_l=label_l, tlt_s=tlt2_str_tm, xla_s=xla_str,
                                    yla_s=y2_lab_str)
            if entry_circle_s == 'on':
                title_te = "Color Transmission in function of f which scale linearly from 0.02 to 0.09 for TE"
                title_tm = ("Color Transmission in function of f which scale linearly from 0.02 to 0.09 for TM with "
                            "alpha = ") + str(alpha_degrees)
                rgb_clr_llt = Light.rgb_i(f_r_list, [t_f_ll], n_v, vector_lambda_l)
                if entry_tm_te_s == "TE":
                    Light.hsv_circle(rgb_clr_llt, hsv_list, title_te)
                elif entry_tm_te_s == "TM":
                    Light.hsv_circle(rgb_clr_llt, hsv_list, title_tm)

        elif calc_type == "R/lambda n3 var":
            r_n3_ll, t_n3_ll = Calculation.reflection_n_3(n_3_list, float(entry_f_s), epsilon_h_list, eta_list[0],
                                                          n_1_list, k0_list, float(entry_d_s), entry_tm_te_s, alpha)
            label_l = ["n3 = " + str(n_3_list[i]) for i in range(len(n_3_list))]
            tlt_str_te = ("reflection with n3 variations for f = " + str(entry_f_s) + " and d = " + str(entry_d_s)
                          + "for TE")
            tlt_str_tm = ("reflection with n3 variations for f = " + str(entry_f_s) + " and d = " + str(entry_d_s)
                          + "for TM with alpha = " + str(alpha_degrees))
            xla_str = "wavelength (nm)"
            y_lab_str = "R"
            tlt2_str_te = ("Transmission with n3 variations for f = " + str(entry_f_s) + " and d = " + str(entry_d_s)
                           + "for TE")
            tlt2_str_tm = ("Transmission with n3 variations for f = " + str(entry_f_s) + " and d = " + str(entry_d_s)
                           + "for TM with alpha = " + str(alpha_degrees))
            y2_lab_str = 'T'
            if entry_reflection_s == "on":
                data = r_n3_ll + []
                if entry_tm_te_s == "TE":
                    name = tlt_str_te+".csv"
                    Graphic.graphic(vector_lambda_l, r_n3_ll, lab_l=label_l, tlt_s=tlt_str_te, xla_s=xla_str,
                                    yla_s=y_lab_str)
                elif entry_tm_te_s == "TM":
                    name = tlt_str_tm+".csv"
                    Graphic.graphic(vector_lambda_l, r_n3_ll, lab_l=label_l, tlt_s=tlt_str_tm, xla_s=xla_str,
                                    yla_s=y_lab_str)
            if entry_transmission_s == "on":
                data = data + t_n3_ll
                if entry_tm_te_s == "TE":
                    if entry_reflection_s == "on":
                        name = name + " and " + tlt2_str_te+".csv"
                    else:
                        name = tlt2_str_te+".csv"
                    Graphic.graphic(vector_lambda_l, t_n3_ll, lab_l=label_l, tlt_s=tlt2_str_te, xla_s=xla_str,
                                yla_s=y2_lab_str)
                elif entry_tm_te_s == "TM":
                    if entry_reflection_s == "on":
                        name = name + " and " + tlt2_str_tm+".csv"
                    else:
                        name = tlt2_str_tm+".csv"
                    Graphic.graphic(vector_lambda_l, t_n3_ll, lab_l=label_l, tlt_s=tlt2_str_tm, xla_s=xla_str,
                                    yla_s=y2_lab_str)
            if entry_circle_s == 'on':
                title_te = "Color Transmission in function of n3 which scale linearly from 1.00 to 1.33"
                title_tm = ("Color Transmission in function of n3 which scale linearly from 1.00 to 1.33 for TM "
                            "with alpha = ") + str(alpha_degrees)
                rgb_clr_llt = Light.rgb_i(n_3_list, [t_n3_ll], n_v, vector_lambda_l)
                if entry_tm_te_s == "TE":
                    Light.hsv_circle(rgb_clr_llt, hsv_list, title_te)
                elif entry_tm_te_s == "TM":
                    Light.hsv_circle(rgb_clr_llt, hsv_list, title_tm)

        elif calc_type == "R/lambda d var":
            r_d_ll, t_d_ll = Calculation.reflection_d(float(entry_n3_s), float(entry_f_s), epsilon_h_list, eta_list[0],
                                                      sd_ll["glass_n"], k0_list, d_r_list, entry_tm_te_s, alpha)
            label_l = ["d = " + str(d_r_list[i]) for i in range(len(d_r_list))]
            tlt_str_te = ("reflection with d variations for f =  " + str(entry_f_s) + " and n3 = " + str(entry_n3_s)
                          + "for TE")
            tlt_str_tm = ("reflection with d variations for f = " + str(entry_f_s) + " and n3 = " + str(entry_n3_s)
                          + "for TM with alpha = " + str(alpha_degrees))
            xla_str = "wavelength (nm)"
            y_lab_str = "R"
            tlt2_str_te = ("reflection with d variations for f = " + str(entry_f_s) + " and n3 = " + str(entry_n3_s)
                           + "for TE")
            tlt2_str_tm = ("reflection with d variations for f = " + str(entry_f_s) + " and n3 = " + str(entry_n3_s)
                           + "for TM with alpha = " + str(alpha_degrees))
            y2_lab_str = 'T'
            if entry_reflection_s == "on":
                data = r_d_ll + []
                if entry_tm_te_s == "TE":
                    name = tlt_str_te+".csv"
                    Graphic.graphic(vector_lambda_l, r_d_ll, lab_l=label_l, tlt_s=tlt_str_te, xla_s=xla_str,
                                    yla_s=y_lab_str)
                elif entry_tm_te_s == "TM":
                    name = tlt_str_tm+".csv"
                    Graphic.graphic(vector_lambda_l, r_d_ll, lab_l=label_l, tlt_s=tlt_str_tm, xla_s=xla_str,
                                    yla_s=y_lab_str)
            if entry_transmission_s == "on":
                data = data + t_d_ll
                if entry_tm_te_s == "TE":
                    if entry_reflection_s == "on":
                        name = name + " and " + tlt2_str_te+".csv"
                    else:
                        name = tlt2_str_te+".csv"
                    Graphic.graphic(vector_lambda_l, t_d_ll, lab_l=label_l, tlt_s=tlt2_str_te, xla_s=xla_str,
                                    yla_s=y2_lab_str)
                elif entry_tm_te_s == "TM":
                    if entry_reflection_s == "on":
                        name = name + " and " + tlt2_str_tm+".csv"
                    else:
                        name = tlt2_str_tm+".csv"
                    Graphic.graphic(vector_lambda_l, t_d_ll, lab_l=label_l, tlt_s=tlt2_str_tm, xla_s=xla_str,
                                    yla_s=y2_lab_str)
            if entry_circle_s == 'on':
                title_te = "Color Transmission in function of d which scale linearly from 20nm to 140 nm for TE"
                title_tm = ("Color Transmission in function of d which scale linearly from 20nm to 140 nm for TM "
                            "with alpha = ") + str(alpha_degrees)
                rgb_clr_llt = Light.rgb_i(n_3_list, [t_d_ll], n_v, vector_lambda_l)
                if entry_tm_te_s == "TE":
                    Light.hsv_circle(rgb_clr_llt, hsv_list, title_te)
                elif entry_tm_te_s == "TM":
                    Light.hsv_circle(rgb_clr_llt, hsv_list, title_tm)

        if calcul_type_s == "DR":
            dynamic_i, dynamic_w = Calculation.dynamic_range(f_r_list, epsilon_h_list, eta_list[0], n_1_list,
                                                             k0_list, d_dr_list, entry_tm_te_s, alpha)
            label_l = ["f = " + str(f_r_list[i]) for i in range(len(f_r_list))]
            if entry_dri_s == "on":
                data = dynamic_i + []
                tlt_str_te, xla_str, y_lab_str = "Intensity Dynamic Range for TE", "thickness d", "Intensity"
                tlt_str_tm = "Intensity Dynamic Range for TM with alpha = " + str(alpha_degrees)
                if entry_tm_te_s == "TE":
                    name = tlt_str_te+".csv"
                    Graphic.graphic(d_dr_list, dynamic_i, lab_l=label_l, tlt_s=tlt_str_te, xla_s=xla_str,
                                    yla_s=y_lab_str)
                elif entry_tm_te_s == "TM":
                    name = tlt_str_tm+".csv"
                    Graphic.graphic(d_dr_list, dynamic_i, lab_l=label_l, tlt_s=tlt_str_tm, xla_s=xla_str,
                                    yla_s=y_lab_str)
            if entry_drw_s == "on":
                data = data + dynamic_w
                tlt2_str_te, xla_str, y_lab_str = "Wavelength Dynamic Range", "thickness d (μm)", "Wavelength (nm)"
                tlt2_str_tm = "Wavelength Dynamic Range for TM with alpha = " + str(alpha_degrees)
                if entry_tm_te_s == "TE":
                    if entry_dri_s == "on":
                        name = name + " and " + tlt2_str_te+".csv"
                    else:
                        name = tlt2_str_te+".csv"
                    Graphic.graphic(d_dr_list, dynamic_w, lab_l=label_l, tlt_s=tlt2_str_te, xla_s=xla_str,
                                    yla_s=y_lab_str)
                elif entry_tm_te_s == "TM":
                    if entry_dri_s == "on":
                        name = name + " and " + tlt2_str_tm+".csv"
                    else:
                        name = tlt2_str_tm+".csv"
                    Graphic.graphic(d_dr_list, dynamic_w, lab_l=label_l, tlt_s=tlt2_str_tm, xla_s=xla_str,
                                    yla_s=y_lab_str)

        print("calcul end")


class Graphic:
    @staticmethod
    def graphic(x_l: List[float], y_ll: List[List[float]], lab_l: List[str] = None, tlt_s: str = None,
                xla_s: str = None, yla_s: str = None) -> None:
        plt.figure()
        n_v: int = len(lab_l)
        x_l = [x*1000 for x in x_l]
        for i, y_l in enumerate(y_ll):
            label: str = lab_l[i] if lab_l and i < n_v else None
            plt.plot(x_l, np.real(y_l), label=label)
        plt.title(tlt_s)
        plt.xlabel(xla_s)
        plt.ylabel(yla_s)
        plt.legend()
        plt.tight_layout()

    @staticmethod
    def print_graph():
        print("start printing")
        plt.show()
        print("end printing")

    class Gui(ctk.CTk):
        def __init__(self):
            super().__init__()

            def calcul_button_click() -> None:
                if self.tabview2.get() == "Color in glass":
                    if self.tabview3.get() == "f variations":
                        self.calcul_type = "I f var"
                    elif self.tabview3.get() == "d variations":
                        self.calcul_type = "I d var"
                    Calculation.do_calculation(self.entry_start.get(),
                                               self.entry_stop.get(),
                                               None,
                                               self.entry_color_host.get(),
                                               self.gold_entry_color.get(),
                                               self.iron_entry_color.get(),
                                               self.copper_entry_color.get(),
                                               self.plat_entry_color.get(),
                                               self.silver_entry_color.get(),
                                               self.color_checkbox_value_f.get(),
                                               self.color_checkbox_value_d.get(),
                                               None,
                                               None,
                                               None,
                                               None,
                                               self.calcul_type,
                                               None,
                                               None,
                                               self.te_tm_button.get()
                                               )
                elif self.tabview2.get() == "Sensor":
                    if self.tabview4.get() == "f variations":
                        self.calcul_type = "R/lambda f var"
                        Calculation.do_calculation(self.entry_start.get(),
                                                   self.entry_stop.get(),
                                                   self.entry_sensor_substrate.get(),
                                                   self.entry_sensor_host.get(),
                                                   self.gold_entry_sensor.get(),
                                                   self.iron_entry_sensor.get(),
                                                   self.copper_entry_sensor.get(),
                                                   self.plat_entry_sensor.get(),
                                                   self.silver_entry_sensor.get(),
                                                   None,
                                                   self.sensor_checkbox_f_d.get(),
                                                   self.sensor_checkbox_f_n3.get(),
                                                   self.check_reflection.get(),
                                                   self.check_transmission.get(),
                                                   self.check_circle.get(),
                                                   self.calcul_type,
                                                   None,
                                                   None,
                                                   self.te_tm_button.get()
                                                   )
                    elif self.tabview4.get() == "d variations":
                        self.calcul_type = "R/lambda d var"
                        Calculation.do_calculation(self.entry_start.get(),
                                                   self.entry_stop.get(),
                                                   self.entry_sensor_substrate.get(),
                                                   self.entry_sensor_host.get(),
                                                   self.gold_entry_sensor.get(),
                                                   self.iron_entry_sensor.get(),
                                                   self.copper_entry_sensor.get(),
                                                   self.plat_entry_sensor.get(),
                                                   self.silver_entry_sensor.get(),
                                                   self.sensor_checkbox_d_f.get(),
                                                   None,
                                                   self.sensor_checkbox_d_n3.get(),
                                                   self.check_reflection.get(),
                                                   self.check_transmission.get(),
                                                   self.check_circle.get(),
                                                   self.calcul_type,
                                                   None,
                                                   None,
                                                   self.te_tm_button.get()
                                                   )
                    elif self.tabview4.get() == "n3 variations":
                        self.calcul_type = "R/lambda n3 var"
                        Calculation.do_calculation(self.entry_start.get(),
                                                   self.entry_stop.get(),
                                                   self.entry_sensor_substrate.get(),
                                                   self.entry_sensor_host.get(),
                                                   self.gold_entry_sensor.get(),
                                                   self.iron_entry_sensor.get(),
                                                   self.copper_entry_sensor.get(),
                                                   self.plat_entry_sensor.get(),
                                                   self.silver_entry_sensor.get(),
                                                   self.sensor_checkbox_n3_f.get(),
                                                   self.sensor_checkbox_n3_d.get(),
                                                   None,
                                                   self.check_reflection.get(),
                                                   self.check_transmission.get(),
                                                   self.check_circle.get(),
                                                   self.calcul_type,
                                                   None,
                                                   None,
                                                   self.te_tm_button.get()
                                                   )
                    elif self.tabview4.get() == "Dynamic Range":
                        self.calcul_type = "DR"
                        Calculation.do_calculation(self.entry_start.get(),
                                                   self.entry_stop.get(),
                                                   self.entry_sensor_substrate.get(),
                                                   self.entry_sensor_host.get(),
                                                   self.gold_entry_sensor.get(),
                                                   self.iron_entry_sensor.get(),
                                                   self.copper_entry_sensor.get(),
                                                   self.plat_entry_sensor.get(),
                                                   self.silver_entry_sensor.get(),
                                                   None,
                                                   None,
                                                   None,
                                                   None,
                                                   None,
                                                   None,
                                                   self.calcul_type,
                                                   self.check_dri.get(),
                                                   self.check_drw.get(),
                                                   self.te_tm_button.get()
                                                   )

            def print_button_click() -> None:
                Graphic.print_graph()

            def export_button_click() -> None:
                Material.save_list_to_csv()

            def checkbox_clicked_color(checkbox) -> None:
                if checkbox == self.color_checkbox_1:
                    if (self.iron_entry_color.get() == self.copper_entry_color.get() == self.plat_entry_color.get() ==
                            self.silver_entry_color.get() == "off"):
                        self.gold_entry_color.set("on")
                elif checkbox == self.color_checkbox_2:
                    if (self.gold_entry_color.get() == self.copper_entry_color.get() == self.plat_entry_color.get() ==
                            self.silver_entry_color.get() == "off"):
                        self.iron_entry_color.set("on")
                elif checkbox == self.color_checkbox_3:
                    if (self.gold_entry_color.get() == self.iron_entry_color.get() == self.plat_entry_color.get() ==
                            self.silver_entry_color.get() == "off"):
                        self.copper_entry_color.set("on")
                elif checkbox == self.color_checkbox_4:
                    if (self.gold_entry_color.get() == self.iron_entry_color.get() == self.copper_entry_color.get() ==
                            self.silver_entry_color.get() == "off"):
                        self.plat_entry_color.set("on")
                elif checkbox == self.color_checkbox_5:
                    if (self.gold_entry_color.get() == self.iron_entry_color.get() == self.copper_entry_color.get() ==
                            self.plat_entry_color.get() == "off"):
                        self.silver_entry_color.set("on")

            def checkbox_clicked(checkbox) -> None:
                checkboxes = [self.sensor_checkbox_1, self.sensor_checkbox_2, self.sensor_checkbox_3,
                              self.sensor_checkbox_4, self.sensor_checkbox_5]
                for cb in checkboxes:
                    if cb != checkbox:
                        if cb == self.sensor_checkbox_1:
                            self.gold_entry_sensor.set("off")
                        elif cb == self.sensor_checkbox_2:
                            self.iron_entry_sensor.set("off")
                        elif cb == self.sensor_checkbox_3:
                            self.copper_entry_sensor.set("off")
                        elif cb == self.sensor_checkbox_4:
                            self.plat_entry_sensor.set("off")
                        elif cb == self.sensor_checkbox_5:
                            self.silver_entry_sensor.set("off")
                if checkbox == self.sensor_checkbox_1:
                    if (self.iron_entry_sensor.get() == self.copper_entry_sensor.get() == self.plat_entry_sensor.get()
                            == self.silver_entry_sensor.get() == "off"):
                        self.gold_entry_sensor.set("on")
                elif checkbox == self.sensor_checkbox_2:
                    if (self.gold_entry_sensor.get() == self.copper_entry_sensor.get() == self.plat_entry_sensor.get()
                            == self.silver_entry_sensor.get() == "off"):
                        self.iron_entry_sensor.set("on")
                elif checkbox == self.sensor_checkbox_3:
                    if (self.gold_entry_sensor.get() == self.iron_entry_sensor.get() == self.plat_entry_sensor.get()
                            == self.silver_entry_sensor.get() == "off"):
                        self.copper_entry_sensor.set("on")
                elif checkbox == self.sensor_checkbox_4:
                    if (self.gold_entry_sensor.get() == self.iron_entry_sensor.get() == self.copper_entry_sensor.get()
                            == self.silver_entry_sensor.get() == "off"):
                        self.plat_entry_sensor.set("on")
                elif checkbox == self.sensor_checkbox_5:
                    if (self.gold_entry_sensor.get() == self.iron_entry_sensor.get() == self.copper_entry_sensor.get()
                            == self.plat_entry_sensor.get() == "off"):
                        self.silver_entry_sensor.set("on")

            def image_change() -> None:
                if self.tabview2.get() == "Color in glass" and self.te_tm_button.get() == "TE":
                    self.image = Image.open("res/image/schema_i.png")
                    self.image = self.image.resize((475, 250))
                    self.photo = ImageTk.PhotoImage(self.image)
                    self.label_image = ctk.CTkLabel(self.frame, image=self.photo, text="")
                    self.label_image.grid(row=0, column=0, pady=(25, 0), padx=(25, 0), columnspan=5)
                elif self.tabview2.get() == "Sensor" and self.te_tm_button.get() == "TE":
                    self.image = Image.open("res/image/schema_sensor.png")
                    self.image = self.image.resize((475, 250))
                    self.photo = ImageTk.PhotoImage(self.image)
                    self.label_image = ctk.CTkLabel(self.frame, image=self.photo, text="")
                    self.label_image.grid(row=0, column=0, pady=(25, 0), padx=(25, 0), columnspan=5)
                elif self.tabview2.get() == "Color in glass" and self.te_tm_button.get() == "TM":
                    self.image = Image.open("res/image/schema_i_tm.png")
                    self.image = self.image.resize((475, 250))
                    self.photo = ImageTk.PhotoImage(self.image)
                    self.label_image = ctk.CTkLabel(self.frame, image=self.photo, text="")
                    self.label_image.grid(row=0, column=0, pady=(25, 0), padx=(25, 0), columnspan=5)
                elif self.tabview2.get() == "Sensor" and self.te_tm_button.get() == "TM":
                    self.image = Image.open("res/image/schema_sensor_tm.png")
                    self.image = self.image.resize((475, 250))
                    self.photo = ImageTk.PhotoImage(self.image)
                    self.label_image = ctk.CTkLabel(self.frame, image=self.photo, text="")
                    self.label_image.grid(row=0, column=0, pady=(25, 0), padx=(25, 0), columnspan=5)

            def te_tm_alpha(value=None) -> None:
                def push_alpha() -> None:
                    global alpha
                    global alpha_degrees
                    alpha_degrees = float(self.alpha.get())
                    if alpha_degrees < 0:
                        alpha_degrees = -alpha_degrees
                    if alpha_degrees > 60:
                        alpha_degrees = 60
                    alpha = alpha_degrees * math.pi / 180
                    self.alpha.delete(0, "end")
                    self.alpha.insert(0, str(alpha_degrees))

                if value == "TM":
                    self.alpha = ctk.CTkEntry(master=self.frame, placeholder_text="alpha", height=25, width=40)
                    self.alpha.grid(row=1, column=6, padx=0, pady=0, sticky="sw")

                    self.alpha_button = ctk.CTkButton(master=self.frame, text='degree', border_width=2, height=25,
                                                      width=40, command=push_alpha)
                    self.alpha_button.grid(row=1, column=6, padx=0, pady=0, sticky="se")
                else:
                    if hasattr(self, 'alpha'):
                        self.alpha.grid_remove()
                        del self.alpha
                    if hasattr(self, 'alpha_button'):
                        self.alpha_button.grid_remove()
                        del self.alpha_button

            def te_tm_alpha_and_image_change(value=None):
                te_tm_alpha(value)
                image_change()

            font_name = ("Roboto", 12)
            self.title("CustomTkinter learning")
            self.geometry(f"{850}x{850}")

            self.frame = ctk.CTkFrame(master=self)
            self.frame.pack(pady=25, padx=25, expand=True)

            for i in range(8):
                self.frame.grid_columnconfigure(i, weight=1, minsize=100)

            self.image = Image.open("res/image/schema_i.png")
            self.image = self.image.resize((475, 250))
            self.photo = ImageTk.PhotoImage(self.image)
            self.label_image = ctk.CTkLabel(self.frame, image=self.photo, text="")
            self.label_image.grid(row=0, column=0, pady=(25, 0), padx=(25, 0), columnspan=5)

            self.label_lambda = ctk.CTkLabel(master=self.frame, text="λ : ", font=font_name, width=75)
            self.label_lambda.grid(row=1, column=0, pady=(40, 0), padx=(25, 0), sticky="nsew")

            self.entry_start = ctk.CTkOptionMenu(master=self.frame, values=["0.400"], font=font_name,  width=80)
            self.entry_start.grid(row=1, column=1, pady=(40, 0), padx=10, sticky="nsew")
            self.entry_start.set("0.400")

            self.label_fleche = ctk.CTkLabel(master=self.frame, text="------------------>", font=font_name,  width=80)
            self.label_fleche.grid(row=1, column=2, pady=(40, 0), padx=10, sticky="nsew")

            self.entry_stop = ctk.CTkOptionMenu(master=self.frame, values=["0.700", "1.000"], font=font_name, width=80)
            self.entry_stop.grid(row=1, column=3, pady=(40, 0), padx=10, sticky="nsew")
            self.entry_stop.set("0.700")

            self.label_lambda_signe = ctk.CTkLabel(master=self.frame, text="μm", font=font_name, width=80)
            self.label_lambda_signe.grid(row=1, column=4, pady=(40, 0), padx=0, sticky="nsew")

            self.te_tm_button = ctk.CTkSegmentedButton(master=self.frame, height=25, width=100,
                                                       command=te_tm_alpha_and_image_change)
            self.te_tm_button.grid(row=1, column=6, pady=0, padx=0, sticky="n")
            self.te_tm_button.configure(values=["TE", "TM"])
            self.te_tm_button.set("TE")

            self.tabview = ctk.CTkTabview(master=self.frame, height=100, width=250)
            self.tabview.grid(row=0, column=5, columnspan=3, padx=25, pady=(10, 0), sticky="n")
            self.tabview.add("n1")
            self.tabview.add("n2")
            self.tabview.add("n3")
            self.tabview.add("f")
            self.tabview.add("d")
            self.tabview.add("TE")
            self.tabview.add("TM")

            self.texbox_n1 = ctk.CTkTextbox(self.tabview.tab("n1"), width=250)
            self.texbox_n1.grid(padx=0, pady=0, sticky="w")
            self.texbox_n1.insert("0.0", "n1 is the refraction indice of the substrate, which is a glass")

            self.textbox_n2 = ctk.CTkTextbox(self.tabview.tab("n2"), width=250)
            self.textbox_n2.grid(padx=0, pady=0)
            self.textbox_n2.insert("0.0",
                                   "n2 is the refraction indice of the film, which is "
                                   "an association of an Host and nanoparticles of a Metal")

            self.textbox_n3 = ctk.CTkTextbox(self.tabview.tab("n3"), width=250)
            self.textbox_n3.grid(padx=0, pady=0)
            self.textbox_n3.insert("0.0",
                                   "n3 is the refraction indice of the ambient space, which is "
                                   "beetween air and water, squale with humidity (1.00 -> 1.33)")

            self.textbox_f = ctk.CTkTextbox(self.tabview.tab("f"), width=250)
            self.textbox_f.grid(padx=0, pady=0)
            self.textbox_f.insert("0.0",
                                  "f is the volume fraction of the Metal on it Host, which is a glass or a polymer")

            self.textbox_d = ctk.CTkTextbox(self.tabview.tab("d"), width=250)
            self.textbox_d.grid(padx=0, pady=0)
            self.textbox_d.insert("0.0", "d is the thickness of the film")

            self.textbox_TE = ctk.CTkTextbox(self.tabview.tab("TE"), width=250)
            self.textbox_TE.grid(padx=0, pady=0)
            self.textbox_TE.insert("0.0", "In a TE wave, the electric field is perpendicular to the direction"
                                          " of propagation")

            self.textbox_TM = ctk.CTkTextbox(self.tabview.tab("TM"), width=250)
            self.textbox_TM.grid(padx=0, pady=0)
            self.textbox_TM.insert("0.0", "In a TM wave, the magnetic field is perpendicular to the direction"
                                          " of propagation")

            self.tabview2 = ctk.CTkTabview(master=self.frame, height=400, width=700, command=image_change)
            self.tabview2.grid(row=3, column=0, columnspan=8, padx=50, pady=25, sticky="nsew")

            self.tabview2.add("Color in glass")
            self.tabview2.add("Sensor")
            for i in range(7):
                self.tabview2.tab("Color in glass").grid_columnconfigure(i, weight=1, minsize=100)
                self.tabview2.tab("Sensor").grid_columnconfigure(i, weight=1, minsize=100)

            self.label_color_host = ctk.CTkLabel(self.tabview2.tab("Color in glass"), text="Host  :",
                                                 font=font_name, width=80)
            self.label_color_host.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

            self.entry_color_host = ctk.CTkOptionMenu(self.tabview2.tab("Color in glass"), values=["glass"],
                                                      width=80)
            self.entry_color_host.grid(row=0, column=1, pady=10, padx=10, sticky="nsew")
            self.entry_color_host.set("glass")

            self.label_color_metal = ctk.CTkLabel(self.tabview2.tab("Color in glass"), text="Metal :",
                                                  font=font_name, width=80)
            self.label_color_metal.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

            self.gold_entry_color = ctk.StringVar(value="on")
            self.iron_entry_color = ctk.StringVar(value="off")
            self.copper_entry_color = ctk.StringVar(value="off")
            self.plat_entry_color = ctk.StringVar(value="off")
            self.silver_entry_color = ctk.StringVar(value="off")

            self.color_checkbox_1 = ctk.CTkCheckBox(self.tabview2.tab("Color in glass"), text="Gold",
                                                    command=lambda: checkbox_clicked_color(self.color_checkbox_1),
                                                    variable=self.gold_entry_color, onvalue="on", offvalue="off",
                                                    width=80)
            self.color_checkbox_1.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")
            self.color_checkbox_2 = ctk.CTkCheckBox(self.tabview2.tab("Color in glass"), text="Iron",
                                                    command=lambda: checkbox_clicked_color(self.color_checkbox_2),
                                                    variable=self.iron_entry_color, onvalue="on", offvalue="off",
                                                    width=80)
            self.color_checkbox_2.grid(row=1, column=2, padx=10, pady=10, sticky="nsew")
            self.color_checkbox_3 = ctk.CTkCheckBox(self.tabview2.tab("Color in glass"), text="Copper",
                                                    command=lambda: checkbox_clicked_color(self.color_checkbox_3),
                                                    variable=self.copper_entry_color, onvalue="on", offvalue="off",
                                                    width=80)
            self.color_checkbox_3.grid(row=1, column=3, padx=10, pady=10, sticky="nsew")
            self.color_checkbox_4 = ctk.CTkCheckBox(self.tabview2.tab("Color in glass"), text="Plat",
                                                    command=lambda: checkbox_clicked_color(self.color_checkbox_4),
                                                    variable=self.plat_entry_color, onvalue="on", offvalue="off",
                                                    width=80)
            self.color_checkbox_4.grid(row=1, column=4, padx=10, pady=10, sticky="nsew")
            self.color_checkbox_5 = ctk.CTkCheckBox(self.tabview2.tab("Color in glass"), text="Silver",
                                                    command=lambda: checkbox_clicked_color(self.color_checkbox_5),
                                                    variable=self.silver_entry_color, onvalue="on", offvalue="off",
                                                    width=80)
            self.color_checkbox_5.grid(row=1, column=5, padx=10, pady=10, sticky="nsew")

            self.tabview3 = ctk.CTkTabview(self.tabview2.tab("Color in glass"), height=100, width=300)
            self.tabview3.grid(row=2, column=2, columnspan=3, pady=10, padx=0, sticky="nsew")
            self.tabview3.add("f variations")
            self.tabview3.add("d variations")
            for i in range(3):
                self.tabview3.tab("f variations").grid_columnconfigure(i, weight=1, minsize=100)
                self.tabview3.tab("d variations").grid_columnconfigure(i, weight=1, minsize=100)

            self.label = ctk.CTkLabel(self.tabview3.tab("f variations"), text="d =", font=font_name, width=80)
            self.label.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

            self.color_checkbox_value_d = ctk.CTkOptionMenu(self.tabview3.tab("f variations"),
                                                            values=["100", "1000", "10000"], width=100)
            self.color_checkbox_value_d.grid(row=0, column=1, padx=0, pady=10, sticky="nsew")

            self.label = ctk.CTkLabel(self.tabview3.tab("f variations"), text="μm", font=font_name, width=80)
            self.label.grid(row=0, column=2, padx=10, pady=10, sticky="nsew")

            self.label = ctk.CTkLabel(self.tabview3.tab("d variations"), text="f =", font=font_name, width=80)
            self.label.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

            self.color_checkbox_value_f = ctk.CTkOptionMenu(self.tabview3.tab("d variations"),
                                                            values=["0.0000001", "0.000001", "0.00001", "0.0001",
                                                                    "0.001"], width=100)
            self.color_checkbox_value_f.grid(row=0, column=1, padx=0, pady=10, sticky="nsew")

            self.label = ctk.CTkLabel(self.tabview3.tab("d variations"), text="", font=font_name, width=80)
            self.label.grid(row=0, column=2, padx=10, pady=10, sticky="nsew")

            self.button_color_calcul = ctk.CTkButton(self.tabview2.tab("Color in glass"), text="Calculation", width=80,
                                                     command=calcul_button_click)
            self.button_color_calcul.grid(row=3, column=0, pady=10, padx=10, sticky="nsew")

            self.button_color_print = ctk.CTkButton(self.tabview2.tab("Color in glass"), text="Show", width=75,
                                                    command=print_button_click)
            self.button_color_print.grid(row=3, column=6, pady=10, padx=(10, 15), sticky="nsew")

            self.button_color_print = ctk.CTkButton(self.tabview2.tab("Color in glass"), text="Export data", width=100,
                                                    command=export_button_click)
            self.button_color_print.grid(row=3, column=3, pady=10, padx=0, sticky="nsew")

            self.label_color_substrate = ctk.CTkLabel(self.tabview2.tab("Sensor"), text="Substrate :", font=font_name,
                                                      width=100)
            self.label_color_substrate.grid(row=0, column=5, padx=0, pady=10, sticky="nsew")

            self.entry_sensor_substrate = ctk.CTkOptionMenu(self.tabview2.tab("Sensor"), values=["glass", "pmma"],
                                                            width=80)
            self.entry_sensor_substrate.grid(row=0, column=6, pady=10, padx=10, sticky="nsew")
            self.entry_sensor_substrate.set("glass")

            self.label_sensor_host = ctk.CTkLabel(self.tabview2.tab("Sensor"), text="Host  :", font=font_name, width=80)
            self.label_sensor_host.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

            self.entry_sensor_host = ctk.CTkOptionMenu(self.tabview2.tab("Sensor"), values=["glass", "pmma"], width=80)
            self.entry_sensor_host.grid(row=0, column=1, pady=10, padx=10, sticky="w")
            self.entry_sensor_host.set("pmma")

            self.label_sensor_metal = ctk.CTkLabel(self.tabview2.tab("Sensor"), text="Metal :", font=font_name,
                                                   width=80)
            self.label_sensor_metal.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

            self.gold_entry_sensor = ctk.StringVar(value="on")
            self.iron_entry_sensor = ctk.StringVar(value="off")
            self.copper_entry_sensor = ctk.StringVar(value="off")
            self.plat_entry_sensor = ctk.StringVar(value="off")
            self.silver_entry_sensor = ctk.StringVar(value="off")

            self.sensor_checkbox_1 = ctk.CTkCheckBox(self.tabview2.tab("Sensor"), text="Gold",
                                                     command=lambda: checkbox_clicked(self.sensor_checkbox_1),
                                                     variable=self.gold_entry_sensor, onvalue="on", offvalue="off",
                                                     width=80)
            self.sensor_checkbox_1.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")
            self.sensor_checkbox_2 = ctk.CTkCheckBox(self.tabview2.tab("Sensor"), text="Iron",
                                                     command=lambda: checkbox_clicked(self.sensor_checkbox_2),
                                                     variable=self.iron_entry_sensor, onvalue="on", offvalue="off",
                                                     width=80)
            self.sensor_checkbox_2.grid(row=1, column=2, padx=10, pady=10, sticky="nsew")
            self.sensor_checkbox_3 = ctk.CTkCheckBox(self.tabview2.tab("Sensor"), text="Copper",
                                                     command=lambda: checkbox_clicked(self.sensor_checkbox_3),
                                                     variable=self.copper_entry_sensor, onvalue="on", offvalue="off",
                                                     width=80)
            self.sensor_checkbox_3.grid(row=1, column=3, padx=10, pady=10, sticky="nsew")
            self.sensor_checkbox_4 = ctk.CTkCheckBox(self.tabview2.tab("Sensor"), text="Plat",
                                                     command=lambda: checkbox_clicked(self.sensor_checkbox_4),
                                                     variable=self.plat_entry_sensor, onvalue="on", offvalue="off",
                                                     width=80)
            self.sensor_checkbox_4.grid(row=1, column=4, padx=10, pady=10, sticky="nsew")
            self.sensor_checkbox_5 = ctk.CTkCheckBox(self.tabview2.tab("Sensor"), text="Silver",
                                                     command=lambda: checkbox_clicked(self.sensor_checkbox_5),
                                                     variable=self.silver_entry_sensor, onvalue="on", offvalue="off",
                                                     width=80)
            self.sensor_checkbox_5.grid(row=1, column=5, padx=10, pady=10, sticky="nsew")

            self.tabview4 = ctk.CTkTabview(self.tabview2.tab("Sensor"), height=150, width=399)
            self.tabview4.grid(row=2, column=1, columnspan=5, pady=10, padx=(50, 51), sticky="nsew")
            self.tabview4.add("f variations")
            self.tabview4.add("d variations")
            self.tabview4.add("n3 variations")
            self.tabview4.add("Dynamic Range")

            for i in range(3):
                self.tabview4.tab("f variations").grid_columnconfigure(i, weight=1, minsize=133)
                self.tabview4.tab("d variations").grid_columnconfigure(i, weight=1, minsize=133)
                self.tabview4.tab("n3 variations").grid_columnconfigure(i, weight=1, minsize=133)
                self.tabview4.tab("Dynamic Range").grid_columnconfigure(i, weight=1, minsize=133)

            self.label = ctk.CTkLabel(self.tabview4.tab("f variations"), text="d =", font=font_name, width=113)
            self.label.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

            self.sensor_checkbox_f_d = ctk.CTkOptionMenu(self.tabview4.tab("f variations"), values=["0.020", "0.080",
                                                                                                    "0.140"], width=113)
            self.sensor_checkbox_f_d.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

            self.label = ctk.CTkLabel(self.tabview4.tab("f variations"), text="μm", font=font_name, width=113)
            self.label.grid(row=0, column=2, padx=10, pady=10, sticky="nsew")

            self.label = ctk.CTkLabel(self.tabview4.tab("f variations"), text="n3 =", font=font_name, width=113)
            self.label.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

            self.sensor_checkbox_f_n3 = ctk.CTkOptionMenu(self.tabview4.tab("f variations"),
                                                          values=["1", "1.05", "1.10", "1.15", "1.20", "1.25", "1.30",
                                                                  "1.33"], width=113)
            self.sensor_checkbox_f_n3.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")

            self.label = ctk.CTkLabel(self.tabview4.tab("f variations"), text="", font=font_name, width=113)
            self.label.grid(row=1, column=2, padx=10, pady=10, sticky="nsew")

            self.label = ctk.CTkLabel(self.tabview4.tab("d variations"), text="f =", font=font_name, width=113)
            self.label.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

            self.sensor_checkbox_d_f = ctk.CTkOptionMenu(self.tabview4.tab("d variations"),
                                                         values=["0.01", "0.03", "0.05", "0.07", "0.09"],
                                                         width=113)
            self.sensor_checkbox_d_f.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

            self.label = ctk.CTkLabel(self.tabview4.tab("d variations"), text="", font=font_name, width=113)
            self.label.grid(row=0, column=2, padx=10, pady=10, sticky="nsew")

            self.label = ctk.CTkLabel(self.tabview4.tab("d variations"), text="n3 =", font=font_name, width=113)
            self.label.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

            self.sensor_checkbox_d_n3 = ctk.CTkOptionMenu(self.tabview4.tab("d variations"),
                                                          values=["1", "1.05", "1.10", "1.15", "1.20", "1.25", "1.30",
                                                                  "1.33"], width=113)
            self.sensor_checkbox_d_n3.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")

            self.label = ctk.CTkLabel(self.tabview4.tab("d variations"), text="", font=font_name, width=113)
            self.label.grid(row=1, column=2, padx=10, pady=10, sticky="nsew")

            self.label = ctk.CTkLabel(self.tabview4.tab("n3 variations"), text="f =", font=font_name, width=113)
            self.label.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

            self.sensor_checkbox_n3_f = ctk.CTkOptionMenu(self.tabview4.tab("n3 variations"),
                                                          values=["0.01", "0.03", "0.05", "0.07", "0.09"], width=113)
            self.sensor_checkbox_n3_f.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

            self.label = ctk.CTkLabel(self.tabview4.tab("n3 variations"), text="", font=font_name, width=113)
            self.label.grid(row=0, column=2, padx=10, pady=10, sticky="nsew")

            self.label = ctk.CTkLabel(self.tabview4.tab("n3 variations"), text="d =", font=font_name, width=113)
            self.label.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

            self.sensor_checkbox_n3_d = ctk.CTkOptionMenu(self.tabview4.tab("n3 variations"),
                                                          values=["0.020", "0.080", "0.140"], width=113)
            self.sensor_checkbox_n3_d.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")

            self.label = ctk.CTkLabel(self.tabview4.tab("n3 variations"), text="μm", font=font_name, width=113)
            self.label.grid(row=1, column=2, padx=10, pady=10, sticky="nsew")

            self.check_dri = ctk.StringVar(value="off")
            self.check_drw = ctk.StringVar(value="off")

            self.label = ctk.CTkLabel(self.tabview4.tab("Dynamic Range"), text="", font=font_name, width=133)
            self.label.grid(row=0, column=0, padx=0, pady=10, sticky="nsew")

            self.sensor_checkbox_dri = ctk.CTkCheckBox(self.tabview4.tab("Dynamic Range"), text="Intensity Dynamic Range",
                                                       variable=self.check_dri, onvalue="on", offvalue="off")
            self.sensor_checkbox_dri.grid(row=0, column=0, columnspan=2, padx=(25, 0), pady=10, sticky="nsew")

            self.label = ctk.CTkLabel(self.tabview4.tab("Dynamic Range"), text="", font=font_name, width=133)
            self.label.grid(row=0, column=2, padx=0, pady=10, sticky="nsew")

            self.label = ctk.CTkLabel(self.tabview4.tab("Dynamic Range"), text="", font=font_name, width=133)
            self.label.grid(row=1, column=0, padx=0,  pady=10, sticky="nsew")

            self.sensor_checkbox_drw = ctk.CTkCheckBox(self.tabview4.tab("Dynamic Range"), text="Wavelength Dynamic Range",
                                                       variable=self.check_drw, onvalue="on", offvalue="off")
            self.sensor_checkbox_drw.grid(row=1, column=0, columnspan=2, padx=(25, 0), pady=10, sticky="nsew")

            self.label = ctk.CTkLabel(self.tabview4.tab("Dynamic Range"), text="", font=font_name, width=133)
            self.label.grid(row=1, column=2, padx=0, pady=10, sticky="nsew")

            self.check_reflection = ctk.StringVar(value="off")
            self.check_transmission = ctk.StringVar(value="off")
            self.check_circle = ctk.StringVar(value="off")

            self.sensor_checkbox_11 = ctk.CTkCheckBox(self.tabview2.tab("Sensor"), text="Reflection",
                                                      variable=self.check_reflection, onvalue="on", offvalue="off",
                                                      width=100)
            self.sensor_checkbox_11.grid(row=3, column=2, padx=0, pady=0, sticky="nsew")

            self.sensor_checkbox_22 = ctk.CTkCheckBox(self.tabview2.tab("Sensor"), text="Transmission",
                                                      variable=self.check_transmission, onvalue="on", offvalue="off",
                                                      width=100)
            self.sensor_checkbox_22.grid(row=3, column=3, columnspan=2, padx=0, pady=0, sticky="nsew")

            self.sensor_checkbox_3 = ctk.CTkCheckBox(self.tabview2.tab("Sensor"), text="Color Circle",
                                                     variable=self.check_circle, onvalue="on", offvalue="off",
                                                     width=90)
            self.sensor_checkbox_3.grid(row=3, column=4, columnspan=2, padx=(10, 0), pady=0, sticky="nsew")

            self.button_sensor_calcul = ctk.CTkButton(self.tabview2.tab("Sensor"), text="Calculation", width=80,
                                                      command=calcul_button_click)
            self.button_sensor_calcul.grid(row=4, column=0, pady=(0, 10), padx=10, sticky="nsew")

            self.button_sensor_print = ctk.CTkButton(self.tabview2.tab("Sensor"), text="Show", width=75,
                                                     command=print_button_click)
            self.button_sensor_print.grid(row=4, column=6, pady=(0, 10), padx=(10, 15), sticky="nsew")

            self.button_color_print = ctk.CTkButton(self.tabview2.tab("Sensor"), text="Export data", width=100,
                                                    command=export_button_click)
            self.button_color_print.grid(row=4, column=3, pady=10, padx=0, sticky="nsew")
