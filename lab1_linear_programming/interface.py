import PySimpleGUI as sg
import numpy as np
from data_parser import *

class Interface:
    def __create_window(self):
        layout = [
            [sg.Text('Введите коэффициенты функции цели'), sg.InputText()],
            [sg.Text('Введите количество переменных'), sg.InputText()],
            [sg.Text('Введите 3 равенства в виде a1 a2 ... an = b, где a1 a2 ... an - коэффициенты, а b - правая часть:')],
            [sg.Text('Равенство 1'), sg.InputText()],
            [sg.Text('Равенство 2'), sg.InputText()],
            [sg.Text('Равенство 3'), sg.InputText()],
            [sg.Text('Введите неравенство в виде a1 a2 ... an <= b, где a1 a2 ... an - коэффициенты, а b - правая часть')],
            [sg.InputText()],
            [sg.Text('Введите неравенство в виде a1 a2 ... an >= b, где a1 a2 ... an - коэффициенты, а b - правая часть')],
            [sg.InputText()],
            [sg.Text('Введите индексы переменных, имеющих ограничения на знак(в виде "i1 i2 ... in":)')],
            [sg.InputText()],
            [sg.Submit(), sg.Cancel()]
        ]
        return sg.Window('Lab1 Linear Programming', layout)

    def get_data(self):
        while True:
            window = self.__create_window()
            event, values = window.read()
            if event in (None, 'Exit', 'Cancel'):
                break
            if event in ('Submit'):
                return values




