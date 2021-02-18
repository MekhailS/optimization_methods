import PySimpleGUI as sg



class Interface:

    def __init__(self):
        self.method = None

    def display_result(self, result):
        if self.method is None:
            return None

        sg.popup("Result of the method ' " + str(self.method) + " ' " + str(result))

    @staticmethod
    def display_error():
        sg.popup_error("There is some problem with input data or method")

    def __create_window(self):
        layout = [
            [sg.Text('Выберите метод', size=(15, 1)), sg.InputCombo(('simplex', 'bruteforce'), size=(30, 3))],
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

