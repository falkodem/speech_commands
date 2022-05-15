import os
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.switch import Switch
from kivy.uix.gridlayout import GridLayout
from kivy.core.window import Window
from kivy.clock import Clock

Window.size = (1000, 400)
Window.clearcolor = (6 / 255, 71 / 255, 57 / 255, 1)


class SysUI(App):
    # define build function
    def __init__(self):
        super().__init__()

        logs_list = ['logs/commands_log', 'logs/answers_log', 'logs/sys_active_log', 'logs/chassis_log']
        for logfile in logs_list:
            try:
                os.remove(logfile)
            except FileNotFoundError:
                pass


        self.swtch_act = Switch(active=False)
        self.swtch_chassis = Switch(active=False)
        self.swtch_rec = Switch(active=False)

        self.switches = GridLayout()
        self.switches.cols = 2
        self.switches.rows = 3
        self.switches.add_widget(Label(text="Система", font_size='25dp', font_name='Roboto-Bold'))
        self.switches.add_widget(self.swtch_act)
        self.switches.add_widget(Label(text="Шасси", font_size='25dp', font_name='Roboto-Bold'))
        self.switches.add_widget(self.swtch_chassis)
        self.switches.add_widget(Label(text="Запись", font_size='25dp', font_name='Roboto-Bold'))
        self.switches.add_widget(self.swtch_rec)

        self.info = BoxLayout()
        self.info.orientation = 'vertical'
        self.curr_command = Label(text='*Команда отсутствует*', font_size='25dp', font_name='Roboto-Bold')
        self.sys_answer = Label(text='*Ответ отсутствует*', font_size='25dp', font_name='Roboto-Bold')
        self.info.add_widget(self.curr_command)
        self.info.add_widget(self.sys_answer)

    def update(self, *args):
        try:
            with open('logs/commands_log', 'r') as f:
                for line in f:
                    pass
                cmd = line
            self.curr_command.text = cmd
        except FileNotFoundError:
            pass

        try:
            with open('logs/answers_log', 'r') as f:
                for line in f:
                    pass
                answr = line
            self.sys_answer.text = answr
        except FileNotFoundError:
            pass

        try:
            with open('logs/sys_active_log', 'r') as f:
                for line in f:
                    pass
                active = line
            self.swtch_act.active = bool(int(active))
        except FileNotFoundError:
            pass

        try:
            with open('logs/chassis_log', 'r') as f:
                for line in f:
                    pass
                active = line
            self.swtch_chassis.active = bool(int(active))
        except FileNotFoundError:
            pass

    def build(self):

        layout = BoxLayout()
        layout.orientation = 'horizontal'

        layout.add_widget(self.info)
        layout.add_widget(self.switches)
        Clock.schedule_interval(self.update, 0.2)
        return layout

SysUI().run()